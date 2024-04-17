import pandas as pd
import ast 
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torch.nn as nn
import numpy as np
import imageio
from sklearn.model_selection import train_test_split
from einops.layers.torch import Rearrange
from einops import rearrange
from torchvision.transforms import Resize, Normalize, ToTensor, Compose
import imageio
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

    
# Example of creating the dataset
videos_dir = "./Videos"


def load_data_from_directory(directory):
    csv_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.csv')]
    list_of_dfs = [pd.read_csv(file) for file in csv_files]
    for df in list_of_dfs:
        df['FileName'] = df['FileName'].str.split('_').str[0]  # Assuming all CSVs need this operation
    combined_df = pd.concat(list_of_dfs, ignore_index=True)
    return combined_df

def extract_tools(tool_info):
    # Check if tool_info is not NaN and is a string (or other suitable type for literal_eval)
    if pd.notna(tool_info) and isinstance(tool_info, str):
        try:
            tool_dicts = ast.literal_eval(tool_info)  # Convert string representation of list to actual list
            return [tool['class'] for tool in tool_dicts]
        except Exception as e:
            # You might want to log or handle the specific exception depending on your needs
            print(f"Error parsing tool_info: {tool_info} with error {e}")
    return []

def tools_to_vector(tools):
    return [1 if tool in tools else 0 for tool in all_tools]

# Directory containing CSV files
csv_directory = './Cataract_Tools'
df = load_data_from_directory(csv_directory)
print(df.head())

# Preprocess the DataFrame as previously
df['Tool Names'] = df['Tool bounding box'].apply(extract_tools)
all_tools = sorted(set(tool for sublist in df['Tool Names'] for tool in sublist))
df['Tools'] = df['Tool Names'].apply(tools_to_vector)
print(df.head())

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=768):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            # Break the image into patches and flatten them
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, emb_size)
        )

    def forward(self, x):
        x = self.projection(x)
        return x

class ViViT(nn.Module):
    def __init__(self, num_frames, num_classes, image_size=224, patch_size=16, emb_size=768, depth=6, heads=8, mlp_dim=1024):
        super().__init__()
        self.patch_embedding = PatchEmbedding(patch_size=patch_size, emb_size=emb_size)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positional_embedding = nn.Parameter(torch.randn((image_size // patch_size) ** 2 + 1, emb_size))
        self.temporal_embedding = nn.Parameter(torch.randn(num_frames, emb_size))

        self.transformer = nn.Transformer(emb_size, nhead=heads, num_encoder_layers=depth, dim_feedforward=mlp_dim)

        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.Linear(emb_size, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, frames, channels, height, width)
        b, t, c, h, w = x.shape
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.patch_embedding(x)
        x = rearrange(x, '(b t) n d -> b (t n) d', t=t)
        
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.positional_embedding
        x += rearrange(self.temporal_embedding, 't d -> 1 t d')
        
        x = rearrange(x, 'b n d -> n b d')
        x = self.transformer(x)
        x = rearrange(x, 'n b d -> b n d')
        
        x = x[:, 0]
        
        x = self.to_cls_token(x)
        x = self.mlp_head(x)
        
        return x


class VideoDataset(Dataset):
    def __init__(self, video_files, annotations, transform=None):
        self.video_files = video_files
        self.annotations = annotations
        self.transform = transform

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        reader = imageio.get_reader(video_path)
        frames = []
        try:
            for frame in reader:
                # Convert frame to RGB if needed, imageio provides frames as numpy arrays
                if frame.ndim == 3 and frame.shape[2] == 3:  # Check if frame is already in RGB
                    image = Image.fromarray(frame)
                else:
                    image = Image.fromarray(frame[:, :, :3])

                if self.transform:
                    image = self.transform(image)
                frames.append(image)
        except Exception as e:
            print(f"Failed to process video {video_path}: {str(e)}")

        reader.close()

        if len(frames) == 0:
            print(f"No frames were read from the video: {video_path}")
            return torch.tensor([]), torch.tensor([])  # Handling failure

        frames_tensor = torch.stack(frames)
        if idx in self.annotations:
            tools_vector = torch.tensor(self.annotations[idx], dtype=torch.float32)
        else:
            print(f"Warning: Fallback for missing annotation at index {idx}.")
            tools_vector = torch.zeros(10, dtype=torch.float32)

        return frames_tensor, tools_vector

# Example transform (adjust as needed)
transform = Compose([
    Resize((256, 256)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard ImageNet normalization
])

# Splitting the data into training, validation, and testing
video_ids = df['FileName'].unique()
train_ids, test_ids = train_test_split(video_ids, test_size=4, random_state=42)  
train_ids, val_ids = train_test_split(train_ids, test_size=2/8, random_state=42) 


train_df = df[df['FileName'].isin(train_ids)]
val_df = df[df['FileName'].isin(val_ids)]
test_df = df[df['FileName'].isin(test_ids)]

def verify_videos(video_files):
    verified_videos = []
    for video_file in video_files:
        try:
            # Attempt to get a reader and read the first frame
            with imageio.get_reader(video_file) as reader:
                _ = reader.get_next_data()  # Try to read the first frame
                verified_videos.append(video_file)  # If successful, add to verified list
        except Exception as e:
            print(f"Failed to open or read video file {video_file}: {e}")
    return verified_videos

# Helper function to extract paths and annotations from DataFrame
def find_file_with_extension(base_path, filename, extensions):
    for ext in extensions:
        full_path = os.path.join(base_path, f"{filename}{ext}")
        if os.path.exists(full_path):
            return full_path
    return None

def is_empty(tools):
    """ Check if the tools data is empty. """
    if isinstance(tools, list):
        return not tools
    elif isinstance(tools, np.ndarray):
        return tools.size == 0
    elif pd.isna(tools):
        return True
    else:
        return not bool(tools)  # Fallback for other types, using bool conversion

def extract_info(df):
    video_files = []
    annotations = {}
    for name in df['FileName'].unique():
        file_path = find_file_with_extension(videos_dir, name, ['.mp4', '.mov'])
        if file_path:
            tools_data = df[df['FileName'] == os.path.splitext(os.path.basename(file_path))[0]].iloc[0]
            if is_empty(tools_data['Tools']):
                print(f"Skipping {name} due to missing annotations.")
            else:
                video_files.append(file_path)
                annotations[file_path] = tools_data['Tools']
        else:
            print(f"No valid file found for {name} with any of the checked extensions")

    video_files = verify_videos(video_files)
    return video_files, annotations



train_files, train_annotations = extract_info(train_df)
val_files, val_annotations = extract_info(val_df)
test_files, test_annotations = extract_info(test_df)

# Create datasets
train_dataset = VideoDataset(train_files, train_annotations, transform=transform)
val_dataset = VideoDataset(val_files, val_annotations, transform=transform)
test_dataset = VideoDataset(test_files, test_annotations, transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


from transformers import VivitModel
# Assume the model and necessary imports are defined
# model = ViViT(num_frames=16, num_classes=len(df.iloc[0]['Tools']), image_size=256, patch_size=16)


class CustomVivit(nn.Module):
    def __init__(self, num_labels):
        super(CustomVivit, self).__init__()
        self.vivit = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400")
        self.dropout = nn.Dropout(0.5)  # Optional: to mitigate overfitting
        self.classifier = nn.Linear(self.vivit.config.hidden_size, num_labels)  # Adjust according to the number of tools
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for multi-label classification

    def forward(self, inputs):
        outputs = self.vivit(inputs)  # Get the base model outputs
        x = self.dropout(outputs.pooler_output)  # Use pooled output for classification
        x = self.classifier(x)  # Get raw scores for each class
        x = self.sigmoid(x)  # Convert to probabilities per class
        return x

# Initialize model with the number of labels/tools
num_labels = len(all_tools)  # Ensure you have defined all_tools array correctly

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs, rank):
    for epoch in range(num_epochs):
        print(epoch)
        train_loader.sampler.set_epoch(epoch)
        model.train()
        total_loss = 0
        for frames, labels in train_loader:
            frames, labels = frames.to(rank), labels.to(rank)
            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if rank == 0:  # Only log on the main process
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for frames, labels in val_loader:
                    frames, labels = frames.to(rank), labels.to(rank)
                    outputs = model(frames)
                    val_loss += criterion(outputs, labels).item()
            print(f'Epoch {epoch}: Train Loss {total_loss / len(train_loader)}, Val loss={val_loss / len(val_loader)}')

def evaluate_model(model, test_loader, criterion, rank):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for frames, labels in test_loader:
            frames, labels = frames.to(rank), labels.to(rank)
            outputs = model(frames)
            test_loss += criterion(outputs, labels).item()
    if rank == 0:
        print(f'Test Loss: {test_loss / len(test_loader)}')

def main(rank, world_size):
    setup(rank, world_size)

    # Model and DataLoader setup
    #model = ViViT(num_frames=16, num_classes=len(train_annotations[0]), image_size=256, patch_size=16).to(rank)
    model = CustomVivit(num_labels)
    model = DDP(model, device_ids=[rank])
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # DataLoaders with DistributedSampler
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=2, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=1, sampler=val_sampler)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  

    train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=5, rank=rank)
    evaluate_model(model, test_loader, criterion, rank)

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)