import pandas as pd
import ast
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_video
from PIL import Image
import os
import torch.nn as nn
import numpy as np
import imageio
from sklearn.model_selection import train_test_split
from torchvision.transforms import Resize, Normalize, ToTensor, Compose
import imageio
import torch
from torchvision.transforms.functional import pad
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
from transformers import VivitModel
from torchvision import transforms


def load_data_from_directory(directory):
    csv_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.csv')]
    list_of_dfs = [pd.read_csv(file) for file in csv_files]
    for df in list_of_dfs:
        frame_index_with_extention = df['FileName'].str.split('_').str[2]
        df['frame_index'] = frame_index_with_extention.str.split('.').str[0]
        df['FileName'] = df['FileName'].str.split('_').str[0]  
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


# Preprocess the DataFrame as previously
df['Tool Names'] = df['Tool bounding box'].apply(extract_tools)
all_tools = sorted(set(tool for sublist in df['Tool Names'] for tool in sublist))
df['Tools'] = df['Tool Names'].apply(tools_to_vector)

print(df.columns)
del df['Unnamed: 0']
del df['Predicted Labels']
del df['Phase']
del df['Label Title']
del df['Tool Bounding Box']
del df['Tool bounding box']
print(df.head())
videos_dir = './Videos'


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.frames_per_clip = 32

    def __len__(self):
        return len(self.dataframe)
    
    def find_video_path(self, base_path):
        """Check for different video file extensions."""
        for ext in ['.mp4', '.mov']:
            if os.path.exists(base_path + ext):
                return base_path + ext
        raise FileNotFoundError(f"No video file found for {base_path} with extensions .mp4 or .mov")

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        video_filename = row['FileName']
        base_video_path = os.path.join(videos_dir, video_filename)
        video_path = self.find_video_path(base_video_path)
        print(video_path)
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        time_recorded = row['Time Recorded']
        start_pts = max(time_recorded - 5, 0)
        end_pts = time_recorded + 5

        # Load the video segment
        video, _, info = read_video(video_path, start_pts=start_pts, end_pts=end_pts, pts_unit='sec')

        total_frames = video.shape[0]
        frame_indices = torch.linspace(0, total_frames - 1, steps=self.frames_per_clip).long()
        video_clip = video[frame_indices]

        if self.transform:
            video_clip = torch.stack([self.transform(frame) for frame in video_clip])
        video_clip = video_clip.permute(0, 1, 2, 3)

        tools_vector = torch.tensor(row['Tools'], dtype=torch.float32)
        print(tools_vector)
        return video_clip, tools_vector
    
transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard ImageNet normalization
])
def transform_frame(frame):

    if frame.dim() == 3 and frame.size(2) == 3:  
        frame = frame.permute(2, 0, 1)  

    transform_ops = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                             std=[0.229, 0.224, 0.225]) ])
    return transform_ops(frame)

# Splitting the data into training, validation, and testing
video_ids = df['FileName'].unique()
# train_ids, test_ids = train_test_split(video_ids, test_size=2, random_state=42)
# train_ids, val_ids = train_test_split(train_ids, test_size=2, random_state=42)
train_ids = ["191R1"]
val_ids = ["191S1"]
test_ids = ["191R1"]

train_df = df[df['FileName'].isin(train_ids)]
val_df = df[df['FileName'].isin(val_ids)]
test_df = df[df['FileName'].isin(test_ids)]


def verify_videos(video_files):
    verified_videos = []
    for video_file in video_files:
        try:
            # Attempt to get a reader and read the first frame
            with imageio.get_reader(video_file) as reader:
                _ = reader.get_next_data()  
                verified_videos.append(video_file)  # If successful, add to verified list
        except Exception as e:
            print(f"Failed to open or read video file {video_file}: {e}")
    return verified_videos

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
        return not bool(tools)  



# Create datasets
train_dataset = VideoDataset(train_df, transform=transform_frame)
val_dataset = VideoDataset(val_df, transform=transform_frame)
test_dataset = VideoDataset(test_df, transform=transform_frame)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)


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


num_labels = len(all_tools)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomVivit(num_labels).to(device)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs, accumulation_steps=4):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        model.train()
        total_train_loss = 0
        print("loading train loader")
        for step, (frames, labels) in enumerate(train_loader):
            frames = frames.unsqueeze(0).repeat(2, 1, 1, 1)
            frames, labels = frames.to(device), labels.to(device)
            outputs = model(frames)
            loss = criterion(outputs, labels) / accumulation_steps  # Scale loss to account for accumulation
            loss.backward()  
            total_train_loss += loss.item() * accumulation_steps  

            
            if (step + 1) % accumulation_steps == 0:
                optimizer.step()  # Update parameters
                optimizer.zero_grad()

        
        if len(train_loader) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        # Calculate average training loss for the epoch
        average_train_loss = total_train_loss / len(train_loader)


#         # Validation phase
#         model.eval()
#         total_val_loss = 0
#         with torch.no_grad():
#             for frames, labels in val_loader:
#                 frames, labels = frames.to(device), labels.to(device)
#                 outputs = model(frames)
#                 val_loss = criterion(outputs, labels)
#                 total_val_loss += val_loss.item()

#         # Calculate average validation loss for the epoch

#         average_val_loss = total_val_loss / len(val_loader)
#         print(f'Epoch {epoch}: Average Train Loss {average_train_loss}, Average Val Loss {average_val_loss}')

# def evaluate_model(model, test_loader, criterion):
#     model.eval()
#     test_loss = 0
#     all_preds = []
#     all_labels = []
#     with torch.no_grad():
#         for frames, labels in test_loader:
#             frames, labels = frames.to(device), labels.to(device)
#             outputs = model(frames)
#             preds = torch.sigmoid(outputs).round()
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
#             test_loss += criterion(outputs, labels).item()
    
#     print("Sample predictions:", all_preds[:10])
#     print("Sample labels:", all_labels[:10])

#     # Flatten lists if necessary (for multi-label scenarios)
#     all_preds = [item for sublist in all_preds for item in sublist]
#     all_labels = [item for sublist in all_labels for item in sublist]

#     # Calculate metrics
#     accuracy = accuracy_score(all_labels, all_preds)
#     precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
#     recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
#     f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    

#     print(f'Test Loss: {test_loss / len(test_loader)}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, '
#           f'Recall: {recall:.4f}, F1 Score: {f1:.4f}')
#     mcm = multilabel_confusion_matrix(all_labels, all_preds)
#     print(mcm)


train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=15, accumulation_steps=4)
# torch.save(model, 'model_complete.pth')
# evaluate_model(model, test_loader, criterion)