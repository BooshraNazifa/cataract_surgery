import ast
import av
import os
import numpy as np
import pandas as pd
import imageio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision.transforms import Resize, Normalize, ToTensor, Compose
from PIL import Image
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
from transformers import VivitModel
torch.cuda.empty_cache()


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
    if pd.notna(tool_info) and isinstance(tool_info, str):
        try:
            tool_dicts = ast.literal_eval(tool_info)  
            return [tool['class'] for tool in tool_dicts]
        except Exception as e:
            print(f"Error parsing tool_info: {tool_info} with error {e}")
    return []

def tools_to_vector(tools):
    return [1 if tool in tools else 0 for tool in all_tools]

# Directory containing CSV files
# csv_directory = './Cataract_Tools'
# videos_dir = './Videos'
csv_directory = '/scratch/booshra/final_project/cataract_surgery/Cataract_Tools'
videos_dir = "/scratch/booshra/tool"
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


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.frames_per_clip = 16

    def __len__(self):
        return len(self.dataframe)
    
    def find_video_path(self, base_path):
        """Check for different video file extensions."""
        for ext in ['.mp4', '.mov']:
            if os.path.exists(base_path + ext):
                return f"{base_path}{ext}"
        raise FileNotFoundError(f"No video file found for {base_path} with extensions .mp4 or .mov")
    

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        video_filename = str(row['FileName'])
        base_video_path = os.path.join(videos_dir, video_filename)
        video_path = self.find_video_path(base_video_path)
        print(video_path)
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        container = av.open(video_path)
        time_recorded = row['Time Recorded']
        start_pts = max(time_recorded - 5, 0)
        end_pts = time_recorded + 5

        stream = container.streams.video[0]

        seek_point = int(start_pts / stream.time_base)
        container.seek(seek_point, any_frame=False, backward=True, stream=stream)
        frame_list = []
        frames_decoded = 0
        for frame in container.decode(video=0):
            frame_time = frame.pts * stream.time_base  
            if frame_time < start_pts:
               continue
            if frame_time > end_pts:
               break
            if frames_decoded < self.frames_per_clip:
               frame_list.append(frame.to_image())
               frames_decoded += 1
            else:
               break

        interpolated = []
        step = len(frame_list) / 32
        frame_array_list = [np.array(frame) for frame in frame_list]
        for i in range(32 - 1):
          index = int(step * i)
          next_index = min(int(step * (i + 1)), len(frame_list) - 1)
          alpha = (step * i) - index
          interpolated_frame = (1 - alpha) * frame_array_list[index].astype(float) + alpha * frame_array_list[next_index].astype(float)
          interpolated_frame = np.clip(interpolated_frame, 0, 255).astype(np.uint8)

          interpolated_frame = Image.fromarray(interpolated_frame)
          interpolated.append(interpolated_frame)
        interpolated.append(frame_list[-1])

        if self.transform:
           video_frames = [self.transform(frame) for frame in interpolated]
        
        video_clip = torch.stack(video_frames)
        video_clip = video_clip.permute(0, 1, 2, 3)

        tools_vector = torch.tensor(row['Tools'], dtype=torch.float32)
        print(tools_vector)
        return video_clip, tools_vector
    

transform = Compose([
    Resize((128, 128)),  
    Resize((224, 224)),  
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Splitting the data into training, validation, and testing
video_ids = df['FileName'].unique()
video_ids = np.random.choice(video_ids, size=5, replace=False)
train_ids, test_ids = train_test_split(video_ids, test_size=2, random_state=42)
train_ids, val_ids = train_test_split(train_ids, test_size=2, random_state=42)
# train_ids = ["191R1"]
# val_ids = ["191S1"]
# test_ids = ["191R1"]

train_df = df[df['FileName'].isin(video_ids)]
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
train_dataset = VideoDataset(train_df, transform=transform)
val_dataset = VideoDataset(val_df, transform=transform)
test_dataset = VideoDataset(test_df, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)


class CustomVivit(nn.Module):
    def __init__(self, num_labels):
        super(CustomVivit, self).__init__()
        self.vivit = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400")
        self.dropout = nn.Dropout(0.5)  # Optional: to mitigate overfitting
        self.classifier = nn.Linear(self.vivit.config.hidden_size, num_labels)  
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for multi-label classification

    def forward(self, inputs):
        outputs = self.vivit(inputs)  
        x = self.dropout(outputs.pooler_output)  
        x = self.classifier(x)  
        x = self.sigmoid(x)  
        return x


num_labels = len(all_tools)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomVivit(num_labels).to(device)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model_path = '/scratch/booshra/final_project/vivit_tool_lastepoch.pth'

if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location='cuda')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Model loaded successfully.")

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs, accumulation_steps=4):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        scaler = GradScaler()

        model.train()
        total_train_loss = 0
        total_steps = len(train_loader)
        print("loading train loader")
        for step, (frames, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            frames, labels = frames.to(device), labels.to(device)
            with autocast():
                outputs = model(frames)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            
            if (step + 1) % accumulation_steps == 0 or step == total_steps - 1:
               scaler.step(optimizer)
               scaler.update()
               optimizer.zero_grad()
             
            total_train_loss += loss.item()  
            if step % 100 == 0:
                print(f"Step {step}/{total_steps}, Current Loss: {loss.item():.4f}")
        average_train_loss = total_train_loss / total_steps
        print(f"Average Training Loss: {average_train_loss:.4f}")

        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for frames, labels in val_loader:
                frames, labels = frames.to(device), labels.to(device)
                outputs = model(frames)
                val_loss = criterion(outputs, labels)
                total_val_loss += val_loss.item()

        # Calculate average validation loss for the epoch

        average_val_loss = total_val_loss / len(val_loader)
        print(f"Average Validation Loss: {average_val_loss:.4f}")
    
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}
        torch.save(checkpoint, model_path)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
        print(f"Epoch {epoch + 1} complete. Training Loss: {average_train_loss:.4f}, Validation Loss: {average_val_loss:.4f}")

def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for frames, labels in test_loader:
            frames, labels = frames.to(device), labels.to(device)
            outputs = model(frames)
            preds = torch.sigmoid(outputs).round()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            test_loss += criterion(outputs, labels).item()
    
    print("Sample predictions:", all_preds[:10])
    print("Sample labels:", all_labels[:10])

    # Flatten lists if necessary (for multi-label scenarios)
    all_preds = [item for sublist in all_preds for item in sublist]
    all_labels = [item for sublist in all_labels for item in sublist]

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    

    print(f'Test Loss: {test_loss / len(test_loader)}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, '
          f'Recall: {recall:.4f}, F1 Score: {f1:.4f}')
    mcm = multilabel_confusion_matrix(all_labels, all_preds)
    print(mcm)


train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=3, accumulation_steps=4)
torch.save(model, 'model_complete.pth')
evaluate_model(model, test_loader, criterion)

