import pandas as pd
import ast
import os
import numpy as np
import torch.nn as nn
from sklearn.model_selection import train_test_split
from transformers import VivitModel
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_video
from torch.optim import AdamW
from torch.nn import BCEWithLogitsLoss
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.cuda.amp import autocast
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torchvision import transforms

videos_dir = "/scratch/booshra/tool"

# Load the Excel file for ground truth
df = pd.read_excel('./tool_detection.xlsx')
# Function to convert time strings to seconds
def time_to_seconds(time_str):
    if pd.isna(time_str):
        return None
    hh, mm, ss, ff = map(int, time_str.split(':'))
    return hh * 3600 + mm * 60 + ss + ff / 100.0

# Find the index of the Capsulorhexis phase
capsulorhexis_row = df[df['Unnamed: 0'] == 'Capsulorhexis'].iloc[0]

phase_times = {}
for i in range(1, len(capsulorhexis_row), 2):
    video_id = df.columns[i].strip()
    start_time = capsulorhexis_row[i]
    end_time = capsulorhexis_row[i+1]

    # Convert start and end times to seconds
    start_seconds = time_to_seconds(start_time)
    end_seconds = time_to_seconds(end_time)

    phase_times[video_id] = (start_seconds, end_seconds)


print(f"Phase and start/end times {phase_times}")


def tools_to_vector(tools):
    return [1 if tool in tools else 0 for tool in all_tools]
def load_data_from_directory(directory):
    csv_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.csv')]
    list_of_dfs = [pd.read_csv(file) for file in csv_files]
    for df in list_of_dfs:
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

# Directory containing CSV files
csv_directory = './Cataract_Tools'
dataframe = load_data_from_directory(csv_directory)

# Preprocess the DataFrame as previously
dataframe['Tool Names'] = dataframe['Tool bounding box'].apply(extract_tools)
all_tools = sorted(set(tool for sublist in dataframe['Tool Names'] for tool in sublist))
dataframe['Tools'] = dataframe['Tool Names'].apply(tools_to_vector)
print(dataframe.head())

# filter frames of the phase
def filter_frames(df, phase_times):
    results = []
    for index, row in df.iterrows():
        video_id = row['FileName']
        time_recorded = row['Time Recorded']
        start_time, end_time = phase_times.get(video_id, (0, 0))
        if start_time <= time_recorded <= end_time:
            results.append(row)
    return pd.DataFrame(results)

dataframe = filter_frames(dataframe, phase_times)
print(dataframe)


# Splitting the data into training, validation, and testing
video_ids = dataframe['FileName'].unique()
print(video_ids)
# video_ids = np.random.choice(video_ids, size=3, replace=False)
train_ids, test_ids = train_test_split(video_ids, test_size=2, random_state=42)
train_ids, val_ids = train_test_split(train_ids, test_size=2, random_state=42)


train_df = dataframe[dataframe['FileName'].isin(train_ids)]
val_df = dataframe[dataframe['FileName'].isin(val_ids)]
test_df = dataframe[dataframe['FileName'].isin(test_ids)]

print(train_df)

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, root_dir, transform=None, frames_per_clip=32):
        self.df = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.frames_per_clip = frames_per_clip  


    def __len__(self):
        return len(self.df)

    def find_video_path(self, base_path):
        """Check for different video file extensions."""
        for ext in ['.mp4', '.mov']:
            if os.path.exists(base_path + ext):
                return base_path + ext
        raise FileNotFoundError(f"No video file found for {base_path} with extensions .mp4 or .mov")

    def __getitem__(self, idx):
        base_video_path = os.path.join(self.root_dir, self.df.iloc[idx]['FileName'])
        video_path = self.find_video_path(base_video_path)
        time_recorded = self.df.iloc[idx]['Time Recorded']

        start_pts = max(time_recorded - 5, 0)
        end_pts = time_recorded + 5

        # Load the video segment
        video, _, info = read_video(video_path, start_pts=start_pts, end_pts=end_pts, pts_unit='sec')

        fps = info['video_fps']
        total_frames = video.shape[0]
        frame_indices = torch.linspace(0, total_frames - 1, steps=self.frames_per_clip).long()
        video_clip = video[frame_indices]

        if self.transform:
            # Transform each frame individually if necessary
            video_clip = torch.stack([self.transform(frame) for frame in video_clip])
        video_clip = video_clip.permute(0, 1, 2, 3)

        # Prepare labels
        labels = torch.tensor(self.df.iloc[idx]['Tools'], dtype=torch.float32)

        # Ensure video clip is shaped [num_frames, channels, height, width]
        return video_clip, labels
    
def transform_frame(frame):

    if frame.dim() == 3 and frame.size(2) == 3:  
        frame = frame.permute(2, 0, 1)  

    transform_ops = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                             std=[0.229, 0.224, 0.225]) ])
    return transform_ops(frame)

train_dataset = VideoDataset(train_df, videos_dir, transform=transform_frame)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

val_dataset = VideoDataset(val_df, videos_dir, transform=transform_frame)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

test_dataset = VideoDataset(test_df, videos_dir, transform=transform_frame)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

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
model = CustomVivit(num_labels)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=0.001)
criterion = BCEWithLogitsLoss()

def train_model(dataloader, model, criterion, optimizer, num_epochs=3):
    model.train()
    scaler = GradScaler()

    for epoch in range(num_epochs):
        print(f"Starting Epoch {epoch}")
        total_train = 0
        total_train_loss = 0

        for videos, labels in dataloader:
            videos, labels = videos.to(device), labels.to(device)
            optimizer.zero_grad()

            try:
                # Enable automatic mixed precision
                with autocast():
                    outputs = model(videos)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                total_train += labels.numel()
                total_train_loss += loss.item() * videos.size(0)

            except RuntimeError as e:
                print(f"Skipping a video due to an error: {e}")
                continue
        
        avg_train_loss = total_train_loss / len(dataloader.dataset)
        print(f"Epoch {epoch}, Training Loss: {avg_train_loss:.4f}")



        # Validation phase
        model.eval()  # Set the model to evaluation mode
        total_val_correct = 0
        total_val = 0
        total_val_loss = 0
        with torch.no_grad():  # No gradients needed for validation
            for videos, labels in val_dataloader:
                try:
                  with autocast():
                    outputs = model(videos)
                    val_loss = criterion(outputs, labels)

                    total_val_loss += val_loss.item() * videos.size(0)

                    predicted = torch.sigmoid(outputs) > 0.5
                    total_val_correct += (predicted == labels).float().sum().item()
                    total_val += labels.numel()
                except RuntimeError as e:
                   print(f"Skipping a video due to an error: {e}")
                   continue
        avg_val_loss = total_val_loss / total_val

        val_accuracy = total_val_correct / total_val
        print(f"Epoch {epoch}, Validation Loss: {avg_val_loss}, Validation Accuracy: {val_accuracy:.2f}")

        print(f"Epoch {epoch} complete.")


train_model(train_dataloader, model, criterion, optimizer)
torch.save(model, 'tool_complete.pth')

def evaluate_model(dataloader, model):
    model.eval()  
    predictions = []
    true_labels = []

    with torch.no_grad():  
        for videos, labels in dataloader:
            videos = videos.to(device)
            labels = labels.to(device)
            with autocast():
                outputs = model(videos)
                probs = torch.sigmoid(outputs)
                
                predicted = (probs > 0.5).float()
                predictions.extend(predicted.cpu().numpy())  
                true_labels.extend(labels.cpu().numpy()) 

    # Flatten lists if necessary (for multi-label scenarios)
    predictions = [item for sublist in predictions for item in sublist]
    true_labels = [item for sublist in true_labels for item in sublist]

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='macro')
    recall = recall_score(true_labels, predictions, average='macro')
    f1 = f1_score(true_labels, predictions, average='macro')
    conf_matrix = confusion_matrix(true_labels, predictions)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f'Confusion Matrix:\n{conf_matrix}')

evaluate_model(test_dataloader, model)