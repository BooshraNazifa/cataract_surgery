import pandas as pd
import ast 
import os
import numpy as np
import torch.nn as nn
from sklearn.model_selection import train_test_split
from transformers import VivitModel
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from torchvision.io import read_video
from transformers import AdamW
from torch.nn import BCEWithLogitsLoss
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.cuda.amp import autocast
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

videos_dir = "./Videos"

## Load the Excel file for Phase detection results

df = pd.read_excel('./test_results_by_phase.xlsx')

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
    video_id = video_id.split(' ')[0]
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
video_ids = np.random.choice(video_ids, size=5, replace=False)
train_ids, test_ids = train_test_split(video_ids, test_size=2, random_state=42)  
train_ids, val_ids = train_test_split(train_ids, test_size=2/8, random_state=42) 


train_df = dataframe[dataframe['FileName'].isin(train_ids)]
val_df = dataframe[dataframe['FileName'].isin(val_ids)]
test_df = dataframe[dataframe['FileName'].isin(test_ids)]

print(train_df)

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.df = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def find_video_path(self, base_path):
        """ Check for different video file extensions. """
        for ext in ['.mp4', '.mov']:
            if os.path.exists(base_path + ext):
                return base_path + ext
        raise FileNotFoundError(f"No video file found for {base_path} with extensions .mp4 or .mov")


    def __getitem__(self, idx):
        base_video_path = os.path.join(self.root_dir, self.df.iloc[idx]['FileName'])
        video_path = self.find_video_path(base_video_path)
        time_recorded = self.df.iloc[idx]['Time Recorded']  # Time recorded should be in seconds
        
        # Load the video
        video, _, info = read_video(video_path, pts_unit='sec')
        
        # Find the frame that corresponds to the time recorded
        fps = info['video_fps']  # frames per second
        frame_idx = int(time_recorded * fps)  # convert time to frame number
        
        if frame_idx >= video.shape[0]:
            frame_idx = video.shape[0] - 1

        frame = video[frame_idx]  
        frame = frame.permute(2, 0, 1)

        if self.transform:
            frame = self.transform(frame)  # Apply transformations on the frame

        # Prepare labels
        labels = torch.tensor(self.df.iloc[idx]['Tools'], dtype=torch.float32)

        return frame, labels

    

transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = VideoDataset(train_df, videos_dir, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

val_dataset = VideoDataset(val_df, videos_dir, transform=transform)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

test_dataset = VideoDataset(test_df, videos_dir, transform=transform)
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

optimizer = AdamW(model.parameters(), lr=1e-4)
criterion = BCEWithLogitsLoss()

def train_model(dataloader, model, criterion, optimizer, num_epochs=3):
    model.train()
    scaler = GradScaler()  

    for epoch in range(num_epochs):
        print(f"Starting Epoch {epoch}")
        for videos, labels in dataloader:
            optimizer.zero_grad()

            # Enable automatic mixed precision
            with autocast():
                outputs = model(videos)  # Forward pass with mixed precision
                loss = criterion(outputs.logits, labels)  # Calculate loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            print(f"Epoch {epoch}, Loss: {loss.item()}")

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():  # No gradients needed for validation
            for videos, labels in val_dataloader:
                with autocast():
                    outputs = model(videos)
                    loss = criterion(outputs.logits, labels)
                    val_loss += loss.item() * videos.size(0)

                    # Calculate accuracy
                    _, predicted = torch.max(outputs.logits, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_dataloader.dataset)
        val_accuracy = correct / total
        print(f"Epoch {epoch}, Validation Loss: {avg_val_loss}, Validation Accuracy: {val_accuracy:.2f}")


        print(f"Epoch {epoch} complete.")

train_model(train_dataloader, model, criterion, optimizer)

def evaluate_model(dataloader, model):
    model.eval()  # Set the model to evaluation mode
    predictions = []
    true_labels = []

    with torch.no_grad():  # No gradients needed
        for videos, labels in dataloader:
            with autocast():
                outputs = model(videos)
                _, predicted = torch.max(outputs.logits, 1)
                predictions.extend(predicted.cpu().numpy())  # Store predictions
                true_labels.extend(labels.cpu().numpy())  # Store true labels

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='macro')
    recall = recall_score(true_labels, predictions, average='macro')
    f1 = f1_score(true_labels, predictions, average='macro')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

evaluate_model(test_dataloader, model)