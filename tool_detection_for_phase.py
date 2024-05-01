import pandas as pd
import ast
import av
import os
import torch
import numpy as np
import torch.nn as nn
from sklearn.model_selection import train_test_split
from transformers import VivitModel
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_video
from torch.optim import AdamW
from torch.nn import BCEWithLogitsLoss
from PIL import Image
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torchvision import transforms


# csv_directory = './Cataract_Tools'
# videos_dir = './Videos'
# Load the Excel file for ground truth
df = pd.read_excel('/scratch/booshra/final_project/cataract_surgery/tool_detection.xlsx')
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
csv_directory = '/scratch/booshra/final_project/cataract_surgery/Cataract_Tools'
videos_dir = "/scratch/booshra/tool"
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
video_ids = np.random.choice(video_ids, size=8, replace=False)
print(video_ids)
train_ids, test_ids = train_test_split(video_ids, test_size=2, random_state=42)
train_ids, val_ids = train_test_split(train_ids, test_size=2, random_state=42)

train_df = dataframe[dataframe['FileName'].isin(video_ids)]
val_df = dataframe[dataframe['FileName'].isin(val_ids)]
test_df = dataframe[dataframe['FileName'].isin(test_ids)]

print(train_df)

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, root_dir, transform=None, frames_per_clip=16):
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
        container = av.open(video_path)
        time_recorded = self.df.iloc[idx]['Time Recorded']

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
            video_clip = torch.stack([self.transform(frame) for frame in interpolated])
        video_clip = video_clip.permute(0, 1, 2, 3)

        # Prepare labels
        labels = torch.tensor(self.df.iloc[idx]['Tools'], dtype=torch.float32)
        print(labels)
        return video_clip, labels
    
transform = transforms.Compose([
    transforms.Resize((128, 128)),  
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = VideoDataset(train_df, videos_dir, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=0.001)
criterion = BCEWithLogitsLoss()

model_path = '/scratch/booshra/final_project/vivit_tool_phase_lastepoch.pth'

if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location='cuda')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Model loaded successfully.")


def train_model(dataloader, model, criterion, optimizer, num_epochs=3, accumulation_steps=4):
    model.train()
    scaler = GradScaler()

    for epoch in range(num_epochs):
        print(f"Starting Epoch {epoch}")
        total_train = 0
        total_train_loss = 0
        optimizer.zero_grad()  
        accum_counter = 0

        for i, (videos, labels) in enumerate(dataloader):
            videos, labels = videos.to(device), labels.to(device)
            try:
              with autocast():
                outputs = model(videos)
                loss = criterion(outputs, labels) / accumulation_steps  

              scaler.scale(loss).backward()
              accum_counter += 1

              if accum_counter == accumulation_steps:
                    scaler.step(optimizer)  
                    scaler.update()
                    optimizer.zero_grad()  
                    accum_counter = 0  

              total_train += labels.numel()
              total_train_loss += loss.item() * videos.size(0) * accumulation_steps  
            except RuntimeError as e:
                   print(f"Skipping a video due to an error: {e}")
                   continue

        if accum_counter != 0:  # Check if there are unapplied gradients
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        avg_train_loss = total_train_loss / total_train
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
        avg_val_loss = total_val_loss / total_val if total_val != 0 else 0
        val_accuracy = total_val_correct / total_val if total_val != 0 else 0
        print(f"Epoch {epoch}, Validation Loss: {avg_val_loss}, Validation Accuracy: {val_accuracy:.2f}")

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}
        torch.save(checkpoint, model_path)

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