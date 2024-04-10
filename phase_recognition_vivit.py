import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from torchvision.io import read_video
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

def timestamp_to_seconds(timestamp, fps=30):
    
    parts = timestamp.split(':')
    if len(parts) == 4:
        hours, minutes, seconds, frames = parts
    else:
        # Handle the case where the timestamp might not include frames (FF)
        hours, minutes, seconds = parts
        frames = 0
    
    # Convert hours, minutes, seconds, and frames to seconds
    total_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(frames) / fps
    return total_seconds

# Load the Excel file
df = pd.read_excel('Cataract_Steps.xlsx', )
df = df.iloc[1:].reset_index(drop=True)

# Define directories
resident_videos_dir = './Videos/Resident_Group'
staff_videos_dir = './Videos/Staff_Group' 

# List of phase names as your classes
phases = ["Paracentesis", "Viscoelastic", "Wound", "Capsulorhexis", "Hydrodissection", 
          "Phaco", "Viscoelastic2", "IOL Insertion", "IOL Positioning", 
          "Viscoelastic removed", "Hydration", "Malyugin Ring Insertion", 
          "Malyugin Ring Removal", "Vision Blue"]

# Specify explicit test videos
test_videos = ['191R1', '191R2', '191R3', '191R4', '191R5', '191R6', '191S1', '191S3', '191S4','191S5', '191S6', '191S7']

def get_videos_excluding_tests(directory, test_videos):
    all_videos = os.listdir(directory)
    
    excluded_videos = []
    for f in all_videos:
        core_identifier = f.split('.')[0]
        
        # Checking for exact match in test_videos list
        if not any(core_identifier == test_vid for test_vid in test_videos) and (f.endswith('.mp4') or f.endswith('.mov')):
            excluded_videos.append(f)
    
    return excluded_videos


class SurgicalPhaseVideoDataset(Dataset):
    def __init__(self, annotations_df, videos_dir, num_frames, transform=None):
        self.annotations_df = annotations_df
        self.videos_dir = videos_dir
        self.num_frames = num_frames
        self.transform = transform
    
    def __getitem__(self, idx):
        row = self.annotations_df.iloc[idx]
        video_id = row['video_id']
        video_path = os.path.join(self.videos_dir, f"{video_id}.mp4")
        
        # Read the video segment between start_time and end_time
        video_segment, _, _ = read_video(video_path, start_pts=row['start_time'], end_pts=row['end_time'], pts_unit='sec')
        
        # Sample frames uniformly
        frame_indices = torch.linspace(0, video_segment.shape[0] - 1, self.num_frames).long()
        frames = video_segment[frame_indices]
        
        if self.transform is not None:
            frames = torch.stack([self.transform(img) for img in frames])
        
        label = phases.index(row['phase'])  
        
        return frames, label
    
    def __len__(self):
        return len(self.annotations_df)
    
# Define the transforms for video frames
frame_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Resize to the input size expected by TimeSformer
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
])

# Step 1: Melting the DataFrame to long format
df_melted = df.melt(id_vars=["Unnamed: 0"], var_name="video_phase", value_name="time")
df_melted['phase'] = df_melted['Unnamed: 0']
df_melted.drop(columns=["Unnamed: 0"], inplace=True)

# Extracting video_id and whether it's start or end time
df_melted['video_id'] = df_melted['video_phase'].apply(lambda x: x if "Unnamed" not in x else pd.NA).ffill()
df_melted['time_type'] = df_melted['video_phase'].apply(lambda x: "start" if "Unnamed" not in x else "end")

# Dropping rows where 'video_phase' contains 'Unnamed', as they were used to fill 'video_id' and 'time_type'
df_melted = df_melted.dropna(subset=["video_id"]).drop(columns=["video_phase"])

# Filtering out rows where 'time' is NA (i.e., missing phases for some videos)
df_melted = df_melted.dropna(subset=["time"])

# Pivot to get start and end times in one row
df_final = df_melted.pivot_table(index=["video_id", "phase"], columns="time_type", values="time", aggfunc='first').reset_index()

# Renaming columns to the correct names
df_final.columns = ['video_id', 'phase', 'end_time', 'start_time']  # Reorder if necessary

# Convert times to seconds (assuming you have a function timestamp_to_seconds for this)
df_final['start_time'] = df_final['start_time'].apply(lambda x: timestamp_to_seconds(x))
df_final['end_time'] = df_final['end_time'].apply(lambda x: timestamp_to_seconds(x))

print(df_final.head(-10))

# Get resident and staff videos excluding the test ones
resident_videos = get_videos_excluding_tests(resident_videos_dir, test_videos)
staff_videos = get_videos_excluding_tests(staff_videos_dir, test_videos)

# Split video names into train and test sets
train_resident, val_resident = train_test_split(resident_videos, test_size=0.2, random_state=42)
train_staff, val_staff = train_test_split(staff_videos, test_size=0.2, random_state=42)
test_resident = ['191R1', '191R2', '191R3', '191R4', '191R5', '191R6']
test_staff = ['191S1', '191S3', '191S4','191S5', '191S6', '191S7']

# Filter annotations for resident and staff videos
train_resident_annotations = df_final[df_final['video_id'].isin([vid_id.replace('.mp4', '') for vid_id in train_resident])]
train_staff_annotations = df_final[df_final['video_id'].isin([vid_id.replace('.mp4', '') for vid_id in train_staff])]

# Similar approach for validation and test datasets
val_resident_annotations = df_final[df_final['video_id'].isin([vid_id.replace('.mp4', '') for vid_id in val_resident])]
val_staff_annotations = df_final[df_final['video_id'].isin([vid_id.replace('.mp4', '') for vid_id in val_staff])]

test_resident_annotations = df_final[df_final['video_id'].isin([vid_id.replace('.mp4', '') for vid_id in test_resident])]
test_staff_annotations = df_final[df_final['video_id'].isin([vid_id.replace('.mp4', '') for vid_id in test_staff])]


num_frames = 36
# Use the filtered annotations for creating datasets
train_resident_dataset = SurgicalPhaseVideoDataset(train_resident_annotations, resident_videos_dir, num_frames, transform=frame_transform)
train_staff_dataset = SurgicalPhaseVideoDataset(train_staff_annotations, staff_videos_dir, num_frames, transform=frame_transform)

val_resident_dataset = SurgicalPhaseVideoDataset(val_resident_annotations, resident_videos_dir, num_frames, transform=frame_transform)
val_staff_dataset = SurgicalPhaseVideoDataset(val_staff_annotations, staff_videos_dir, num_frames, transform=frame_transform)

test_resident_dataset = SurgicalPhaseVideoDataset(test_resident_annotations, resident_videos_dir, num_frames, transform=frame_transform)
test_staff_dataset = SurgicalPhaseVideoDataset(test_staff_annotations, staff_videos_dir, num_frames, transform=frame_transform)


# Combine datasets using ConcatDataset
train_dataset = ConcatDataset([train_resident_dataset, train_staff_dataset])
val_dataset = ConcatDataset([val_resident_dataset, val_staff_dataset])
test_dataset = ConcatDataset([test_resident_dataset, test_staff_dataset])

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=4,  num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=4,  num_workers=4)

# Load a pre-trained X3D model
model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_s', pretrained=True)
num_classes = len(phases)  

# Replace the final fully connected layer to match the number of classes
model.blocks[-1].proj = torch.nn.Linear(in_features=model.blocks[-1].proj.in_features, out_features=num_classes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5

def validate_model(model, dataloader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    val_running_loss = 0.0
    val_correct_predictions = 0
    val_total_predictions = 0

    with torch.no_grad():  # No gradients needed for validation
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_correct_predictions += (predicted == labels).sum().item()
            val_total_predictions += labels.size(0)

    val_loss = val_running_loss / len(dataloader)
    val_accuracy = 100 * val_correct_predictions / val_total_predictions
    return val_loss, val_accuracy


for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    for inputs, labels in train_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)

    # Print statistics
    print(f'Epoch {epoch+1}, Loss: {running_loss / len(train_dataloader)}, Accuracy: {100 * correct_predictions / total_predictions}%')
    
    # Inside your training loop, after completing one epoch of training:
    val_loss, val_accuracy = validate_model(model, val_dataloader, criterion, device)
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')
    
    
def test_model(model, dataloader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    test_running_loss = 0.0
    test_correct_predictions = 0
    test_total_predictions = 0

    with torch.no_grad():  # No gradients needed for testing
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_correct_predictions += (predicted == labels).sum().item()
            test_total_predictions += labels.size(0)

    test_loss = test_running_loss / len(dataloader)
    test_accuracy = 100 * test_correct_predictions / test_total_predictions
    return test_loss, test_accuracy

# Perform the test using the test dataloader
test_loss, test_accuracy = test_model(model, test_dataloader, criterion, device)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

