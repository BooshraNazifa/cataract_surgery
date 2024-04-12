import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch
from sklearn.model_selection import train_test_split
import imageio
from transformers import VivitImageProcessor, VivitForVideoClassification


def time_to_frames(time_str, fps):
    h, m, s, f = map(int, time_str.split(':'))
    return int((h * 3600 + m * 60 + s) * fps + f)

class SurgicalPhaseVideoDataset(Dataset):
    def __init__(self, annotations, videos_dir, processor, num_frames=32):
        self.annotations = annotations
        self.videos_dir = videos_dir
        self.processor = processor
        self.num_frames = num_frames

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        video_path = os.path.join(self.videos_dir, f"{row['video_id']}.mp4")
        fps = self.get_fps(video_path)
        start_frame = time_to_frames(row['start_time'], fps)
        end_frame = time_to_frames(row['end_time'], fps)
        frames = self.extract_phase_frames(video_path, start_frame, end_frame, fps)
        
        if len(frames) < self.num_frames:
            raise ValueError(f"Not enough frames extracted for video {row['video_id']}. Extracted {len(frames)} frames.")

        # Preprocess frames using ViViT's image processor
        inputs = self.processor(frames, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)  # Remove extra dimensions if any
        
        label = phases.index(row['phase'])
        label = torch.tensor(label, dtype=torch.long)
        
        return pixel_values, label

    def extract_phase_frames(self, video_path, start_frame, end_frame, fps):
        vid = imageio.get_reader(video_path, 'ffmpeg')
        total_frames_to_extract = int(30 * (end_frame - start_frame) / fps)
        step_size = (end_frame - start_frame) / total_frames_to_extract if total_frames_to_extract > 0 else 1

        frames = []
        for i in range(total_frames_to_extract):
            frame_index = int(start_frame + i * step_size)
            try:
                frame = vid.get_data(frame_index)
                frames.append(frame)
            except Exception as e:
                print(f"Error reading frame at index {frame_index}: {str(e)}")
                continue

        vid.close()
        frames = np.array(frames)  
        frames = torch.tensor(frames) 
        return frames

    def get_fps(self, video_path):
        with imageio.get_reader(video_path, 'ffmpeg') as vid:
            return vid.get_meta_data()['fps']

# Load the Excel file
df = pd.read_excel('/home/booshra/final_project/cataract_surgery/Cataract_Steps.xlsx', )
df = df.iloc[1:].reset_index(drop=True)


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

print(df_final.head(-10))


# Define directories
resident_videos_dir = '/home/booshra/projects/def-holden/Cataract_data/Resident_Group'
staff_videos_dir = '/home/booshra/projects/def-holden/Cataract_data/Staff_Group' 

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


# Initialize the processor and model
processor = VivitImageProcessor.from_pretrained("/home/booshra/final_project")
model = VivitForVideoClassification.from_pretrained("/home/booshra/final_project")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Use the filtered annotations for creating datasets
train_resident_dataset = SurgicalPhaseVideoDataset(train_resident_annotations, resident_videos_dir, processor)
train_staff_dataset = SurgicalPhaseVideoDataset(train_staff_annotations, staff_videos_dir, processor)

val_resident_dataset = SurgicalPhaseVideoDataset(val_resident_annotations, resident_videos_dir, processor)
val_staff_dataset = SurgicalPhaseVideoDataset(val_staff_annotations, staff_videos_dir, processor)

test_resident_dataset = SurgicalPhaseVideoDataset(test_resident_annotations, resident_videos_dir, processor)
test_staff_dataset = SurgicalPhaseVideoDataset(test_staff_annotations, staff_videos_dir, processor)


# Combine datasets using ConcatDataset
train_dataset = ConcatDataset([train_resident_dataset, train_staff_dataset])
val_dataset = ConcatDataset([val_resident_dataset, val_staff_dataset])
test_dataset = ConcatDataset([test_resident_dataset, test_staff_dataset])

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=4,  num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=4,  num_workers=4)


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train_one_epoch(model, dataloader, device, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs).logits
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)
    
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = 100.0 * correct_predictions / total_predictions
    return epoch_loss, epoch_accuracy

def validate_model(model, dataloader, device, criterion):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs).logits
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = 100.0 * correct_predictions / total_predictions
    return epoch_loss, epoch_accuracy

# Training and validation epochs
num_epochs = 1
for epoch in range(num_epochs):
    print(epoch)
    train_loss, train_accuracy = train_one_epoch(model, train_dataloader, device, optimizer, criterion)
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
    
    val_loss, val_accuracy = validate_model(model, val_dataloader, device, criterion)
    print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

