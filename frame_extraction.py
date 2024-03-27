import cv2
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Function to convert time string to total frames
def time_to_frames(time_str, fps):
    h, m, s, f = map(int, time_str.split(':'))
    return int((h * 3600 + m * 60 + s) * fps + f)

# Load the Excel file
df = pd.read_excel('Cataract_Steps.xlsx', )
print(df.head())

df = df.iloc[1:].reset_index(drop=True)

# Define directories
resident_videos_dir = './Videos/Resident_Group'
staff_videos_dir = './Videos/Staff_Group'
output_dirs = {'train': './TrainFrames', 'test': './TestFrames'}
# Create output directories if they don't exist
for dir in output_dirs.values():
    os.makedirs(dir, exist_ok=True)

# Get video names from directories
resident_videos = [f for f in os.listdir(resident_videos_dir) if f.endswith('.mp4') or f.endswith('.mov')]
staff_videos = [f for f in os.listdir(staff_videos_dir) if f.endswith('.mp4') or f.endswith('.mov')]

# Split video names into train and test sets
train_resident, test_resident = train_test_split(resident_videos, test_size=0.2, random_state=42)
train_staff, test_staff = train_test_split(staff_videos, test_size=0.2, random_state=42)


# Modify the extract_frames function to accept group directory
def extract_frames(video_list, group_dir, output_dir, df):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate through DataFrame for each video
    for index, row in df.iterrows():
        phase = row['Unnamed: 0']  # Extract phase name
        for video_filename in video_list:  # Iterate through video list
            video_code = video_filename.split('.')[0]  # Remove file extension for comparison
            for i in range(1, len(df.columns), 2):  # Iterate through columns for start and end times
                if df.columns[i].startswith(video_code):
                    start_time, end_time = row[i], row[i + 1]
                    if pd.isna(start_time) or pd.isna(end_time):
                        continue  # Skip if start or end times are NaN
                    
                    video_path = os.path.join(group_dir, video_filename)  # Use the full path to video
                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        print(f"Failed to open video: {video_path}")
                        continue
                    
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    start_frame = time_to_frames(start_time, fps)
                    end_frame = time_to_frames(end_time, fps)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                    
                    frame_counter = start_frame
                    count = 0
                    while frame_counter <= end_frame:
                        ret, frame = cap.read()
                        if not ret:
                            break  # End of video or error
                        
                        if count % (fps // 1) == 0:  # Save 1 frame per second
                            frame_filename = f"{phase}_{video_code}_{frame_counter}.jpg"
                            cv2.imwrite(os.path.join(output_dir, frame_filename), frame)
                        
                        count += 1
                        frame_counter += 1

                    cap.release()


# Use the function to extract frames for each set
extract_frames(train_resident, resident_videos_dir, output_dirs['train'], df)
extract_frames(test_resident, resident_videos_dir, output_dirs['test'], df)
extract_frames(train_staff, staff_videos_dir, output_dirs['train'], df)
extract_frames(test_staff, staff_videos_dir, output_dirs['test'], df)



