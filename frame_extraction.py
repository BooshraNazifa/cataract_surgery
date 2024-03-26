import cv2
import pandas as pd
import os

# Function to convert time string to total frames
def time_to_frames(time_str, fps):
    h, m, s, f = map(int, time_str.split(':'))
    return int((h * 3600 + m * 60 + s) * fps + f)

# Load the Excel file
df = pd.read_excel('Cataract_steps_test.xlsx', )
print(df.head())

df = df.iloc[1:].reset_index(drop=True)

# Video directory and frame save directory
video_dir = './test_videos'
frame_save_dir = './testimagesfps'
if not os.path.exists(frame_save_dir):
    os.makedirs(frame_save_dir)

# Iterate through DataFrame
for index, row in df.iterrows():
    phase = row['Unnamed: 0']  # Phase name
    for i in range(1, len(df.columns), 2):  # Start from 1 to skip phase column, step by 2 for start-end pairs
        video_filename = df.columns[i].split()[0] + '.mp4'  # Construct video filename
        start_time, end_time = row[i], row[i+1]
        if pd.isna(start_time) or pd.isna(end_time):  # Skip if any time is NaN
            continue
        
        # Load video
        video_path = os.path.join(video_dir, video_filename)
        cap = cv2.VideoCapture(video_path)
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
            
            if count % (fps // 1) == 0:  # Save 1 frames per second
                frame_filename = f"{phase}_{video_filename}_{frame_counter}.jpg"
                cv2.imwrite(os.path.join(frame_save_dir, frame_filename), frame)

            count += 1  # Increment count for every frame
            frame_counter += 1  # Increment to move to the next frame
        
        cap.release()