from skimage import io
import numpy as np
from skimage.transform import resize
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import imageio

# Function to convert time string to total frames
def time_to_frames(time_str, fps):
    h, m, s, f = map(int, time_str.split(':'))
    return int((h * 3600 + m * 60 + s) * fps + f)

def get_videos_excluding_tests(directory, test_videos):
    all_videos = os.listdir(directory)
    
    excluded_videos = []
    for f in all_videos:
        core_identifier = f.split('.')[0]
        
        # Checking for exact match in test_videos list
        if not any(core_identifier == test_vid for test_vid in test_videos) and (f.endswith('.mp4') or f.endswith('.mov')):
            excluded_videos.append(f)
    
    return excluded_videos


# Load the Excel file
df = pd.read_excel('Cataract_Steps.xlsx', )

df = df.iloc[1:].reset_index(drop=True)
print(df.head(10))

# Define directories
# resident_videos_dir = './Videos/Resident_Group'
# staff_videos_dir = './Videos/Staff_Group' 
# output_dirs = {'train': './TrainFrames', 'val': './ValFrames', 'test': './TestFrames'}

# Server
resident_videos_dir = '/home/booshra/projects/def-holden/Cataract_data/Resident_Group'
staff_videos_dir = '/home/booshra/projects/def-holden/Cataract_data/Staff_Group'
output_dirs = {'train': '/scratch/booshra/50/TrainFrames', 'val': '/scratch/booshra/50/ValFrames', 'test': '/scratch/booshra/50/TestFrames'}

# Create output directories if they don't exist
for dir in output_dirs.values():
    os.makedirs(dir, exist_ok=True)



# Specify explicit test videos
test_videos = ['191R1', '191R2', '191R3', '191R4', '191R5', '191R6', '191S1', '191S3', '191S4','191S5', '191S6', '191S7']


# Get resident and staff videos excluding the test ones
resident_videos = get_videos_excluding_tests(resident_videos_dir, test_videos)
staff_videos = get_videos_excluding_tests(staff_videos_dir, test_videos)

# Get test vdeos
test_filenames_resident = ['191R1.mp4','191R2.mp4', '191R3.mp4', '191R4.mp4', '191R5.mp4', '191R6.mp4']
test_filenames_staff = ['191S1.mp4', '191S3.mp4', '191S4.mov','191S5.mp4', '191S6.mp4', '191S7.mov']

# Limit to the first 50 videos for each group
# resident_videos = resident_videos
# staff_videos = staff_videos

print("Resident videos:", resident_videos)
print("Staff videos:", staff_videos)


# Split video names into train and test sets
train_resident, val_resident = train_test_split(resident_videos, test_size=0.2, random_state=42)
train_staff, val_staff = train_test_split(staff_videos, test_size=0.2, random_state=42)

def preprocess_dataframe(df):
    # Skip the header row and start from actual data
    df_data = df.iloc[1:]
    video_info = {}

    # Iterate over the rows, each row represents a phase
    for index, row in df_data.iterrows():
        phase = row['Unnamed: 0']
        
        # Iterate over each column pair (start and end time)
        for i in range(1, len(df.columns), 2):  # Step by 2 to get pairs of start/end times
            video_code = df.columns[i].split()[0]  # Split the column name to get the video code
            start_time = row[i]
            end_time = row[i+1]
            
            # Check if start time is not NaN, assuming that if start time is provided, end time is too
            if pd.notna(start_time):
                if video_code not in video_info:
                    video_info[video_code] = []
                video_info[video_code].append({
                    'phase': phase,
                    'start_time': start_time,
                    'end_time': end_time
                })
    
    return video_info


def extract_frames(video_list, group_dir, output_dir, df, start_from_video=None):
    print(f"Processing videos in {group_dir} for {output_dir}")
    video_info = preprocess_dataframe(df)
    start_processing = start_from_video is None

    for video_filename in video_list:
        video_code = video_filename.split('.')[0]

        if start_from_video and video_code == start_from_video:
            start_processing = True
        if not start_processing or video_code not in video_info:
            continue

        print(f"Processing {video_filename}...")
        video_path = os.path.join(group_dir, video_filename)
        for phase_info in video_info[video_code]:
            extract_phase_frames(video_path, output_dir, phase_info)

def extract_phase_frames(video_path, output_dir, phase_info):
    try:
        vid = imageio.get_reader(video_path, 'ffmpeg')
        fps = vid.get_meta_data()['fps']
    except Exception as e:
        print(f"Failed to open video: {video_path} with error {e}")
        return

    # Calculate the start and end frames based on the video's fps
    start_frame = time_to_frames(phase_info['start_time'], fps)
    end_frame = time_to_frames(phase_info['end_time'], fps)

    # Calculate the duration in seconds
    duration_in_seconds = (end_frame - start_frame) / fps

    # Calculate how many frames to extract 
    total_frames_to_extract = int(15 * duration_in_seconds)

    # Calculate the step size to spread the frame extraction evenly across the duration
    if total_frames_to_extract > 0:
        step_size = (end_frame - start_frame) / total_frames_to_extract
    else:
        step_size = 1  # Fallback to avoid division by zero

    phase = phase_info['phase']
    video_code = os.path.basename(video_path).split('.')[0]

    current_frame = start_frame
    for i in range(total_frames_to_extract):
        try:
            # Calculate the next frame to extract
            frame_index = int(current_frame)
            frame = vid.get_data(frame_index)
            frame_timestamp_seconds = frame_index / fps  # Timestamp in seconds

            frame_filename = f"{phase}_{video_code}_{frame_index}_{frame_timestamp_seconds:.2f}.jpg"
            io.imsave(os.path.join(output_dir, frame_filename), np.array(frame))
            current_frame += step_size
            print(f"Extracting frame {frame_filename}")
        except Exception as e:
            print(f"Error reading frame {frame_index} from {video_path} with error {e}")
            break




# # Use the function to extract frames for each set
# extract_frames(train_resident, resident_videos_dir, output_dirs['train'], df)
# extract_frames(val_resident, resident_videos_dir, output_dirs['val'], df)
# extract_frames(train_staff, staff_videos_dir, output_dirs['train'], df)
# extract_frames(val_staff, staff_videos_dir, output_dirs['val'], df)

# Extract frames for test videos
extract_frames(test_filenames_resident, resident_videos_dir, output_dirs['test'], df)
extract_frames(test_filenames_staff, staff_videos_dir, output_dirs['test'], df)



