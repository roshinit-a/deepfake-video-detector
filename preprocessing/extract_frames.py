import os
import cv2
import glob
from tqdm import tqdm
from decord import VideoReader, cpu
import numpy as np
def extract_frames(video_path, output_dir, num_frames=30):
    """
    Extracts a specific number of evenly spaced frames from a video file using Decord.
    Decord is significantly faster than OpenCV's VideoCapture for random access and batch reading.
    
    Args:
        video_path (str): The absolute path to the input video file.
        output_dir (str): The directory where the extracted frames will be saved.
        num_frames (int): The total number of frames to sample from the video.
    """
    
    # Check if the output directory for this specific video already exists
    # If it does not exist, create it to organize frames per video
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        # Initialize Decord VideoReader to open the video file
        # ctx=cpu(0) forces it to use the CPU for decoding since decord GPU support can sometimes be fragile
        vr = VideoReader(video_path, ctx=cpu(0))
    except Exception as e:
        # Catch errors if the video is corrupt or cannot be opened by decord
        print(f"Error: Could not open video {video_path}. {str(e)}")
        return

    # Get the total number of frames in the video
    total_frames = len(vr)
    
    # If the video is empty or corrupt, skip it
    if total_frames <= 0:
        print(f"Error: Video {video_path} has 0 frames.")
        return

    # Generate an array of evenly spaced indices using numpy
    # np.linspace creates 'num_frames' values from 0 to 'total_frames - 1'
    # We cast them to integers using astype(int) to use them as array indices
    # This guarantees we exactly hit the number of frames requested, distributed evenly across the video.
    frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)

    try:
        # The true power of Decord: get_batch loads all requested frames efficiently at once
        # This is orders of magnitude faster than iterating frame-by-frame seeking with OpenCV
        frames = vr.get_batch(frame_indices).asnumpy()
    except Exception as e:
        print(f"Warning: Could not read frame batch from {video_path}. {str(e)}")
        return
        
    # Iterate over the resulting multi-dimensional numpy array of frames
    for i, frame in enumerate(frames):
        # Decord reads frames in RGB color space by default
        # OpenCV's cv2.imwrite expects BGR color space, so we must convert it back
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Construct the filename, e.g., "frame_0001.jpg"
        # Using zfill(4) ensures consistent padding (e.g., 0001, 0010)
        frame_filename = os.path.join(output_dir, f"frame_{str(i).zfill(4)}.jpg")
        
        # Write the resulting BGR image array to the disk as a JPG file
        cv2.imwrite(frame_filename, frame_bgr)

def process_dataset(dataset_dir="dataset", frames_dir="frames", num_frames_per_video=30):
    """
    Processes all videos in the dataset and extracts frames.
    
    Args:
        dataset_dir (str): The root directory containing 'real' and 'fake' folders.
        frames_dir (str): The root directory where output frames will be saved.
        num_frames_per_video (int): How many frames to extract from each video.
    """
    
    # Define the categories we expect in our dataset structure
    categories = ["real", "fake"]
    
    # Create the root output directory for frames if it doesn't exist
    os.makedirs(frames_dir, exist_ok=True)

    # Iterate over both 'real' and 'fake' categories
    for category in categories:
        # Construct the path to the current category folder
        category_path = os.path.join(dataset_dir, category)
        
        # Create the corresponding output category folder in the frames directory
        output_category_path = os.path.join(frames_dir, category)
        os.makedirs(output_category_path, exist_ok=True)

        # Use glob to find all MP4 video files recursively within this category
        # Using recursive=True allows it to look inside subfolders like 'original' or 'Deepfakes'
        video_paths = glob.glob(os.path.join(category_path, "**", "*.mp4"), recursive=True)
        
        # Print a status message indicating how many videos were found
        print(f"Found {len(video_paths)} videos in {category_path}")

        # Iterate over all found video paths, using tqdm for a progress bar
        for video_path in tqdm(video_paths, desc=f"Processing {category} videos"):
            # Extract just the filename (e.g., "000.mp4") across OS types
            video_name = os.path.basename(video_path)
            
            # Remove the file extension (e.g., "000.mp4" -> "000")
            video_name_without_ext = os.path.splitext(video_name)[0]
            
            # Get the name of the parent folder (e.g., "original" or "Deepfakes")
            parent_folder = os.path.basename(os.path.dirname(video_path))

            # Construct the final output directory for this specific video's frames
            # Structure: frames/real/original/000/
            video_output_dir = os.path.join(output_category_path, parent_folder, video_name_without_ext)
            
            # Call the frame extraction function on this specific video
            extract_frames(video_path, video_output_dir, num_frames=num_frames_per_video)

if __name__ == "__main__":
    # Start the extraction process using the predefined standard relative paths
    print("Starting video frame extraction...")
    process_dataset()
    print("Frame extraction complete.")
