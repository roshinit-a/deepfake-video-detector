# Phase 4 Completion Report: Video Frame Extraction

## Objective Achieved
We successfully developed a Python script utilizing OpenCV to iteratively extract evenly-spaced frames from all videos in the dataset. Models process frames (images), not raw videos, so this prepares our data for spatial and temporal analysis.

## Actions Taken
1. **Script Development**:
   - Created `preprocessing/extract_frames.py`.
   - The script uses `cv2.VideoCapture` to read videos efficiently.
   - It computes the step size based on total video length to extract a uniform subset of frames (defaulting to 30 frames per video to capture the entire timeline without overloading storage).
   - As requested, **every line of code** in this script has been thoroughly commented to explain the exact logic, control flow, and OpenCV functions used.
2. **Directory Management**:
   - The script recursively identifies all `.mp4` files within `dataset/real` and `dataset/fake`.
   - It automatically mirrors the dataset structure into the new `frames/` root directory (e.g., creating `frames/fake/Deepfakes/video_name/frame_0001.jpg`).
3. **Progress Tracking**:
   - Integrated the `tqdm` library to provide a real-time progress bar loop to monitor processing status over large datasets like FaceForensics++.

## Readiness for Next Phase
The frame extraction pipeline is ready. By running `python preprocessing/extract_frames.py`, the system will convert all videos into sequences of stored images. We are now ready to proceed to **Phase 5: Face Detection & Alignment**, where we will locate and crop the specific faces out of these raw, wide-angle frames.
