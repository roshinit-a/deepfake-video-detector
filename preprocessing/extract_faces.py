import os
import cv2
import glob
from tqdm import tqdm
from retinaface import RetinaFace

def detect_and_crop_faces(input_dir="frames", output_dir="faces", target_size=(224, 224)):
    """
    Reads extracted frames, detects faces using RetinaFace, crops them,
    and resizes them to the target size for CNN feature extraction.
    
    Args:
        input_dir (str): Directory containing the extracted video frames.
        output_dir (str): Directory to save the cropped face images.
        target_size (tuple): The (width, height) to resize the cropped faces to.
    """
    
    # Check if the input directory (frames) exists before proceeding
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' not found. Please run frame extraction first.")
        return

    # Create the root output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Use glob to find all JPG frame images recursively in the input directory
    # This captures paths like: frames/fake/Deepfakes/000/frame_0000.jpg
    frame_paths = glob.glob(os.path.join(input_dir, "**", "*.jpg"), recursive=True)
    
    # Output how many frames we found
    print(f"Found {len(frame_paths)} frames to process for face detection.")

    # Iterate over all found frames using tqdm for a progress bar
    for frame_path in tqdm(frame_paths, desc="Detecting & Cropping Faces", unit="frame"):
        try:
            # Read the image frame using OpenCV
            img_np = cv2.imread(frame_path)
            if img_np is None:
                print(f"Warning: Could not read image {frame_path}. Skipping.")
                continue

            # Detect faces in the image using RetinaFace
            # This returns a dictionary of detected faces and their bounding boxes/landmarks
            # RetinaFace natively works with BGR OpenCV images or paths.
            resp = RetinaFace.detect_faces(frame_path)
            
            # If RetinaFace returns an empty tuple or dict, no face was found
            if not isinstance(resp, dict) or len(resp) == 0:
                continue

            # Iterate through all detected faces in the current frame
            # Although usually there is just one primary face, we handle multiple just in case
            for face_key, face_data in resp.items():
                
                # Extract the bounding box coordinates [x1, y1, x2, y2]
                box = face_data["facial_area"]
                
                # Unpack the coordinates for clarity and convert to integers
                x1, y1, x2, y2 = [int(b) for b in box]
                
                # Expand the bounding box slightly (optional padding) to ensure 
                # we don't cut off the chin or forehead too tightly. 
                # We add a 10% margin on all sides.
                h, w = img_np.shape[:2] # Get original image height and width
                margin_x = int((x2 - x1) * 0.1)
                margin_y = int((y2 - y1) * 0.1)
                
                # Apply margins while keeping coordinates within image boundaries
                nx1 = max(0, x1 - margin_x)
                ny1 = max(0, y1 - margin_y)
                nx2 = min(w, x2 + margin_x)
                ny2 = min(h, y2 + margin_y)

                # Crop the face from the original numpy image (using BGR for saving)
                cropped_face = img_np[ny1:ny2, nx1:nx2]
                
                # Check if the crop is valid (not empty)
                if cropped_face.size == 0:
                    continue

                # Resize the cropped face to the target size (e.g., 224x224) 
                # This is a strict requirement for models like ResNet50
                resized_face = cv2.resize(cropped_face, target_size)

                # Construct the output path by mirroring the input directory structure
                # E.g., replace 'frames/' with 'faces/' in the path
                relative_path = os.path.relpath(frame_path, input_dir)
                output_path = os.path.join(output_dir, relative_path)
                
                # If multiple faces were found in one frame, append an index to the filename
                if len(resp) > 1:
                    base, ext = os.path.splitext(output_path)
                    output_path = f"{base}_{face_key}{ext}"

                # Create the specific subdirectories for this output file
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # Save the resized, cropped face to the disk
                cv2.imwrite(output_path, resized_face)
                
        except Exception as e:
            # Catch and log any unexpected errors during processing a specific frame
            print(f"Error processing {frame_path}: {str(e)}")

if __name__ == "__main__":
    print("Starting face detection, cropping, and resizing using RetinaFace...")
    detect_and_crop_faces()
    print("Face processing complete.")
