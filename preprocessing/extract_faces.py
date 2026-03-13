import os                            # For file and directory operations
import glob                           # For finding all frame images recursively
import cv2                            # OpenCV for image reading and saving
import torch                          # PyTorch for GPU detection
import numpy as np                    # NumPy for image array handling
from facenet_pytorch import MTCNN    # GPU-accelerated face detector
from tqdm import tqdm                 # Progress bar
from PIL import Image as PILImage    # PIL required by MTCNN


def detect_and_crop_faces_fast(
    input_dir="frames",
    output_dir="faces",
    target_size=(224, 224),
    batch_size=32,                    # Number of images per GPU batch
    skip_existing=True,               # If True, skip frames that already have a face output
):
    """
    High-speed face detection and cropping using GPU-batched MTCNN.

    Key improvements over the previous RetinaFace per-frame approach:
    1. Batch processing: feeds 32 images at once to the GPU detector
    2. GPU acceleration: MTCNN runs on CUDA for ~50x speedup vs CPU RetinaFace
    3. Skip logic: frames with existing output are skipped entirely
    4. PIL format: MTCNN natively accepts PIL images

    Args:
        input_dir (str):   Root directory of extracted frames.
        output_dir (str):  Root directory to save cropped face images.
        target_size (tuple): Final face size for model input (width, height).
        batch_size (int):  Number of frames to process per GPU batch.
        skip_existing (bool): Skip frames whose output face already exists.
    """
    # Detect whether a GPU is available and use it for the face detector
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device for face detection: {device}")

    # Initialize MTCNN — a multi-stage CNN face detector from facenet_pytorch
    # keep_all=False: keep only the highest-confidence face per frame (one face per video frame)
    # thresholds: detection confidence for each stage (P-Net, R-Net, O-Net)
    # min_face_size=40: skip tiny faces smaller than 40px (likely background noise)
    mtcnn = MTCNN(
        keep_all=False,
        device=device,
        thresholds=[0.6, 0.7, 0.8],  # Detection confidence thresholds for 3 MTCNN stages
        min_face_size=40,              # Minimum face size in pixels
        margin=20,                     # Extra margin around the face bounding box
        image_size=target_size[0],     # Output face image size (224)
        post_process=False,            # Return raw pixel tensor, not normalized [−1,1]
    )

    # Ensure the output root directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Find all JPG frame images recursively under the input (frames) directory
    all_frame_paths = glob.glob(os.path.join(input_dir, "**", "*.jpg"), recursive=True)
    print(f"Found {len(all_frame_paths)} total frames.")

    # ── Skip already-processed frames ─────────────────────────────────────────
    if skip_existing:
        # Build the expected output path for each frame and check if it exists
        pending_paths = []
        for frame_path in all_frame_paths:
            relative_path = os.path.relpath(frame_path, input_dir)  # Relative path
            output_path   = os.path.join(output_dir, relative_path)  # Mirror in output_dir
            if not os.path.exists(output_path):
                pending_paths.append(frame_path)  # Add only if output is missing
        print(f"Skipping {len(all_frame_paths) - len(pending_paths)} already-processed frames.")
        print(f"Processing {len(pending_paths)} new frames...")
    else:
        pending_paths = all_frame_paths

    # ── Batch Processing Loop ─────────────────────────────────────────────────
    # Group the pending frames into batches of 'batch_size' for GPU processing
    total = len(pending_paths)
    with tqdm(total=total, desc="Face Detection (GPU batch)", unit="frame") as pbar:
        for i in range(0, total, batch_size):
            # Get the current batch of frame paths
            batch_paths = pending_paths[i : i + batch_size]

            # ── Load images as PIL objects (all resized to uniform dims) ──────
            UNIFORM_SIZE = (640, 480)  # Standard size for all input images before batching
            pil_images = []     # PIL images for MTCNN
            valid_paths = []    # Keep track of paths that loaded successfully
            for frame_path in batch_paths:
                img_bgr = cv2.imread(frame_path)   # Read with OpenCV (BGR)
                if img_bgr is None:
                    continue                         # Skip unreadable frames
                # Resize to uniform dimensions so all images in the batch have same size
                # This is REQUIRED by MTCNN batch mode — all images must have equal dimensions
                img_bgr = cv2.resize(img_bgr, UNIFORM_SIZE)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # Convert BGR → RGB
                pil_images.append(PILImage.fromarray(img_rgb))       # Convert to PIL
                valid_paths.append(frame_path)                        # Track valid path

            if not pil_images:
                pbar.update(len(batch_paths))
                continue

            # ── Run MTCNN on the full batch (GPU) ─────────────────────────────
            # face_tensors: list of tensors or None (None = no face detected)
            # Each tensor has shape (3, 224, 224) with pixel values in [0, 255]
            try:
                face_tensors = mtcnn(pil_images)   # One call processes the whole batch on GPU
            except Exception as e:
                print(f"Batch inference error: {e}")
                pbar.update(len(batch_paths))
                continue

            # ── Save detected faces to disk ────────────────────────────────────
            for frame_path, face_tensor in zip(valid_paths, face_tensors):
                if face_tensor is None:
                    # No face detected in this frame — skip
                    pbar.update(1)
                    continue

                # Convert the MTCNN output tensor (3, H, W) → NumPy array (H, W, 3)
                # The tensor contains values in [0.0, 255.0], so we clamp and cast
                face_np = face_tensor.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
                face_np = np.clip(face_np, 0, 255).astype(np.uint8)   # Ensure valid uint8
                face_bgr = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)   # Convert RGB → BGR for cv2

                # Resize to target size (224×224) if not already (MTCNN's image_size handles this)
                if face_bgr.shape[:2] != target_size:
                    face_bgr = cv2.resize(face_bgr, target_size)

                # Construct output path mirroring the input directory structure
                relative_path = os.path.relpath(frame_path, input_dir)
                output_path   = os.path.join(output_dir, relative_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # Save the cropped, resized face image to disk
                cv2.imwrite(output_path, face_bgr)
                pbar.update(1)

    # Count how many face images were saved in total
    total_saved = len(glob.glob(os.path.join(output_dir, "**", "*.jpg"), recursive=True))
    print(f"\nFace extraction complete! Total face images saved: {total_saved}")


if __name__ == "__main__":
    print("Starting GPU-accelerated face detection with MTCNN...")
    detect_and_crop_faces_fast()
    print("Done.")
