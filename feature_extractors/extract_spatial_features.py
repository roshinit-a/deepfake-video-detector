import os                        # For file and directory path operations
import glob                       # For recursively finding all face image files
import torch                      # PyTorch core library for tensor operations
import torch.nn as nn             # Neural network module from PyTorch
import numpy as np                # For numerical operations and array handling
import cv2                        # OpenCV for reading images from disk
from torchvision import models, transforms   # Pre-trained models and image transforms
from tqdm import tqdm              # Progress bar utility for long loops


def load_resnet50(device):
    """
    Loads a pre-trained ResNet-50 model and modifies it to output feature embeddings.
    The original classification head (final fully connected layer) is removed
    so that the model outputs a raw 2048-dimensional feature vector instead of class scores.

    Args:
        device (torch.device): The device to load and run the model on (CPU or CUDA GPU).

    Returns:
        torch.nn.Module: The modified ResNet-50 model ready for feature extraction.
    """
    # Load ResNet-50 model with pre-trained ImageNet weights
    # Pre-trained means the network already knows powerful, general visual features
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Remove the final classification layer (the "fc" head)
    # By replacing it with Identity(), the model now outputs the 2048-dim feature map
    # instead of 1000 ImageNet class probabilities
    model.fc = nn.Identity()

    # Move the model to the specified device (GPU or CPU)
    # This ensures all tensor operations happen on the correct hardware
    model = model.to(device)

    # Set the model to evaluation mode
    # This disables dropout and batch normalization training behavior
    # which is critically important for consistent, reproducible feature extraction
    model.eval()

    return model


def get_image_transform():
    """
    Returns the standard image transformation pipeline for ResNet-50 input.
    The network was pre-trained on ImageNet data with specific normalization,
    so we must apply the exact same preprocessing to get meaningful embeddings.

    Returns:
        transforms.Compose: A PyTorch transform pipeline.
    """
    # Compose multiple transforms into a single pipeline
    return transforms.Compose([
        # Resize the image to 224x224 pixels (expected input size for ResNet-50)
        # This ensures every face, regardless of original size, is standardized
        transforms.Resize((224, 224)),

        # Convert the image from a NumPy array (H, W, C uint8) to a PyTorch tensor
        # This also normalizes pixel values from [0, 255] to [0.0, 1.0]
        transforms.ToTensor(),

        # Normalize each color channel using the exact ImageNet training mean and std
        # This centers the pixel values and scales them appropriately for the model
        # Without this step, the pre-trained weights would produce garbage features
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean per channel (R, G, B)
            std=[0.229, 0.224, 0.225]    # ImageNet standard deviation per channel
        ),
    ])


def extract_features(face_dir="faces", embedding_dir="embeddings", batch_size=64):
    """
    Iterates over all cropped face images, batches them, and extracts 2048-dimensional
    feature embeddings from a pre-trained ResNet-50 model.
    These embeddings are then saved as .npy files for use in model training.

    Args:
        face_dir (str):       Root directory of cropped face images (output of Phase 5).
        embedding_dir (str):  Root directory where .npy embedding files will be saved.
        batch_size (int):     Number of face images to process in each model forward pass.
    """
    # Detect if a CUDA GPU is available; use it for speed, otherwise fall back to CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the modified ResNet-50 model onto the target device
    model = load_resnet50(device)

    # Load the standard image preprocessing transform pipeline
    transform = get_image_transform()

    # Check that the input face directory from Phase 5 actually exists
    if not os.path.exists(face_dir):
        print(f"Error: Face directory '{face_dir}' not found. Please run face extraction first.")
        return

    # Create the top-level output directory for storing embeddings
    os.makedirs(embedding_dir, exist_ok=True)

    # Recursively find all .jpg face images under faces/real/ and faces/fake/
    face_paths = glob.glob(os.path.join(face_dir, "**", "*.jpg"), recursive=True)
    print(f"Found {len(face_paths)} face images for feature extraction.")

    # Group face images by the video they came from
    # This allows us to store a single embedding .npy file per video
    # Dictionary: video_key -> list of face image paths
    video_to_faces = {}
    for face_path in face_paths:
        # Compute the relative path from the faces root (e.g., real/original/000/frame_0000.jpg)
        relative_path  = os.path.relpath(face_path, face_dir)
        path_parts     = relative_path.split(os.sep)  # Split into list of directory names

        # Reconstruct the "video key" as category/subset/video_id
        # e.g., "real/original/000" or "fake/Deepfakes/000"
        if len(path_parts) >= 3:
            video_key = os.path.join(*path_parts[:3])
        else:
            video_key = os.path.join(*path_parts[:-1])

        # Add this face path to the list for its video
        video_to_faces.setdefault(video_key, []).append(face_path)

    print(f"Processing embeddings for {len(video_to_faces)} unique videos...")

    # Process each video's set of face frames
    for video_key, paths in tqdm(video_to_faces.items(), desc="Extracting embeddings"):
        # Sort face paths so frames are always in consistent temporal order
        paths = sorted(paths)

        # Accumulate tensors for the entire batch of frames belonging to this video
        batch_tensors = []

        for face_path in paths:
            # Read the face image from disk using OpenCV (BGR format)
            img_bgr = cv2.imread(face_path)
            if img_bgr is None:
                # Skip corrupt or unreadable images
                continue

            # Convert from BGR (OpenCV default) to RGB (expected by PyTorch models)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Convert the numpy array to a PIL Image so PyTorch transforms can operate on it
            from PIL import Image as PILImage
            img_pil = PILImage.fromarray(img_rgb)

            # Apply the preprocessing transform pipeline to the image
            # Result is a normalized tensor of shape (3, 224, 224)
            tensor = transform(img_pil)

            # Add this face tensor to the batch list for this video
            batch_tensors.append(tensor)

        # Skip this video entirely if no valid face frames were found
        if not batch_tensors:
            continue

        # Stack all individual face tensors into a single batch tensor
        # Resulting shape: (num_frames, 3, 224, 224)
        batch = torch.stack(batch_tensors).to(device)

        # Disable gradient computation since we only need forward inference
        # This saves memory and speeds up the process significantly
        with torch.no_grad():
            # Pass the entire batch through ResNet-50 up to the penultimate layer
            # Output shape: (num_frames, 2048) — one 2048-dim vector per frame
            embeddings = model(batch)

        # Move the embeddings back to CPU and convert to a NumPy array for disk storage
        embeddings_np = embeddings.cpu().numpy()

        # Construct the output .npy file path mirroring the input directory structure
        embedding_save_path = os.path.join(embedding_dir, video_key + ".npy")

        # Create the parent directory for this embedding file if it doesn't exist
        os.makedirs(os.path.dirname(embedding_save_path), exist_ok=True)

        # Save the embedding array to disk in compressed NumPy format
        # Shape stored: (num_frames, 2048), e.g., (30, 2048)
        np.save(embedding_save_path, embeddings_np)

    print("\nFeature extraction complete! Embeddings saved to:", embedding_dir)


if __name__ == "__main__":
    extract_features()
