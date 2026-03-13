import os                           # For file and directory path operations
import glob                          # For recursively finding all face image files
import numpy as np                   # For numerical operations and array handling
import cv2                           # OpenCV for reading images from disk
import torch                         # PyTorch core for tensor operations
from facenet_pytorch import InceptionResnetV1  # FaceNet pre-trained face recognition model
from torchvision import transforms   # Image preprocessing transforms
from PIL import Image as PILImage    # PIL for converting NumPy arrays to PIL format
from tqdm import tqdm                # Progress bar for long-running loops


def load_facenet_model(device):
    """
    Loads a pre-trained FaceNet (InceptionResnetV1) model for face identity embedding.

    FaceNet was trained to produce 512-dimensional embeddings where:
    - The same person's faces cluster closely together in embedding space
    - Different people's faces are far apart
    
    In deepfakes, the identity is often swapped, so consecutive frames may have
    subtle inconsistencies in their identity embeddings, even if they look real.

    Args:
        device (torch.device): Device to run the model on (GPU or CPU).

    Returns:
        torch.nn.Module: The pre-trained FaceNet model in eval mode.
    """
    # Load InceptionResnetV1 pre-trained on VGGFace2 dataset
    # VGGFace2 is a large-scale face recognition dataset with 3.3M images
    # This gives us powerful identity representations for any face
    model = InceptionResnetV1(pretrained='vggface2')

    # Move the model to the appropriate device (GPU for speed)
    model = model.to(device)

    # Set model to evaluation mode to disable dropout and batch norm training behavior
    model.eval()

    return model


def get_facenet_transform():
    """
    Returns the preprocessing pipeline required by FaceNet (InceptionResnetV1).
    FaceNet expects input images of size 160x160, scaled to [-1, 1].

    Returns:
        transforms.Compose: A composed PyTorch image transform pipeline.
    """
    return transforms.Compose([
        # Resize face image to 160x160 pixels as required by InceptionResnetV1 input
        transforms.Resize((160, 160)),

        # Convert PIL Image to a PyTorch tensor, scaling pixel values to [0.0, 1.0]
        transforms.ToTensor(),

        # Normalize pixel values from [0, 1] to [-1, 1] using mean=0.5, std=0.5
        # This matches the normalization FaceNet was trained with
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def compute_identity_consistency(embeddings):
    """
    Given an array of identity embeddings across video frames, computes
    the pairwise cosine similarity between consecutive frames.

    In a real video, the same person's face should produce highly similar
    embeddings frame to frame (cosine similarity close to 1.0).

    In a deepfake, there may be subtle identity shifts causing lower similarity,
    especially around the mouth and eye regions where the synthesis can be imperfect.

    Args:
        embeddings (np.ndarray): Array of shape (num_frames, 512) — one embedding per frame.

    Returns:
        np.ndarray: A 1D array of shape (num_frames - 1,) with cosine similarities
                    between each pair of consecutive frames.
    """
    # Initialize an empty list to hold consecutive-frame similarity scores
    similarities = []

    # Iterate over each consecutive (i, i+1) pair of frame embeddings
    for i in range(len(embeddings) - 1):
        # Get the embedding for frame i (shape: 512,)
        emb_a = embeddings[i]

        # Get the embedding for the next frame i+1 (shape: 512,)
        emb_b = embeddings[i + 1]

        # Compute cosine similarity between the two embedding vectors
        # Formula: cos(θ) = (A · B) / (||A|| × ||B||)
        dot_product  = np.dot(emb_a, emb_b)                  # Dot product of the two vectors
        norm_a       = np.linalg.norm(emb_a)                  # L2 norm of embedding A
        norm_b       = np.linalg.norm(emb_b)                  # L2 norm of embedding B
        cosine_sim   = dot_product / (norm_a * norm_b + 1e-8)  # Add epsilon to avoid divide by zero

        # Append the similarity score for this frame pair
        similarities.append(cosine_sim)

    # Return the list of similarities as a NumPy array
    return np.array(similarities)


def extract_identity_features(face_dir="faces", identity_dir="identity_features"):
    """
    Main pipeline for Phase 9: Identity Consistency Detection.
    
    For each video, it:
    1. Reads all cropped face frames
    2. Extracts a 512-dim FaceNet embedding per frame (on GPU if available)
    3. Computes consecutive-frame cosine similarities
    4. Saves two .npy files per video:
       - embeddings: (num_frames, 512) raw identity vectors
       - similarities: (num_frames-1,) consistency scores between frames

    Args:
        face_dir (str):     Root directory of cropped face images (from Phase 5).
        identity_dir (str): Root directory where identity feature files will be saved.
    """
    # Detect and use GPU if available, otherwise fall back to CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the pre-trained FaceNet model onto the selected device
    model = load_facenet_model(device)

    # Get the image preprocessing transforms required for FaceNet
    transform = get_facenet_transform()

    # Verify that the face directory from Phase 5 actually exists
    if not os.path.exists(face_dir):
        print(f"Error: Face directory '{face_dir}' not found. Run extract_faces.py first.")
        return

    # Create the top-level output directory for identity features
    os.makedirs(identity_dir, exist_ok=True)

    # Recursively find all .jpg face images under faces/real/ and faces/fake/
    face_paths = glob.glob(os.path.join(face_dir, "**", "*.jpg"), recursive=True)
    print(f"Found {len(face_paths)} face images for identity feature extraction.")

    # Group face images by video using the same logic as Phases 6 and 8
    # Dictionary maps video_key -> list of face image paths
    video_to_faces = {}
    for face_path in face_paths:
        # Get path relative to the face_dir root
        relative_path = os.path.relpath(face_path, face_dir)
        path_parts    = relative_path.split(os.sep)  # Split path into its directory components

        # Build the video key from the first 3 parts: category/subset/video_id
        if len(path_parts) >= 3:
            video_key = os.path.join(*path_parts[:3])
        else:
            video_key = os.path.join(*path_parts[:-1])

        # Add this face path to its video group
        video_to_faces.setdefault(video_key, []).append(face_path)

    print(f"Extracting identity features for {len(video_to_faces)} unique videos...")

    # Process each video's group of face frames
    for video_key, paths in tqdm(video_to_faces.items(), desc="Identity features"):
        # Sort paths to maintain consistent temporal frame ordering
        paths = sorted(paths)

        # Accumulate FaceNet embedding tensors for all frames in this video
        batch_tensors = []

        for face_path in paths:
            # Read the face image from disk using OpenCV (BGR color format)
            img_bgr = cv2.imread(face_path)
            if img_bgr is None:
                # Skip corrupt or missing face images
                continue

            # Convert from BGR (OpenCV default) to RGB (expected by FaceNet/PyTorch)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Convert the NumPy array to a PIL Image for PyTorch transforms
            img_pil = PILImage.fromarray(img_rgb)

            # Apply the FaceNet preprocessing pipeline: resize to 160x160, normalize to [-1,1]
            tensor = transform(img_pil)

            # Append the processed tensor (shape: 3x160x160) to the batch
            batch_tensors.append(tensor)

        # Skip video if no valid frames were found
        if not batch_tensors:
            continue

        # Stack all individual frame tensors into a single batched tensor
        # Shape: (num_frames, 3, 160, 160)
        batch = torch.stack(batch_tensors).to(device)

        # Disable gradient computation since we only need the forward pass
        with torch.no_grad():
            # Pass the batch through FaceNet to get identity embeddings
            # Output shape: (num_frames, 512)
            embeddings = model(batch)

        # Move the embeddings tensor back to CPU and convert to NumPy array
        embeddings_np = embeddings.cpu().numpy()

        # Compute consecutive-frame cosine similarities from the embeddings
        # This is our "identity consistency score" — low values may indicate a deepfake
        similarities = compute_identity_consistency(embeddings_np)

        # Construct output directory path mirroring the input face structure
        video_out_dir = os.path.join(identity_dir, video_key)
        os.makedirs(video_out_dir, exist_ok=True)

        # Save the raw embeddings array: shape (num_frames, 512)
        np.save(os.path.join(video_out_dir, "embeddings.npy"), embeddings_np)

        # Save the consecutive-frame similarities: shape (num_frames - 1,)
        np.save(os.path.join(video_out_dir, "similarities.npy"), similarities)

    print(f"\nIdentity feature extraction complete! Features saved to: {identity_dir}")


if __name__ == "__main__":
    # Run the identity consistency extraction pipeline
    extract_identity_features()
