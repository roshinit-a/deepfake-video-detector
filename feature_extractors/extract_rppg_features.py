import os                           # For file and directory path operations
import glob                          # For finding all face image files recursively
import numpy as np                   # For numerical operations and array handling
import cv2                           # OpenCV for reading images from disk
from scipy.signal import butter, filtfilt, welch  # Signal processing: Butterworth bandpass filter & Power Spectral Density
from tqdm import tqdm                # Progress bar for long-running loops


def extract_rgb_signal(face_paths):
    """
    Extracts the mean RGB color value from the forehead region of each face frame.

    The rPPG (remote Photoplethysmography) technique works because blood absorbs
    and reflects light differently depending on oxygenation level. As the heart
    pumps blood, the skin color very subtly changes in the green channel.

    Real faces: this pulse signal creates a periodic, naturally rhythmic signal.
    Deepfakes: the synthesis process destroys this biological signal, leaving
               only random noise with no physiological rhythm.

    Args:
        face_paths (list): Sorted list of face image file paths for one video.

    Returns:
        tuple: Three 1D NumPy arrays (r_signal, g_signal, b_signal) of mean pixel
               values for each frame, one value per channel per frame.
    """
    r_signal, g_signal, b_signal = [], [], []  # Lists to hold per-frame mean RGB values

    for face_path in face_paths:
        # Read the face image from disk using OpenCV (BGR color format by default)
        img = cv2.imread(face_path)
        if img is None:
            # Skip corrupt or missing images silently
            continue

        # Resize to a fixed 224x224 to ensure consistent sampling region
        img = cv2.resize(img, (224, 224))

        # Define the forehead region of interest (ROI)
        # The forehead is the most stable skin region for rPPG — less affected by mouth/eye movement
        # We take the top-center strip: rows 10-60, columns 70-155
        h, w = img.shape[:2]                          # Get height and width of the image
        y1, y2 = int(h * 0.05), int(h * 0.25)       # Forehead row bounds: top 5% to 25%
        x1, x2 = int(w * 0.30), int(w * 0.70)       # Horizontal bounds: center 40% of face
        forehead = img[y1:y2, x1:x2]                 # Crop the forehead region

        # Compute the mean pixel value for each channel over the forehead ROI
        # OpenCV uses BGR ordering so: channel 2=R, channel 1=G, channel 0=B
        b_mean = np.mean(forehead[:, :, 0])           # Mean blue channel value
        g_mean = np.mean(forehead[:, :, 1])           # Mean green channel value (most sensitive to pulse)
        r_mean = np.mean(forehead[:, :, 2])           # Mean red channel value

        # Append per-frame means to corresponding channel signal lists
        r_signal.append(r_mean)
        g_signal.append(g_mean)
        b_signal.append(b_mean)

    # Convert Python lists to NumPy arrays for signal processing
    return np.array(r_signal), np.array(g_signal), np.array(b_signal)


def bandpass_filter(signal, fps=30, low_hz=0.7, high_hz=3.0):
    """
    Applies a Butterworth bandpass filter to isolate the heart rate frequency band.
    Uses second-order sections (SOS) representation which is numerically stable
    and avoids the padlen crash that occurs with ba-form filters on short signals.

    Normal resting heart rate is 42–180 BPM = 0.7–3.0 Hz.

    Args:
        signal (np.ndarray): 1D input signal (e.g., mean green channel per frame).
        fps (int):           Frames per second of the original video.
        low_hz (float):      Lower cutoff frequency in Hz (42 BPM = 0.7 Hz).
        high_hz (float):     Upper cutoff frequency in Hz (180 BPM = 3.0 Hz).

    Returns:
        np.ndarray: Bandpass-filtered signal of the same length.
    """
    from scipy.signal import sosfiltfilt, butter  # Use SOS-form filter for stability

    # Need at least 15 frames for meaningful filtering; return signal unchanged otherwise
    if len(signal) < 15:
        return signal

    # Compute normalized cutoff frequencies (fraction of Nyquist = fps/2)
    nyquist = fps / 2.0
    low  = low_hz  / nyquist
    high = high_hz / nyquist

    # Validate cutoff range
    if not (0.0 < low < 1.0 and 0.0 < high < 1.0 and low < high):
        return signal

    # Design a 2nd-order Butterworth bandpass filter in SOS form
    # SOS avoids the numerical instability of ba form on long filter chains
    # sosfiltfilt uses padlen = 3 * max(sos_order), which is smaller than ba padlen
    sos = butter(2, [low, high], btype='band', output='sos')

    # Apply the filter zero-phase (forward + backward) using the SOS representation
    filtered = sosfiltfilt(sos, signal)

    return filtered




def compute_rppg_features(r_signal, g_signal, b_signal, fps=30):
    """
    Given filtered RGB signals from the forehead, computes rPPG-based features for ML.

    Features include:
    1. The filtered green channel signal itself (primary rPPG signal)
    2. Power Spectral Density (PSD) — the strength of each frequency in the signal
    3. Peak heart rate frequency — where is the pulse peak?
    4. Signal-to-noise ratio of the heart rate band

    Args:
        r_signal (np.ndarray): Mean red channel values over time.
        g_signal (np.ndarray): Mean green channel values over time (primary rPPG signal).
        b_signal (np.ndarray): Mean blue channel values over time.
        fps (int): Frames per second.

    Returns:
        np.ndarray: A 1D feature vector representing the rPPG signal characteristics.
    """
    # The green channel is the most sensitive to blood volume changes
    # because hemoglobin absorbs green light most strongly
    filtered_g = bandpass_filter(g_signal, fps=fps)

    # Compute Power Spectral Density (PSD) of the filtered green signal
    # PSD tells us how much "power" (variation) exists at each frequency
    # freqs = frequency values, psd = power at each frequency
    freqs, psd = welch(filtered_g, fs=fps, nperseg=min(len(filtered_g), 64))

    # Normalize the PSD so values are on a consistent scale regardless of brightness
    psd_normalized = psd / (np.sum(psd) + 1e-8)

    # Find the dominant frequency in the heart rate band (0.7–3.0 Hz)
    # This is our estimated heart rate
    hr_mask         = (freqs >= 0.7) & (freqs <= 3.0)  # Mask for the HR band
    hr_psd          = psd[hr_mask]                      # PSD values in the HR band
    hr_freqs        = freqs[hr_mask]                    # Frequencies in the HR band

    # Identify the frequency with the highest power in the HR band
    peak_hr_freq = hr_freqs[np.argmax(hr_psd)] if len(hr_psd) > 0 else 0.0

    # Signal statistics: mean and std of the filtered signal
    # These measure the amplitude and variability of the detected pulse
    signal_mean = np.mean(filtered_g)     # Average value of the filtered pulse signal
    signal_std  = np.std(filtered_g)      # Standard deviation (variability of pulse)

    # Concatenate all features into a single 1D feature vector
    # [signal samples, normalized PSD, peak HR frequency, signal mean, signal std]
    features = np.concatenate([
        filtered_g,           # The raw filtered signal (num_frames values)
        psd_normalized,       # Normalized PSD spectrum
        [peak_hr_freq],       # Dominant heart rate frequency (scalar)
        [signal_mean],        # Mean signal amplitude (scalar)
        [signal_std],         # Signal standard deviation (scalar)
    ])

    return features


def extract_rppg_features(face_dir="faces", rppg_dir="rppg_features", fps=30):
    """
    Main pipeline for Phase 10: Physiological Signal Detection.

    For each video, it:
    1. Reads all sorted face frames
    2. Extracts mean RGB values from the forehead region
    3. Applies a bandpass filter to isolate the heart rate band
    4. Computes rPPG feature vector and saves it as a .npy file

    Args:
        face_dir (str):   Root directory of cropped face images (from Phase 5).
        rppg_dir (str):   Output directory for rPPG feature files.
        fps (int):        Assumed frames per second of original videos.
    """

    # Verify the face directory from Phase 5 exists
    if not os.path.exists(face_dir):
        print(f"Error: Face directory '{face_dir}' not found. Run extract_faces.py first.")
        return

    # Create the output directory for rPPG features
    os.makedirs(rppg_dir, exist_ok=True)

    # Recursively find all .jpg face images under the faces/ tree
    face_paths = glob.glob(os.path.join(face_dir, "**", "*.jpg"), recursive=True)
    print(f"Found {len(face_paths)} face images for rPPG extraction.")

    # Group face images by video — same grouping logic as Phases 6, 8, and 9
    video_to_faces = {}
    for face_path in face_paths:
        relative_path = os.path.relpath(face_path, face_dir)   # Relative path from faces/ root
        path_parts    = relative_path.split(os.sep)             # Split into directory parts

        # Build a video key: category/subset/video_id (first 3 path segments)
        if len(path_parts) >= 3:
            video_key = os.path.join(*path_parts[:3])
        else:
            video_key = os.path.join(*path_parts[:-1])

        video_to_faces.setdefault(video_key, []).append(face_path)

    print(f"Processing rPPG signals for {len(video_to_faces)} unique videos...")

    # Process each video
    for video_key, paths in tqdm(video_to_faces.items(), desc="rPPG extraction"):
        # Sort paths to ensure chronological frame order
        paths = sorted(paths)

        # Step 1: Extract mean RGB from forehead ROI for each frame
        r_signal, g_signal, b_signal = extract_rgb_signal(paths)

        # Skip video if no valid frames were found
        if len(g_signal) < 5:
            # Need at least 5 frames for meaningful signal processing
            continue

        # Step 2: Compute rPPG features (filtered signal + PSD + stats)
        features = compute_rppg_features(r_signal, g_signal, b_signal, fps=fps)

        # Construct output save path mirroring input structure
        save_path = os.path.join(rppg_dir, video_key + ".npy")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save the rPPG feature vector as a NumPy binary file
        np.save(save_path, features)

    print(f"\nrPPG feature extraction complete! Features saved to: {rppg_dir}")


if __name__ == "__main__":
    # Run the rPPG physiological signal extraction pipeline
    extract_rppg_features()
