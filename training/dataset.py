import os                             # For file and directory operations
import glob                            # For finding .npy feature files recursively
import numpy as np                     # For loading .npy arrays from disk
import torch                           # Core PyTorch library
from torch.utils.data import Dataset  # Base class for all PyTorch datasets


class DeepfakeDataset(Dataset):
    """
    Custom PyTorch Dataset for the multi-branch deepfake detection system.

    Each sample consists of 4 pre-extracted feature files for a single video:
    - Spatial embeddings   (ResNet-50) → embeddings/
    - Frequency features  (DCT)       → frequency_features/
    - Identity scores     (FaceNet)   → identity_features/<video>/similarities.npy
    - rPPG features       (Phase 10)  → rppg_features/

    Labels:
        0 = Real video
        1 = Fake (deepfake) video
    """

    def __init__(
        self,
        embeddings_dir="embeddings",          # Root dir: spatial .npy files
        freq_dir="frequency_features",        # Root dir: DCT .npy files
        identity_dir="identity_features",     # Root dir: identity similarity dirs
        rppg_dir="rppg_features",             # Root dir: rPPG .npy files
        seq_len=30,                            # Number of frames to expect per video
        freq_dim=192,                          # DCT feature dimension per frame
        identity_dim=29,                       # Number of identity similarity scores
        rppg_dim=97,                           # rPPG feature vector dimension
    ):
        super().__init__()

        # Store all configuration as instance variables
        self.embeddings_dir  = embeddings_dir
        self.freq_dir        = freq_dir
        self.identity_dir    = identity_dir
        self.rppg_dir        = rppg_dir
        self.seq_len         = seq_len
        self.freq_dim        = freq_dim
        self.identity_dim    = identity_dim
        self.rppg_dim        = rppg_dim

        # Build the sample list: each entry is a (video_key, label) pair
        self.samples = []
        self._build_sample_list()

    def _build_sample_list(self):
        """
        Scans the spatial embeddings directory to build the list of all valid samples.
        A sample is valid if ALL 4 feature files exist (spatial, freq, identity, rPPG).
        Labels are derived from the directory structure: 'real' → 0, 'fake' → 1.
        """
        # Find all .npy spatial embedding files recursively
        emb_paths = glob.glob(
            os.path.join(self.embeddings_dir, "**", "*.npy"), recursive=True
        )

        for emb_path in emb_paths:
            # Compute the video key relative to the embeddings root
            # e.g., "real/original/000"
            video_key = os.path.splitext(
                os.path.relpath(emb_path, self.embeddings_dir)
            )[0]

            # Determine the label from the first path component
            # "real" → 0, anything else (fake, Deepfakes, etc.) → 1
            top_level = video_key.split(os.sep)[0].lower()
            label = 0 if top_level == "real" else 1

            # Build paths for the other three feature files
            freq_path     = os.path.join(self.freq_dir, video_key + ".npy")
            identity_path = os.path.join(self.identity_dir, video_key, "similarities.npy")
            rppg_path     = os.path.join(self.rppg_dir, video_key + ".npy")

            # Only include this sample if ALL 4 feature files exist on disk
            if all(os.path.exists(p) for p in [emb_path, freq_path, identity_path, rppg_path]):
                self.samples.append((video_key, label, emb_path, freq_path, identity_path, rppg_path))

        print(f"Dataset built: {len(self.samples)} valid samples "
              f"({sum(1 for s in self.samples if s[1]==0)} real, "
              f"{sum(1 for s in self.samples if s[1]==1)} fake)")

    def __len__(self):
        """Returns the total number of valid video samples in the dataset."""
        return len(self.samples)

    def _pad_or_trim(self, arr, target_len, feature_dim):
        """
        Pads or trims a 2D array of shape (actual_len, feature_dim) to (target_len, feature_dim).
        This ensures all samples have the same sequence length, which is required for batching.

        Args:
            arr (np.ndarray): Input array of shape (T, D).
            target_len (int): Desired sequence length T_target.
            feature_dim (int): Feature dimension D.

        Returns:
            np.ndarray: Array of shape (target_len, feature_dim).
        """
        T = arr.shape[0]
        if T >= target_len:
            # Trim: take the first 'target_len' frames
            return arr[:target_len, :]
        else:
            # Pad: add zero rows at the end to reach 'target_len'
            pad = np.zeros((target_len - T, feature_dim), dtype=np.float32)
            return np.concatenate([arr, pad], axis=0)

    def _fix_len(self, arr, target_len):
        """
        Pads or trims a 1D array to exactly 'target_len' elements.
        Used for identity similarities and rPPG vectors.
        """
        if len(arr) >= target_len:
            return arr[:target_len]          # Trim to target length
        else:
            pad = np.zeros(target_len - len(arr), dtype=np.float32)
            return np.concatenate([arr, pad])  # Zero-pad to target length

    def __getitem__(self, idx):
        """
        Loads and returns one sample as a dictionary of tensors.

        Returns:
            dict: {
                'spatial':   FloatTensor (seq_len, 2048),
                'frequency': FloatTensor (seq_len, 192),
                'identity':  FloatTensor (identity_dim,),
                'rppg':      FloatTensor (rppg_dim,),
                'label':     FloatTensor scalar (0.0 or 1.0),
            }
        """
        _, label, emb_path, freq_path, identity_path, rppg_path = self.samples[idx]

        # ── Load spatial embeddings: (T, 2048) ───────────────────────────────
        spatial = np.load(emb_path).astype(np.float32)
        spatial = self._pad_or_trim(spatial, self.seq_len, 2048)

        # ── Load DCT frequency features: (T, 192) ────────────────────────────
        freq = np.load(freq_path).astype(np.float32)
        freq = self._pad_or_trim(freq, self.seq_len, self.freq_dim)

        # ── Load identity similarity scores: (identity_dim,) ─────────────────
        identity = np.load(identity_path).astype(np.float32)
        identity = self._fix_len(identity, self.identity_dim)

        # ── Load rPPG features: (rppg_dim,) ──────────────────────────────────
        rppg = np.load(rppg_path).astype(np.float32)
        rppg = self._fix_len(rppg, self.rppg_dim)

        # ── Assemble and return as a dictionary of PyTorch tensors ───────────
        return {
            "spatial":   torch.from_numpy(spatial),              # (seq_len, 2048)
            "frequency": torch.from_numpy(freq),                 # (seq_len, 192)
            "identity":  torch.from_numpy(identity),             # (identity_dim,)
            "rppg":      torch.from_numpy(rppg),                 # (rppg_dim,)
            "label":     torch.tensor(label, dtype=torch.float32),  # scalar (0.0 or 1.0)
        }
