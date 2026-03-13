import torch                        # Core PyTorch library
import torch.nn as nn               # Neural network modules and layers
import torch.nn.functional as F     # Functional operations (activations, etc.)


# ──────────────────────────────────────────────────────────────────────────────
# Branch 1: Spatial Branch (ResNet-50 backbone)
# Receives: raw face frames → outputs 2048-dim spatial features
# ──────────────────────────────────────────────────────────────────────────────

class SpatialBranch(nn.Module):
    """
    Encodes the 2048-dimensional spatial feature vector (from ResNet-50 in Phase 6)
    into a compact 256-dimensional representation.

    This branch captures: skin texture anomalies, blending artifacts,
    unnatural lighting, and pixel-level inconsistencies.
    """

    def __init__(self, input_dim=2048, output_dim=256):
        """
        Args:
            input_dim (int):  Input size — matches ResNet-50 output (2048).
            output_dim (int): Size of the compressed spatial representation.
        """
        super(SpatialBranch, self).__init__()

        # Global Average Pooling over the sequence dimension
        # Converts (B, seq_len, 2048) → (B, 2048) by averaging across frames
        # This reduces the temporal sequence to a single spatial summary
        self.global_avg_pool = lambda x: x.mean(dim=1)

        # A two-layer MLP to compress and refine spatial features
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),   # Reduce 2048-dim to 512-dim
            nn.BatchNorm1d(512),         # Normalize activations for training stability
            nn.ReLU(),                   # Non-linear activation
            nn.Dropout(0.3),             # Dropout regularization to prevent overfitting
            nn.Linear(512, output_dim),  # Further compress to 256-dim
            nn.ReLU(),                   # Final activation
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Shape (batch, seq_len, 2048) — spatial embeddings per frame.
        Returns:
            Tensor: Shape (batch, 256) — compressed spatial feature.
        """
        x = self.global_avg_pool(x)   # Average across the frame sequence: (B, 2048)
        x = self.encoder(x)            # Compress to 256-dim: (B, 256)
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Branch 2: Temporal Branch (BiLSTM)
# Receives: sequence of spatial embeddings → outputs 256-dim temporal features
# ──────────────────────────────────────────────────────────────────────────────

class TemporalBranch(nn.Module):
    """
    Processes the temporal sequence of spatial frame embeddings using a BiLSTM.
    Captures: motion inconsistencies, flickering, unnatural transitions between frames.

    This is the same BiLSTM architecture from Phase 7, integrated here as a branch.
    """

    def __init__(self, input_dim=2048, hidden_dim=256, num_layers=2, output_dim=256):
        """
        Args:
            input_dim (int):  Input feature size per frame (ResNet-50 output = 2048).
            hidden_dim (int): BiLSTM hidden state size (output is 2 * hidden_dim).
            num_layers (int): Number of BiLSTM layers to stack.
            output_dim (int): Final output size after the projection head.
        """
        super(TemporalBranch, self).__init__()

        # Bidirectional LSTM: processes the frame sequence forward and backward
        # Output size = hidden_dim * 2 because it concatenates both directions
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,                           # Input shape is (B, L, F)
            bidirectional=True,                         # Forward + backward passes
            dropout=0.3 if num_layers > 1 else 0.0     # Dropout between LSTM layers
        )

        # Project the BiLSTM output from (hidden_dim * 2) → output_dim
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),  # Project 512 → 256
            nn.ReLU(),                               # Non-linear activation
            nn.Dropout(0.3),                         # Dropout for regularization
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Shape (batch, seq_len, 2048) — spatial embeddings over time.
        Returns:
            Tensor: Shape (batch, 256) — temporal feature vector.
        """
        lstm_out, _ = self.lstm(x)         # BiLSTM output: (B, L, hidden_dim * 2)
        pooled      = lstm_out.mean(dim=1) # Global average pool over time: (B, hidden_dim*2)
        out         = self.proj(pooled)    # Project to 256-dim: (B, 256)
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Branch 3: Frequency Branch (DCT features)
# Receives: (num_frames, 192) frequency feature array → outputs 256-dim features
# ──────────────────────────────────────────────────────────────────────────────

class FrequencyBranch(nn.Module):
    """
    Encodes DCT-based frequency features extracted in Phase 8.
    Captures: GAN checkerboard artifacts and unnatural frequency distributions.

    Input: flattened or averaged DCT feature vector of shape (batch, freq_dim).
    """

    def __init__(self, input_dim=192, output_dim=256):
        """
        Args:
            input_dim (int): Dimension of frequency features (64 coefficients × 3 channels = 192).
            output_dim (int): Size of the compressed frequency representation.
        """
        super(FrequencyBranch, self).__init__()

        # Simple MLP to encode the frequency feature vector
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),    # Expand to intermediate representation
            nn.BatchNorm1d(128),          # Batch normalization for training stability
            nn.ReLU(),                    # Non-linear activation
            nn.Dropout(0.3),              # Dropout regularization
            nn.Linear(128, output_dim),   # Project to 256-dim output
            nn.ReLU(),                    # Final non-linearity
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Shape (batch, seq_len, 192) — DCT features per frame.
        Returns:
            Tensor: Shape (batch, 256) — compressed frequency feature.
        """
        x = x.mean(dim=1)    # Average DCT features across frames: (B, 192)
        x = self.encoder(x)  # Encode to 256-dim: (B, 256)
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Branch 4: Identity + rPPG Branch
# Receives: cosine similarity scores + rPPG signal → outputs 256-dim features
# ──────────────────────────────────────────────────────────────────────────────

class IdentityRPPGBranch(nn.Module):
    """
    Encodes the identity consistency scores (Phase 9) and rPPG signal (Phase 10)
    into a combined feature vector.

    Identity scores indicate how much the face "shifts" between frames.
    rPPG features indicate the presence or absence of a biological heartbeat signal.
    """

    def __init__(self, identity_dim=29, rppg_dim=97, output_dim=256):
        """
        Args:
            identity_dim (int): Number of consecutive-frame similarity scores (num_frames - 1 = 29).
            rppg_dim (int):     Dimension of the rPPG feature vector (varies; defaults to 97).
            output_dim (int):   Final compressed output size.
        """
        super(IdentityRPPGBranch, self).__init__()

        # Concatenate identity + rPPG features, then pass through MLP
        combined_dim = identity_dim + rppg_dim   # Total input dimensions

        self.encoder = nn.Sequential(
            nn.Linear(combined_dim, 128),  # Compress the concatenated features
            nn.BatchNorm1d(128),           # Normalize for training stability
            nn.ReLU(),                     # Non-linear activation
            nn.Dropout(0.3),               # Dropout for regularization
            nn.Linear(128, output_dim),    # Project to 256-dim output
            nn.ReLU(),                     # Final activation
        )

    def forward(self, identity_scores, rppg_features):
        """
        Args:
            identity_scores (Tensor): Shape (batch, 29) — consecutive cosine similarities.
            rppg_features (Tensor):   Shape (batch, rppg_dim) — rPPG feature vector.
        Returns:
            Tensor: Shape (batch, 256) — combined identity + rPPG feature.
        """
        # Concatenate identity and rPPG features along the feature dimension
        x = torch.cat([identity_scores, rppg_features], dim=1)  # (B, identity_dim + rppg_dim)
        x = self.encoder(x)                                       # (B, 256)
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Full Fusion Model: Combines all 4 branches
# ──────────────────────────────────────────────────────────────────────────────

class DeepfakeFusionModel(nn.Module):
    """
    The complete multi-branch deepfake detection model.

    Architecture:
        Branch 1 (Spatial):          ResNet-50 embeddings → 256-dim
        Branch 2 (Temporal):         BiLSTM on frame sequence → 256-dim
        Branch 3 (Frequency):        DCT features → 256-dim
        Branch 4 (Identity + rPPG):  Similarity + pulse → 256-dim
                                              ↓
                          Concatenate all 4 branches → 1024-dim
                                              ↓
                              Fusion MLP with attention → 512-dim
                                              ↓
                              Final classifier → 1 logit (Real vs Fake)
    """

    def __init__(
        self,
        spatial_dim=2048,     # ResNet-50 output dimension
        freq_dim=192,         # DCT feature dimension (64 coefficients * 3 channels)
        identity_dim=29,      # Number of consecutive-frame similarity scores
        rppg_dim=97,          # rPPG feature vector dimension
        branch_dim=256,       # Output dimension of each branch
        num_classes=1         # 1 output for binary classification (BCEWithLogitsLoss)
    ):
        super(DeepfakeFusionModel, self).__init__()

        # Instantiate all four branches
        self.spatial_branch   = SpatialBranch(input_dim=spatial_dim, output_dim=branch_dim)
        self.temporal_branch  = TemporalBranch(input_dim=spatial_dim, output_dim=branch_dim)
        self.frequency_branch = FrequencyBranch(input_dim=freq_dim, output_dim=branch_dim)
        self.identity_branch  = IdentityRPPGBranch(identity_dim=identity_dim, rppg_dim=rppg_dim, output_dim=branch_dim)

        # Total fused dimension = 4 branches × branch_dim
        fused_dim = branch_dim * 4  # 256 * 4 = 1024

        # Learned attention (channel weighting) over the 4 branches
        # This lets the model learn which branch is most discriminative per sample
        self.attention = nn.Sequential(
            nn.Linear(fused_dim, 4),     # Compute 4 scalar attention weights
            nn.Softmax(dim=-1),          # Normalize weights to sum to 1
        )

        # Fusion MLP: takes the attention-weighted concatenation and classifies it
        self.fusion_head = nn.Sequential(
            nn.Linear(fused_dim, 512),   # First fusion layer: 1024 → 512
            nn.BatchNorm1d(512),         # Batch normalization
            nn.ReLU(),                   # Non-linear activation
            nn.Dropout(0.4),             # Stronger dropout at the classification stage
            nn.Linear(512, 128),         # Second fusion layer: 512 → 128
            nn.ReLU(),                   # Non-linear activation
            nn.Dropout(0.3),             # Dropout
            nn.Linear(128, num_classes), # Final output logit
        )

    def forward(self, spatial_emb, freq_emb, identity_scores, rppg_features):
        """
        Full forward pass through all branches and the fusion head.

        Args:
            spatial_emb     (Tensor): (B, seq_len, 2048) — ResNet-50 frame embeddings
            freq_emb        (Tensor): (B, seq_len, 192)  — DCT frequency features
            identity_scores (Tensor): (B, 29)            — consecutive cosine similarities
            rppg_features   (Tensor): (B, rppg_dim)      — rPPG feature vector

        Returns:
            Tensor: (B, 1) — raw logit for Binary CrossEntropy loss
        """
        # ── Run each branch independently ──────────────────────────────────────
        s_feat = self.spatial_branch(spatial_emb)                         # (B, 256)
        t_feat = self.temporal_branch(spatial_emb)                        # (B, 256)
        f_feat = self.frequency_branch(freq_emb)                          # (B, 256)
        i_feat = self.identity_branch(identity_scores, rppg_features)     # (B, 256)

        # ── Concatenate all 4 branch outputs ───────────────────────────────────
        fused = torch.cat([s_feat, t_feat, f_feat, i_feat], dim=1)        # (B, 1024)

        # ── Compute attention weights over each branch ─────────────────────────
        # attn_weights: (B, 4) — one weight per branch per sample
        attn_weights = self.attention(fused)                               # (B, 4)

        # Split the fused vector back into 4 branch chunks for weighting
        # Each split has shape (B, 256)
        branch_feats = torch.stack([s_feat, t_feat, f_feat, i_feat], dim=1)  # (B, 4, 256)

        # Multiply each branch feature by its learned attention weight
        # attn_weights.unsqueeze(-1) → (B, 4, 1) for broadcasting
        weighted = branch_feats * attn_weights.unsqueeze(-1)    # (B, 4, 256)

        # Flatten the attention-weighted features back to 1D
        weighted_flat = weighted.view(weighted.size(0), -1)     # (B, 1024)

        # ── Pass through the final fusion MLP classifier ───────────────────────
        logit = self.fusion_head(weighted_flat)                 # (B, 1)

        return logit


def get_fusion_model(device='cpu', **kwargs):
    """
    Utility function to create and move the fusion model to the target device.

    Args:
        device (str or torch.device): Target compute device.
        **kwargs: Any constructor arguments to override defaults.

    Returns:
        DeepfakeFusionModel: Initialized model on the target device.
    """
    model = DeepfakeFusionModel(**kwargs)  # Build the model with given hyperparameters
    model = model.to(device)               # Move all parameters to the target device
    return model


if __name__ == "__main__":
    # ── Self-test: verify the entire forward pass with dummy tensors ────────────
    print("Testing DeepfakeFusionModel forward pass...")

    # Create dummy input tensors matching expected shapes
    B = 4           # Batch size
    L = 30          # Sequence length (number of frames per video)

    dummy_spatial   = torch.randn(B, L, 2048)  # ResNet-50 spatial embeddings
    dummy_freq      = torch.randn(B, L, 192)   # DCT frequency features
    dummy_identity  = torch.randn(B, 29)       # Identity cosine similarities (L-1 = 29)
    dummy_rppg      = torch.randn(B, 97)       # rPPG feature vector

    # Instantiate the full fusion model
    model = DeepfakeFusionModel()

    # Run the forward pass
    output = model(dummy_spatial, dummy_freq, dummy_identity, dummy_rppg)

    print(f"  Spatial input:   {dummy_spatial.shape}")
    print(f"  Frequency input: {dummy_freq.shape}")
    print(f"  Identity input:  {dummy_identity.shape}")
    print(f"  rPPG input:      {dummy_rppg.shape}")
    print(f"  Output logit:    {output.shape}")
    print("Multi-Branch Fusion Model verification successful!")
