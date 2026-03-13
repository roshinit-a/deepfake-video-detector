import os                                       # File and directory operations
import sys                                       # For importing local modules
import torch                                     # Core PyTorch library
import torch.nn as nn                            # Neural network modules
from torch.utils.data import DataLoader, random_split  # Data loading utilities
from torch.optim import AdamW                    # AdamW optimizer (weight decay decoupled)
from torch.optim.lr_scheduler import CosineAnnealingLR  # Cosine learning rate scheduler
from sklearn.metrics import roc_auc_score        # AUC-ROC for evaluation
import numpy as np                               # Numerical operations

# Add the project root to path so we can import local modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.fusion_model import DeepfakeFusionModel   # Our multi-branch fusion model
from training.dataset import DeepfakeDataset          # Custom dataset class (Phase 12)


# ─────────────────────────────── HYPERPARAMETERS ────────────────────────────

BATCH_SIZE      = 16      # Number of videos per gradient update step
NUM_EPOCHS      = 50      # Maximum training epochs before stopping
LEARNING_RATE   = 1e-4    # Initial learning rate for AdamW
WEIGHT_DECAY    = 1e-4    # L2 regularization factor to reduce overfitting
VAL_SPLIT       = 0.2     # Fraction of data to hold out for validation
PATIENCE        = 7       # Early stopping patience (stop if no improvement for N epochs)
CHECKPOINT_PATH = "training/best_model.pth"  # Where to save the best model weights

# ─────────────────────────────── DATASET SETUP ──────────────────────────────

def get_dataloaders(batch_size=BATCH_SIZE, val_split=VAL_SPLIT):
    """
    Builds the full dataset, splits it into train and validation sets,
    and wraps them in DataLoader instances for batched iteration.

    Args:
        batch_size (int): Number of samples per batch.
        val_split (float): Fraction of data to use for validation (e.g., 0.2 = 20%).

    Returns:
        tuple: (train_loader, val_loader) — two PyTorch DataLoader objects.
    """
    # Instantiate the full multi-branch dataset
    full_dataset = DeepfakeDataset(
        embeddings_dir="embeddings",
        freq_dir="frequency_features",
        identity_dir="identity_features",
        rppg_dir="rppg_features",
    )

    # Check that we have at least some data
    if len(full_dataset) == 0:
        raise RuntimeError("No valid samples found! Run all feature extraction scripts first.")

    # Compute the sizes for train and validation splits
    total     = len(full_dataset)
    val_size  = int(total * val_split)        # Number of validation samples
    train_size = total - val_size             # Remaining samples go to training

    # Randomly split the dataset (fixed seed for reproducibility)
    # Setting generator ensures the same split every run
    generator = torch.Generator().manual_seed(42)  # Seed for reproducible splits
    train_set, val_set = random_split(full_dataset, [train_size, val_size], generator=generator)

    print(f"Train samples: {len(train_set)} | Validation samples: {len(val_set)}")

    # Create DataLoader for the training set
    # shuffle=True randomizes order each epoch to prevent the model memorizing order
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,          # Randomize every epoch for better generalization
        num_workers=4,         # Parallel workers to speed up data loading
        pin_memory=True,       # Pins memory for faster CPU→GPU transfer
    )

    # Create DataLoader for the validation set
    # shuffle=False keeps the same order for consistent evaluation metrics
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,         # Consistent order for evaluation
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader


# ──────────────────────────────── TRAINING EPOCH ────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    """
    Runs one full epoch of model training over the training DataLoader.

    Uses AMP (Automatic Mixed Precision) via GradScaler for GPU memory efficiency
    and ~2x speedup on modern NVIDIA GPUs with Tensor Cores.

    Args:
        model:     The DeepfakeFusionModel to train.
        loader:    DataLoader for the training split.
        optimizer: AdamW optimizer.
        criterion: Loss function (BCEWithLogitsLoss).
        device:    Compute device (GPU/CPU).
        scaler:    GradScaler for AMP (mixed precision) training.

    Returns:
        float: Average training loss over all batches in this epoch.
    """
    model.train()      # Enable training mode (activates dropout, batch norm training behavior)
    total_loss = 0.0   # Accumulate total loss for this epoch

    for batch in loader:
        # Move each feature tensor to the compute device (GPU)
        spatial   = batch["spatial"].to(device)    # (B, seq_len, 2048)
        frequency = batch["frequency"].to(device)  # (B, seq_len, 192)
        identity  = batch["identity"].to(device)   # (B, 29)
        rppg      = batch["rppg"].to(device)       # (B, 97)
        labels    = batch["label"].to(device)      # (B,)

        # Zero out gradients from the previous batch — must be done every step
        optimizer.zero_grad()

        # ── AMP Forward Pass ─────────────────────────────────────────────────
        # autocast() automatically casts operations to float16 where safe
        # This halves memory usage and doubles throughput on supported GPUs
        with torch.amp.autocast(device_type="cuda"):
            logits = model(spatial, frequency, identity, rppg)  # (B, 1)
            loss   = criterion(logits.squeeze(1), labels)        # Scalar loss

        # ── Backward Pass (AMP-aware) ─────────────────────────────────────────
        # scaler.scale() scales the loss to prevent underflow in float16 gradients
        scaler.scale(loss).backward()

        # Unscale gradients and clip them to prevent exploding gradient problem
        # max_norm=1.0 clips the global gradient norm to 1.0
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Perform the optimizer step (unscales internally and checks for inf/nan)
        scaler.step(optimizer)

        # Update the scaler for the next iteration
        scaler.update()

        # Accumulate the loss value for reporting
        total_loss += loss.item()

    # Return the average loss over all batches
    return total_loss / len(loader)


# ────────────────────────────── VALIDATION EPOCH ────────────────────────────

def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set without updating weights.

    Computes:
    - Average validation loss
    - AUC-ROC score (area under the ROC curve, the primary metric for deepfake detection)

    Args:
        model:     The DeepfakeFusionModel to evaluate.
        loader:    DataLoader for the validation split.
        criterion: Loss function (BCEWithLogitsLoss).
        device:    Compute device.

    Returns:
        tuple: (avg_val_loss, auc_score)
    """
    model.eval()       # Disable dropout, use running stats for batch norm
    total_loss  = 0.0
    all_labels  = []   # Accumulated ground-truth labels
    all_probs   = []   # Accumulated predicted probabilities

    # Disable gradient computation — we are only doing inference
    with torch.no_grad():
        for batch in loader:
            # Move tensors to device
            spatial   = batch["spatial"].to(device)
            frequency = batch["frequency"].to(device)
            identity  = batch["identity"].to(device)
            rppg      = batch["rppg"].to(device)
            labels    = batch["label"].to(device)

            # Forward pass — no AMP needed for eval (no memory pressure)
            logits = model(spatial, frequency, identity, rppg)   # (B, 1)
            loss   = criterion(logits.squeeze(1), labels)         # Scalar loss

            total_loss += loss.item()

            # Convert logits → probabilities using Sigmoid activation
            probs = torch.sigmoid(logits.squeeze(1))  # (B,) probabilities in [0, 1]

            # Collect predictions and ground truth for AUC computation
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Compute AUC-ROC over the entire validation set
    # AUC = 0.5 means random guess; AUC = 1.0 means perfect prediction
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        # roc_auc_score fails if only one class is present in validation
        auc = 0.5  # Default to chance-level AUC

    return total_loss / len(loader), auc


# ──────────────────────────────── MAIN TRAINING LOOP ────────────────────────

def train():
    """
    Main training entry point with:
    - Multi-branch fusion model
    - BCEWithLogitsLoss (numerically stable binary cross entropy)
    - AdamW optimizer with weight decay
    - CosineAnnealingLR scheduler for smooth learning rate decay
    - AMP (Automatic Mixed Precision) for GPU memory efficiency
    - Early stopping to prevent overfitting
    - Checkpoint saving: saves the model with the best validation AUC
    """
    # ── Device Setup ─────────────────────────────────────────────────────────
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, val_loader = get_dataloaders()

    # ── Model ─────────────────────────────────────────────────────────────────
    model = DeepfakeFusionModel().to(device)   # Build and move model to GPU
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Loss Function ─────────────────────────────────────────────────────────
    # BCEWithLogitsLoss combines Sigmoid + Binary Cross Entropy in one numerically stable op
    # pos_weight handles class imbalance: if dataset has more real than fake (or vice versa),
    # this weight balances the gradient signal from each class
    num_fake = sum(1 for s in train_loader.dataset.dataset.samples if s[1] == 1)
    num_real = len(train_loader.dataset.dataset.samples) - num_fake
    pos_weight = torch.tensor([num_real / max(num_fake, 1)], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # AdamW decouples weight decay from gradient updates (better regularization than Adam)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # ── Learning Rate Scheduler ───────────────────────────────────────────────
    # CosineAnnealingLR decays the learning rate from LR → 0 following a cosine curve
    # This gradually reduces step size as training converges, improving final accuracy
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # ── AMP Scaler ────────────────────────────────────────────────────────────
    # GradScaler scales gradients to prevent underflow in float16 during AMP training
    scaler = torch.amp.GradScaler()

    # ── Training State ────────────────────────────────────────────────────────
    best_auc       = 0.0    # Track the best AUC score seen so far
    patience_count = 0      # Counter for early stopping patience

    print(f"\nStarting training for up to {NUM_EPOCHS} epochs (patience={PATIENCE})...\n")
    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>8}  {'Val AUC':>8}")
    print("-" * 45)

    for epoch in range(1, NUM_EPOCHS + 1):
        # ── Train for one epoch ───────────────────────────────────────────────
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)

        # ── Validate on the held-out validation set ───────────────────────────
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # ── Step the LR scheduler ─────────────────────────────────────────────
        scheduler.step()

        # ── Print progress ────────────────────────────────────────────────────
        print(f"{epoch:>6}  {train_loss:>10.4f}  {val_loss:>8.4f}  {val_auc:>8.4f}")

        # ── Save checkpoint if this is the best model so far ──────────────────
        if val_auc > best_auc:
            best_auc = val_auc         # Update the best known AUC
            patience_count = 0          # Reset patience counter

            # Save all model weights to disk
            os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
            torch.save({
                "epoch":       epoch,             # Epoch number for reference
                "model_state": model.state_dict(),# All trainable parameters
                "optimizer":   optimizer.state_dict(),
                "best_auc":    best_auc,
            }, CHECKPOINT_PATH)
            print(f"         ✓ New best AUC: {best_auc:.4f} — model saved.")
        else:
            patience_count += 1         # No improvement: increment patience counter
            if patience_count >= PATIENCE:
                print(f"\nEarly stopping triggered after {epoch} epochs (no AUC improvement in {PATIENCE} epochs).")
                break

    print(f"\nTraining complete. Best Validation AUC: {best_auc:.4f}")
    print(f"Best model checkpoint saved at: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    train()
