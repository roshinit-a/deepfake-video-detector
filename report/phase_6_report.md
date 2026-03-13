# Phase 6 Completion Report: Feature Extraction (Spatial Features)

## Objective Achieved
We built a GPU-accelerated spatial feature extraction pipeline that converts each cropped 224×224 face image into a **2048-dimensional embedding vector** using a pre-trained ResNet-50 CNN backbone.

## Actions Taken
1. **Pre-trained Model Selection**:
   - Used **ResNet-50** with ImageNet pre-trained weights, which are loaded via PyTorch's `torchvision.models`.
   - Removed the final fully-connected classification layer and replaced it with `nn.Identity()` so the model outputs raw 2048-dim feature vectors (`embeddings`) instead of class probabilities.
2. **Image Preprocessing**:
   - Applied the mandatory ImageNet normalization pipeline: `Resize(224x224)`, `ToTensor()`, `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`.
3. **Batched GPU Inference**:
   - All face frames for a single video are batched into a single tensor and pushed to the GPU in one forward pass.
   - `torch.no_grad()` disables gradient computation, saving memory and maximizing throughput.
4. **Embedding Storage**:
   - Saved per-video embeddings as `.npy` files (e.g., `embeddings/fake/Deepfakes/000.npy`).
   - Each file stores an array of shape `(num_frames, 2048)`, e.g., `(30, 2048)`.
5. **Fully Commented Code**:
   - Every line in `feature_extractors/extract_spatial_features.py` is thoroughly commented for clarity.

## Output Format
```
embeddings/
├── real/
│   └── original/
│       └── 000.npy   ← shape: (30, 2048)
└── fake/
    └── Deepfakes/
        └── 000.npy   ← shape: (30, 2048)
```

## Readiness for Next Phase
With per-video spatial feature tensors saved to disk, we are ready for **Phase 7: Temporal Modeling**, where we will feed these `(30, 2048)` time-series embeddings into a BiLSTM or Transformer to learn motion inconsistencies across frames.
