# 🕵️ Deepfake Video Detector

> **AI-powered multi-modal deepfake detection** — combining spatial CNNs, temporal modeling, frequency analysis, and physiological signal detection to expose manipulated videos with high accuracy.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GPU](https://img.shields.io/badge/GPU-CUDA%20Accelerated-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)

---

## 🎯 Objective

| Input | Output |
|-------|--------|
| 🎬 A video file | 📊 Deepfake probability score (0–1) |

The system extracts faces from video frames, analyzes them for spatial, temporal, and frequency anomalies, and produces a final classification.

---

## 🗂️ Project Structure

```
deepFaceDetection/
│
├── 📁 data_pipeline/          # Dataset directory setup scripts
├── 📁 preprocessing/          # Frame extraction & face detection (RetinaFace)
├── 📁 feature_extractors/     # CNN-based spatial feature extraction (ResNet-50)
├── 📁 models/                 # Temporal (BiLSTM), Fusion, and classifier models
├── 📁 training/               # Train loop, loss functions, optimizer configs
├── 📁 evaluation/             # Metrics, ROC curves, confusion matrices
├── 📁 inference/              # Inference pipeline for new videos
├── 📁 extension/              # Browser extension for real-time deepfake detection
│
├── check_env.py               # 🔍 Checks GPU/CUDA availability
├── requirements.txt           # 📦 All Python dependencies
└── README.md
```

---

## 🔬 Detection Pipeline

```
🎬 Input Video
      │
      ▼
🖼️  Frame Extraction          ← Decord (GPU-accelerated, ~3-4x faster than OpenCV)
      │
      ▼
👤  Face Detection             ← RetinaFace (landmark-aware)
      │
      ▼
✂️   Face Crop + Resize        ← 224 × 224 px (with 10% margin)
      │
      ├─────────────────┬──────────────────────┐
      ▼                 ▼                      ▼
🧠 Spatial CNN      📈 Temporal BiLSTM    🌊 Frequency DCT
   (ResNet-50)       (2-layer BiLSTM)     (DCT anomaly map)
   2048-dim emb.     Detects flickers     Detects GAN artifacts
      │                 │                      │
      └─────────────────┴──────────────────────┘
                        │
                        ▼
               🔀 Multi-Branch Fusion
                        │
                        ▼
              📊 Deepfake Probability
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/roshinit-a/deepfake-video-detector.git
cd deepfake-video-detector
```

### 2. Set up environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Check your GPU
```bash
python3 check_env.py
```

### 4. Prepare dataset
Place your videos inside:
```
dataset/
├── real/
└── fake/
```

### 5. Run the pipeline
```bash
# Step 1: Extract frames (GPU-accelerated via Decord)
python3 preprocessing/extract_frames.py

# Step 2: Detect and crop faces (RetinaFace)
python3 preprocessing/extract_faces.py

# Step 3: Extract spatial features (ResNet-50)
python3 feature_extractors/extract_spatial_features.py
```

---

## 📦 Key Dependencies

| Library | Purpose |
|--------|---------|
| `torch` / `torchvision` | Deep learning backbone |
| `retina-face` | State-of-the-art face detection |
| `decord` | Fast GPU-accelerated video decoding |
| `opencv-python` | Image processing & visualization |
| `numpy` | Numerical operations |
| `tqdm` | Progress bars |
| `fastapi` | Backend API server |

---

## 🗺️ Roadmap

- [x] Phase 1: Project Structure
- [x] Phase 2: Environment Setup
- [x] Phase 3: Dataset Acquisition
- [x] Phase 4: Frame Extraction
- [x] Phase 5: Face Detection (RetinaFace)
- [x] Phase 6: Spatial Feature Extraction (ResNet-50)
- [x] Phase 7: Temporal Modeling (BiLSTM)
- [ ] Phase 8: Frequency-Domain Detection (DCT)
- [ ] Phase 9: Identity Consistency Detection
- [ ] Phase 10: Physiological Signal Detection
- [ ] Phase 11: Multi-Branch Fusion Model
- [ ] Phase 12: Training Strategy
- [ ] Phase 13: Explainability
- [ ] Phase 14: Model Optimization
- [ ] Phase 15: Backend API
- [ ] Phase 16: Browser Extension
- [ ] Phase 17: Visualization Dashboard
- [ ] Phase 18: Evaluation & Research Report
- [ ] Phase 19: Final Presentation

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

---

## 📄 License

[MIT](LICENSE) © 2025 roshinit-a
