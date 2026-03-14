# 🕵️ Deepfake Video Detector

> **AI-powered multi-modal deepfake detection** — An advanced ensemble system combining spatial CNNs, temporal modeling, frequency analysis, and physiological signal detection to expose AI-generated manipulations.

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![NVIDIA](https://img.shields.io/badge/GPU-CUDA--Accelerated-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-F1D900?logo=github&logoColor=black)](LICENSE)

</div>

---

## 🌟 Overview

This project implements a state-of-the-art **Multi-Branch Fusion Model** designed to detect deepfakes by looking beyond simple pixel artifacts. It analyzes:
- **Spatial Features**: Frame-by-frame visual consistency.
- **Temporal Dynamics**: Flicker and motion irregularities across time.
- **Frequency Domain**: Micro-structural GAN artifacts via Discrete Cosine Transform (DCT).
- **Identity & Physiology**: Identity consistency across frames and heart rate (rPPG) anomalies.

---

## 🚀 Key Features

- **Multi-Modal Fusion**: Combines 4 distinct feature branches using a learned **Channel Attention** mechanism.
- **High Performance**: Achieved a baseline **AUC of 0.69** on the FF++ dataset (work-in-progress).
- **Explainable AI (XAI)**:
  - **Grad-CAM**: Visualize which parts of a face triggered the "Fake" classification.
  - **Attention Mapping**: Understand which feature branch (e.g., Spatial vs. Frequency) contributed most to a specific decision.
- **GPU Optimized**: Utilizes `Decord` and `MTCNN` for ultra-fast frame extraction and face detection.

---

## 🔬 Detection Architecture

```mermaid
graph TD
    A[🎬 Input Video] --> B[🖼️ Frame Extraction]
    B --> C[👤 Face Detection & Alignment]
    C --> D{Multi-Branch Extraction}
    
    D --> D1[🧠 Spatial Branch - ResNet-50]
    D --> D2[📈 Temporal Branch - BiLSTM]
    D --> D3[🌊 Frequency Branch - DCT]
    D --> D4[💓 Identity & rPPG Branch]
    
    D1 & D2 & D3 & D4 --> E[🔀 Channel Attention Fusion]
    E --> F[📊 Deepfake Probability]
    F --> G[🔍 XAI Visualization]
```

---

## 🛠️ Project Structure

```bash
deepFaceDetection/
├── 📁 models/                 # Multi-branch fusion model & backbones
├── 📁 preprocessing/          # Face extraction & alignment pipeline
├── 📁 feature_extractors/     # DCT, Identity, and Spatial processors
├── 📁 training/               # Optimized training loops (AMP, AdamW)
├── 📁 utils/                  # Explainability (Grad-CAM, Attention)
├── 📁 report/                 # Detailed phase-by-phase documentation
└── 📄 README.md               # You are here!
```

---

## 📈 Roadmap

- [x] **Phase 1-5**: Data pipeline, Frame & Face extraction (RetinaFace).
- [x] **Phase 6-7**: Spatial (ResNet-50) & Temporal (BiLSTM) modeling.
- [x] **Phase 8-10**: Frequency (DCT), Identity (FaceNet), and rPPG (Physiological).
- [x] **Phase 11-12**: Attention Fusion Model & Multi-modal training.
- [x] **Phase 13**: **Explainability Integration (Grad-CAM & XAI).**
- [ ] **Phase 14**: Model Fine-tuning & Optimization.
- [ ] **Phase 15-17**: Backend API, Dashboard & Browser Extension.
- [ ] **Phase 18-19**: Evaluation & Final Presentation.

---

## 🧪 Explainability Example

Want to know *why* the model made a prediction? 

```python
from utils.explainability import DeepfakeExplainer

# Load explainer with trained model
explainer = DeepfakeExplainer("training/best_model.pth")

# Generate Attention weights and Grad-CAM
explainer.visualize_attention("fake/Deepfakes/000_003", feature_roots)
explainer.generate_gradcam("faces/fake/Deepfakes/000_003/frame_0000.jpg")
```

---

## 🤝 Contributing

We are actively improving the model! Feel free to open issues or PRs.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Developed with ❤️ by [roshinit-a](https://github.com/roshinit-a)
