# Phase 5 Completion Report: Face Detection & Alignment

## Objective Achieved
We successfully built the face detection and extraction pipeline using the **RetinaFace** model (via the `retina-face` standalone library), ensuring that only the relevant facial data is forwarded to the CNN models.

## Actions Taken
1. **Dependency Installation**:
   - Installed the `retina-face` standalone package which provides a robust facial detector.
2. **Script Development (`preprocessing/extract_faces.py`)**:
   - Built a comprehensive Python script that iterates through the output of Phase 4 (`frames/`).
   - The script feeds each frame to the RetinaFace detector to identify face bounding boxes.
   - It captures the bounding box coordinates, applies a 10% safety margin to ensure no useful facial data is lost, crops the face out of the original frame, and immediately resizes it to **224x224** (the standard input size for ResNet and most CNNs).
   - The cropped faces are mirrored into a new `faces/` root directory (e.g., `faces/fake/Deepfakes/...`).
   - **As requested, every line of code inside this script is fully commented for maximum clarity.**

## Flow Implemented
The requested data flow is now fully operational:
`video` → `frame extraction` → `RetinaFace detection` → `face crop` → `resize to 224 × 224` → `<ready for CNN>`

## Readiness for Next Phase
With the faces perfectly cleaned, cropped, and resized, we are ready to proceed to **Phase 6: Feature Extraction (Spatial Features)**, where we will push these faces through a pre-trained CNN (like ResNet50 or EfficientNet) to generate numerical embeddings.
