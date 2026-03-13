# Deepfake Detection System

## Overview
This project builds a comprehensive system to detect deepfake videos by leveraging spatial artifacts, temporal inconsistencies, and physiological signals (such as blinking or micro-expressions).

## Target
**Input**: Video
**Output**: Deepfake probability (Fake/Real)

## Directory Structure
- `data_pipeline/`: Scripts for downloading, cleaning, and managing datasets.
- `preprocessing/`: Tools for frame extraction, face detection, and alignment.
- `feature_extractors/`: Pre-trained models and wrappers for extracting spatial (CNN) and identity (FaceNet) embeddings.
- `models/`: Architecture definitions for temporal tracking (Sequence Models/Transformers) and multimodal fusion.
- `training/`: Training loops, loss functions, and optimization scripts.
- `evaluation/`: Scripts for computing metrics (AUC, F1) and Grad-CAM interpretability.
- `inference/`: APIs and standalone prediction scripts for new videos.
- `extension/`: Source code for the browser extension UI and client-side processing.

## Roadmap
This project follows a 19-phase roadmap starting from problem definition and environment setup, moving through feature extraction and model training, and culminating in a deployable browser extension.
