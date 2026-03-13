import torch
import cv2
import numpy as np

def check_environment():
    print("=== Environment Check ===")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"OpenCV Version: {cv2.__version__}")
    print(f"NumPy Version: {np.__version__}")
    
    # Check GPU availability
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA Available: {cuda_available}")
    if cuda_available:
        print(f"CUDNN Version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current Device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("\nWARNING: CUDA is not available. Training will be extremely slow on CPU.")
        print("Please ensure you have an NVIDIA GPU and installed the PyTorch version with CUDA support.")

if __name__ == "__main__":
    check_environment()
