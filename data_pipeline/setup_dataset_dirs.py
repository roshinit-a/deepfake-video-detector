import os

def create_dataset_structure():
    """Creates the necessary directory structure for the datasets."""
    base_dir = "dataset"
    sub_dirs = ["real", "fake"]

    for sub_dir in sub_dirs:
        path = os.path.join(base_dir, sub_dir)
        os.makedirs(path, exist_ok=True)
        print(f"Created directory: {path}")

    print("\nDataset structure generated successfully.")
    print("Please download FaceForensics++ videos and place them in the respective 'real' or 'fake' directories.")

if __name__ == "__main__":
    create_dataset_structure()
