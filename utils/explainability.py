import os                       # For path manipulation
import torch                    # PyTorch for model inference
import numpy as np               # NumPy for data handling
import matplotlib.pyplot as plt  # Standard plotting library
import seaborn as sns            # Statistical data visualization
import cv2                       # OpenCV for image handling
from PIL import Image            # PIL for image processing
from models.fusion_model import get_fusion_model  # Our fusion model
from torchvision import models, transforms        # ResNet-50 and image transforms
from pytorch_grad_cam import GradCAM             # Grad-CAM implementation
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ──────────────────────────────────────────────────────────────────────────────
# Explainability Manager for Deepfake Detection
# ──────────────────────────────────────────────────────────────────────────────

class DeepfakeExplainer:
    """
    Provides tools to interpret and visualize the DeepfakeFusionModel's decisions.
    Supports:
    1. Attention Mapping: Which feature branch (Spatial, Temporal, etc.) was most important?
    2. Grad-CAM: Which parts of the face triggered the "Fake" classification?
    3. Probability Over Time: How did the model's confidence change across frames?
    """

    def __init__(self, checkpoint_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # 1. Load the Fusion Model
        self.model = get_fusion_model(device=self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Check if checkpoint is the full dict (from our train.py) or just the state_dict
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.eval()
        
        # 2. Setup ResNet-50 Backbone for Grad-CAM
        # Since our fusion model takes 2048-dim embeddings, we need to run the original 
        # ResNet-50 to get the gradients on the feature maps.
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.resnet50 = self.resnet50.to(self.device)
        self.resnet50.eval()
        
        # Remove the final classification layers of ResNet to get the 2048-dim embedding
        self.resnet_features = torch.nn.Sequential(*(list(self.resnet50.children())[:-1]))
        
        # Setup ResNet-50 Image Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _pad_or_trim(self, arr, target_len, feature_dim):
        """Pads or trims a 2D array to (target_len, feature_dim)."""
        T = arr.shape[0]
        if T >= target_len:
            return arr[:target_len, :]
        else:
            pad = np.zeros((target_len - T, feature_dim), dtype=np.float32)
            return np.concatenate([arr, pad], axis=0)

    def _fix_len(self, arr, target_len):
        """Pads or trims a 1D array to exactly target_len elements."""
        if len(arr) >= target_len:
            return arr[:target_len]
        else:
            pad = np.zeros(target_len - len(arr), dtype=np.float32)
            return np.concatenate([arr, pad])

    def get_features_for_video(self, video_id, feature_roots):
        """
        Loads pre-extracted features for a specific video ID.
        video_id example: 'real/original/000' or 'fake/Deepfakes/000_003'
        """
        features = {}
        
        # Spatial: embeddings/{video_id}.npy
        spatial_path = os.path.join(feature_roots['spatial'], video_id + ".npy")
        spatial = np.load(spatial_path).astype(np.float32)
        spatial = self._pad_or_trim(spatial, 30, 2048)
        features['spatial'] = torch.from_numpy(spatial).unsqueeze(0).to(self.device)
        
        # Frequency: frequency_features/{video_id}.npy
        freq_path = os.path.join(feature_roots['frequency'], video_id + ".npy")
        freq = np.load(freq_path).astype(np.float32)
        freq = self._pad_or_trim(freq, 30, 192)
        features['frequency'] = torch.from_numpy(freq).unsqueeze(0).to(self.device)
        
        # Identity: identity_features/{video_id}/similarities.npy
        identity_path = os.path.join(feature_roots['identity'], video_id, "similarities.npy")
        identity = np.load(identity_path).astype(np.float32)
        identity = self._fix_len(identity, 29)
        features['identity'] = torch.from_numpy(identity).unsqueeze(0).to(self.device)
        
        # rPPG: rppg_features/{video_id}.npy
        rppg_path = os.path.join(feature_roots['rppg'], video_id + ".npy")
        rppg = np.load(rppg_path).astype(np.float32)
        rppg = self._fix_len(rppg, 97)
        features['rppg'] = torch.from_numpy(rppg).unsqueeze(0).to(self.device)
        
        return features

    def visualize_attention(self, video_id, feature_roots, save_path=None):
        """
        Generates a bar chart showing the attention weights for each branch.
        """
        features = self.get_features_for_video(video_id, feature_roots)
        
        with torch.no_grad():
            logit, attn_weights = self.model(
                features['spatial'], 
                features['frequency'], 
                features['identity'], 
                features['rppg'], 
                return_attention=True
            )
            prob = torch.sigmoid(logit).item()
        
        weights = attn_weights.cpu().numpy()[0]
        labels = ['Spatial', 'Temporal', 'Frequency', 'Identity+rPPG']
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=labels, y=weights, palette='viridis')
        plt.title(f"Branch Attention Weights - Video {video_id}\n(Fake Probability: {prob:.4f})")
        plt.ylabel("Weight (sums to 1.0)")
        plt.ylim(0, 1)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Attention plot saved to {save_path}")
        else:
            plt.show()
        plt.close()

    def generate_gradcam(self, face_image_path, save_path=None):
        """
        Generates a Grad-CAM heatmap for a single face image using ResNet-50.
        Note: This explains what the backbone sees as suspicious in terms of spatial features.
        """
        # Load and transform image
        rgb_img = cv2.imread(face_image_path)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(Image.fromarray(rgb_img)).unsqueeze(0).to(self.device)
        
        # Targeting the last convolutional layer of ResNet-50
        target_layers = [self.resnet50.layer4[-1]]
        
        # Initialize Grad-CAM
        cam = GradCAM(model=self.resnet50, target_layers=target_layers)
        
        # We target the classifier to see what regions push towards specific features
        # Since this is a generic ImageNet model, we can just look for the most active class
        # or a specific index if we had a fine-tuned ResNet.
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)
        grayscale_cam = grayscale_cam[0, :]
        
        # Overlay heatmap on original image
        visualization = show_cam_on_image(rgb_img.astype(np.float32) / 255, grayscale_cam, use_rgb=True)
        
        # Save or show
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
            print(f"Grad-CAM saved to {save_path}")
        else:
            plt.imshow(visualization)
            plt.show()

    def plot_confidence_over_time(self, video_id, faces_root, feature_roots, save_path=None):
        """
        Runs the model on sliding windows or sequentially to plot confidence over time.
        For simplicity here, we visualize the frames and their corresponding probability.
        """
        # Load full features for the final verdict
        features = self.get_features_for_video(video_id, feature_roots)
        
        with torch.no_grad():
            logit = self.model(features['spatial'], features['frequency'], features['identity'], features['rppg'])
            final_prob = torch.sigmoid(logit).item()
            
        print(f"Video {video_id} Final Probability: {final_prob:.4f}")
        
        # In a real scenario, we could re-run features for sub-segments
        # but here we'll just plot a mock confidence for visualization purposes
        # or if we had a per-frame model.

if __name__ == "__main__":
    # Example usage (Mock)
    explainer = DeepfakeExplainer(checkpoint_path="training/best_model.pth")
    
    feature_roots = {
        'spatial': 'embeddings',
        'frequency': 'frequency_features',
        'identity': 'identity_features',
        'rppg': 'rppg_features'
    }
    
    # Select a sample fake video from deepfakes
    video_id = "fake/Deepfakes/000_003" # Example naming convention
    
    # explainer.visualize_attention(video_id, feature_roots, save_path="report/attention_weights.png")
    # explainer.generate_gradcam("faces/fake/Deepfakes/000_003/frame_0000.jpg", save_path="report/gradcam_frame0.png")
