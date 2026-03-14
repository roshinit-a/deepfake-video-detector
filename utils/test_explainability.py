from utils.explainability import DeepfakeExplainer
import os

def run_test():
    # 1. Initialize Explainer
    explainer = DeepfakeExplainer(checkpoint_path="training/best_model.pth")
    
    # 2. Define feature roots
    feature_roots = {
        'spatial': 'embeddings',
        'frequency': 'frequency_features',
        'identity': 'identity_features',
        'rppg': 'rppg_features'
    }
    
    # Ensure reports directory exists for output
    os.makedirs("report/visuals", exist_ok=True)
    
    # 3. Test on a REAL video
    real_video_id = "real/original/000"
    print(f"\nProcessing REAL video: {real_video_id}")
    explainer.visualize_attention(real_video_id, feature_roots, save_path="report/visuals/real_attention.png")
    
    # Grad-CAM on first frame of real video
    real_face_path = "faces/real/original/000/frame_0000.jpg"
    if os.path.exists(real_face_path):
        explainer.generate_gradcam(real_face_path, save_path="report/visuals/real_gradcam.png")
    else:
        print(f"Warning: Real face not found at {real_face_path}")
        
    # 4. Test on a FAKE video
    fake_video_id = "fake/Deepfakes/000_003"
    print(f"\nProcessing FAKE video: {fake_video_id}")
    explainer.visualize_attention(fake_video_id, feature_roots, save_path="report/visuals/fake_attention.png")
    
    # Grad-CAM on first frame of fake video
    fake_face_path = "faces/fake/Deepfakes/000_003/frame_0000.jpg"
    if os.path.exists(fake_face_path):
        explainer.generate_gradcam(fake_face_path, save_path="report/visuals/fake_gradcam.png")
    else:
        print(f"Warning: Fake face not found at {fake_face_path}")

if __name__ == "__main__":
    run_test()
