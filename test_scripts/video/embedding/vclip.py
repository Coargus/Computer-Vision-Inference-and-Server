from cvias.video.embedding import ViClip
import numpy as np

if __name__ == "__main__":
    # Example usage
    viclip = ViClip(
        "/opt/mars/mnt/model_weights/viclip/ViClip-InternVid-10M-FLT.pth"
    )
    seed = 42  # Set your desired seed
    rng = np.random.default_rng(seed)  # Initialize RNG with a seed
    
    # 1. Pass 8 frames as a numpy 
    frames = rng.random((8, 224, 224, 3))
    feature = viclip.get_feature(frames)
    a = rng.random((224, 224, 3))
    b = rng.random((224, 224, 3))
    # 2. Pass 8 frames as a sequence of frames 
    feature = viclip.get_feature([a, b, a, b, a, b, a, a])

