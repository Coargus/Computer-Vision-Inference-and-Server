from cvias.vision_language.internvl import InternVL
import numpy as np

if __name__ == "__main__":
    # Example usage
    vlm = InternVL()
    prompt = "Please describe the video in detail."
    vlm.infer_with_video(language=prompt,
                         image_path="/opt/sandbox/internvl/test.png")
    