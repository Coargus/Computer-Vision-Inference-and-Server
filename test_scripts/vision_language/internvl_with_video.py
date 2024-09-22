from cvias.vision_language.internvl import InternVL
import numpy as np

if __name__ == "__main__":
    # Example usage
    vlm = InternVL(model_name="InternVL2-40B")
    prompt = "Please describe the video in detail."
    vlm.infer_with_video(language=prompt,
                         video_path="<path_to_video>")
    