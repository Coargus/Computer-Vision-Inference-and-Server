from cvias.vision_language.internvl import InternVL
import numpy as np

if __name__ == "__main__":
    # Example usage
    vlm = InternVL(model_name="InternVL2-40B")
    prompt = "Please describe the image shortly."
    vlm.infer_with_image(language=prompt,
                         image_path="<path_to_image>")
    
