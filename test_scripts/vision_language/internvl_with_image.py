from cvias.vision_language.internvl import InternVL
import numpy as np

if __name__ == "__main__":
    # Example usage
    vlm = InternVL()
    prompt = "<image>\nPlease describe the image shortly."
    vlm.infer_with_image(language=prompt,
                         image_path="/opt/sandbox/internvl/test.png")