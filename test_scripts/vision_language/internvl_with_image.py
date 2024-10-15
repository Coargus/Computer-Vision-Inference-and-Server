from cvias.vision_language.internvl import InternVL
import numpy as np
from PIL import Image

if __name__ == "__main__":
    vlm = InternVL(model_name="InternVL2-8B")
    # * * * Example usage - 1 * * * #
    # prompt = "Please describe the image shortly."
    # vlm.infer_with_image(language=prompt,
    #                      image_path="<path_to_image>")
    
    # * * * Example usage - 2 * * * #
    # image_path="<path_to_image>"
    # image = Image.open(image_path).convert("RGB")
    # queries = ["Red Bus",
    #            "Blue Car",
    #            "Car", 
    #            "Person",
    #            "Traffic Sign or Traffic Signal"]
    # for query in queries:
    #     detection = vlm.detect(
    #         frame_img=np.array(image),
    #         scene_description=query,
    #         threshold=0.1)
    #     print("Query: ", query, "Detection: ", detection.is_detected, "Probability: ", detection.probability)