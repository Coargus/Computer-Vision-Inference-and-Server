from cvias.vision_language.internvl import InternVL
import numpy as np

if __name__ == "__main__":
    # Example usage
    vlm = InternVL(model_name="InternVL2-8B")
    prompt = "Please describe the image shortly."
    des = vlm.infer_with_image(language=prompt,
                         image_path="/home/hg22723/projects/computer-vision-inference-and-server/sample_data/coco_bus_and_car.png")
    
    print(des)