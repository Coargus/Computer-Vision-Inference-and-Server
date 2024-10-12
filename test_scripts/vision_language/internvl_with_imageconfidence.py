from cvias.vision_language.internvl import InternVL
import numpy as np
from PIL import Image

if __name__ == "__main__":
    # Example usage
    vlm = InternVL(model_name="InternVL2-2B")
    queries = ["Red Bus",
               "Blue Car",
               "Blue Bus",
               "Car", 
               "Person",
               "Bus", 
               "Road and Road Sign",
               "Building and Pavement", 
               "Traffic Sign or Traffic Signal", 
               "Blue Bus or Red Bus",
               "Blue Bus or Green Bus"]
    image_path="/home/hg22723/projects/computer-vision-inference-and-server/sample_data/coco_bus_and_car.png"
 
    image = Image.open(image_path).convert("RGB")

            
    for query in queries:
        detection = vlm.detect(
            frame_img=np.array(image),
            scene_description=query,
            threshold=0.5)
        print("Query: ", query, "Detection: ", detection.is_detected, "Probability: ", detection.probability)
