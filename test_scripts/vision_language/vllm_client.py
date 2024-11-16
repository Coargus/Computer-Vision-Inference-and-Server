from cvias.vision_language.vllm_client import VisionLanguageModelVLLM
from PIL import Image
import numpy as np

if __name__ == "__main__":
    vision_language_model = VisionLanguageModelVLLM(
        endpoint="http://<your_endpoint>/v1",
        api_key="empty",
        model="/nas/mars/model_weights/OpenGVLab_InternVL2-8B",
        parallel_inference=False,
    )

    prompt = "Describe this image"

    # * * * Example usage - 1 (Detection) * * * #
    image_path = "<path_to_image>"
    image = Image.open(image_path).convert("RGB")
    obj = vision_language_model.detect(
        frame_img=np.array(image),
        scene_description=prompt,
        confidence_as_token_probability=False,
    )
    print(obj)
