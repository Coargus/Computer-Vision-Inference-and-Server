from pathlib import Path

from cogcvutil import read_image

from cvias.image.detection.object.yolo import Yolo

ROOT_DIR = Path(__file__).parent.parent.parent.parent
SAMPLE_DATA_DIR = Path(__file__).parent.parent.parent.parent / "sample_data"
if __name__ == "__main__":
    image_path = SAMPLE_DATA_DIR / "titanic.png"
    image_array = read_image(image_path, "numpy")
    model = Yolo(
        model_name="YOLOv9e", explicit_checkpoint_path=None, gpu_number=0
    )
    detected_object = model.detect(image_array, ["person"])
