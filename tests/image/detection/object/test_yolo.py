from pathlib import Path

import pytest
from cogcvutil import read_image

from cvias.image.detection.object.yolo import Yolo

ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent
SAMPLE_DATA_DIR = ROOT_DIR / "sample_data"


def test_yolo_detection() -> None:
    """Test Yolo detection on a sample image."""
    image_path = SAMPLE_DATA_DIR / "titanic.png"
    assert image_path.exists(), f"Sample image {image_path} does not exist."

    image_array = read_image(image_path, "numpy")
    assert image_array is not None, "Failed to read image as numpy array."

    model = Yolo(
        model_name="YOLOv9e", explicit_checkpoint_path=None, gpu_number=0
    )
    detected_objects = model.detect(image_array, ["person"])
    assert detected_objects.name == "person"
    assert detected_objects.confidence > 0.92


if __name__ == "__main__":
    pytest.main()
