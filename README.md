# Coargus's Computer Vision Inference and Server (cvias)

The cvias module is a computer vision inference and server package for Coargus CV project.

## Installation

1. Development

```python
pip install -e .["dev","test"]
```

2. Production

```python
pip install -e .
```

##### MMdetection (Third Party)

1. FasterRCNN
Available model list in the class script

```python
from cvias.image.detection.object.open_vocabulary.faster_rcnn import FasterRCNN

model_list = FasterRCNN.available_models()

model = FasterRCNN(
        model_name="<select model from the list>",
        explicit_checkpoint_path=None,
        gpu_number=0,
    )

model.detect(image)
```

##### Yolo (Third Party)

```python
from cvias.image.detection.object.yolo import Yolo

model_list = YoloWorld.available_models()

model = Yolo(
    model_name="<select model from the list>",
    explicit_checkpoint_path=None, 
    gpu_number=0
)

model.detect(image, ["person"])
```

##### YoloWorld (Third Party)

```python
from cvias.image.detection.object.open_vocabulary.yolo_world import YoloWorld

model_list = YoloWorld.available_models()

model = YoloWorld(
        model_name="<select model from the list>",
        explicit_checkpoint_path=None,
        gpu_number=0,
    )

model.detect(image)
```
