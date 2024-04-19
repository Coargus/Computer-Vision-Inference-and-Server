# Python-Project-Template

This is a template repository. Please initialize your python project using this template.

## Image

### Detection

#### Object

##### Yolo (Third Party)

Available model list: YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l, YOLOv8x, YOLOv9c, YOLOv9e

```python
yolo = Yolo(
    model_name="YOLOv9e", 
    explicit_checkpoint_path=None, 
    gpu_number=0
)

yolo.detect(image, ["person"])

```
