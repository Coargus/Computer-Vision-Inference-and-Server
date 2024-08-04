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
