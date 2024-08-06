"""Ultralytics's Yolo Model."""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

from cog_cv_abstraction.schema.detected_object import DetectedObject
from cog_cv_abstraction.schema.detected_object_set import DetectedObjectSet
from cogutil import download
from cogutil.parser import parse_f_str
from cogutil.torch import get_device
from ultralytics import YOLOWorld

from cvias.image.detection import CviasDetectionModel

warnings.filterwarnings("ignore")
if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from ultralytics.engine.results import Results

MODEL_PATH = {
    "YOLOv8x-worldv2": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-worldv2.pt",
}


class YoloWorld(CviasDetectionModel):
    """Yolo."""

    def __init__(
        self,
        model_name: str = "YOLOv8x-worldv2",
        explicit_checkpoint_path: Path | None = None,
        gpu_number: int = 0,
    ) -> None:
        """Initialization."""
        super().__init__()
        if explicit_checkpoint_path:
            self.checkpoint = explicit_checkpoint_path
            model_name = explicit_checkpoint_path.split("/")[-1]
        else:
            if model_name not in MODEL_PATH:
                msg = parse_f_str(f"""
                    Model name {model_name} is not supported.
                    Supported models are {MODEL_PATH.keys()}""")
                raise ValueError(msg)
            self.checkpoint = download.coargus_downloader(
                url=MODEL_PATH[model_name], model_dir=model_name
            )
        self.model_name = model_name
        self.model = self.load_model(self.checkpoint)
        self.device = get_device(gpu_number)
        self.model.to(self.device)
        self.english_to_class_id = {v: k for k, v in self.model.names.items()}
        self.class_id_to_english = {k: v for k, v in self.model.names.items()}  # noqa: C416

    def load_model(self, weight_path: str) -> YOLOWorld:
        """Load weight.

        Args:
            weight_path (str): Path to weight file.

        Returns:
            None
        """
        return YOLOWorld(weight_path)

    def validate_classes(self, classes: list) -> bool:
        """Validate classes whether they are detectable from the model..

        Args:
            classes (list): List of classes.

        Returns:
            list: List of classes.
        """
        return len(classes) > 0

    def get_bounding_boxes(self, detected_objects: Results) -> list:
        """Get bounding boxes.

        Args:
            detected_objects (DetectedObject): Detected object.

        Returns:
            list: Bounding boxes.
        """
        bboxes = []
        if detected_objects:
            for row in detected_objects.boxes.data.cpu().numpy():
                bbox = row[:4].tolist()
                bboxes.append(bbox)
        return bboxes

    def detect(
        self, frame_img: np.ndarray, classes: list | None = None
    ) -> DetectedObjectSet:
        """Detect object in frame.

        Args:
            frame_img (np.ndarray): Frame image.
            classes (list[str]): List of class names.

        Returns:
            any: Detections.
        """
        if classes:
            msg = (
                "Classes are not supported for open vocabulary detection models"
            )
            logging.warning(msg)

        detected_objects = self.model.predict(source=frame_img)[0]
        num_detections = len(detected_objects.boxes)
        is_detected = bool(num_detections > 0)
        confidence_from_model = list(
            detected_objects.boxes.conf.cpu().detach().numpy()
        )

        detected_obj_set = DetectedObjectSet()

        if is_detected:
            for idx, detected_class_id in enumerate(
                detected_objects.boxes.cls.cpu().detach().numpy()
            ):
                class_name = self.class_id_to_english.get(detected_class_id)
                if class_name not in detected_obj_set:
                    detected_obj_set[class_name] = DetectedObject(
                        name=class_name,
                        model_name=self.model_name,
                        is_detected=bool(num_detections > 0),
                        bounding_box_of_all_obj=[],
                    )
                detected_obj_set[class_name].confidence_of_all_obj.append(
                    confidence_from_model[idx]
                )
                detected_obj_set[class_name].bounding_box_of_all_obj.append(
                    detected_objects.boxes.data.cpu().numpy()[idx][:4].tolist()
                )
                detected_obj_set[class_name].number_of_detection += 1
                detected_obj_set[class_name].confidence = max(
                    detected_obj_set[class_name].confidence_of_all_obj
                )
        if self.calibration_method:
            # calibrate confidence score
            for detected_object in detected_obj_set.values():
                self.calibrate(detected_object)

        return detected_obj_set

    def calibrate(self, detected_object: DetectedObject) -> DetectedObject:
        """Calibrate detection results.

        Args:
            detected_object (DetectedObject): Detected object.

        Returns:
            DetectedObject: Calibrated detected object.
        """
        probabilities = []
        if detected_object.is_detected:
            for confidence in detected_object.confidence_of_all_obj:
                probabilities.append(self.calibrate_confidence(confidence))
            detected_object.probability_of_all_obj = probabilities
            detected_object.probability = max(probabilities)
            detected_object.confidence = max(
                detected_object.confidence_of_all_obj
            )
        return detected_object

    def get_class_id_from_name(self, class_name: str) -> int:
        """Get class id.

        Args:
            class_name (str): Class name.

        Returns:
            int: Class id.
        """
        return self.english_to_class_id.get(class_name)
