"""Ultralytics's Yolo Model."""

from __future__ import annotations

import logging
import warnings
from pprint import pprint
from typing import TYPE_CHECKING

import numpy as np
from cog_cv_abstraction.schema.detected_object import DetectedObject
from cogutil import download
from cogutil.torch import get_device
from ultralytics import YOLO

from cvias.image.detection import CviasDetectionModel

warnings.filterwarnings("ignore")
if TYPE_CHECKING:
    from pathlib import Path

    from ultralytics.engine.results import Results

# Get the home directory of the current user
MODEL_PATH = {
    "YOLOv8n": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
    "YOLOv8s": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt",
    "YOLOv8m": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt",
    "YOLOv8l": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l.pt",
    "YOLOv8x": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt",
    "YOLOv9c": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9c.pt",
    "YOLOv9e": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9e.pt",
}


class Yolo(CviasDetectionModel):
    """Yolo."""

    def __init__(
        self,
        model_name: str = "YOLOv9e",
        explicit_checkpoint_path: Path | None = None,
        gpu_number: int = 0,
        calibration_method: str | None = None,
    ) -> None:
        """Initialization."""
        super().__init__(calibration_method=calibration_method)
        if explicit_checkpoint_path:
            self.checkpoint = explicit_checkpoint_path
            model_name = explicit_checkpoint_path.split("/")[-1]
        else:
            self.checkpoint = download.coargus_downloader(
                url=MODEL_PATH[model_name], model_dir=model_name
            )
        self.model_name = model_name
        self.model = self.load_model(self.checkpoint)
        self.device = get_device(gpu_number)
        self.model.to(self.device)
        self.english_to_class_id = {v: k for k, v in self.model.names.items()}
        self.class_id_to_english = {k: v for k, v in self.model.names.items()}  # noqa: C416

    def load_model(self, weight_path: str) -> YOLO:
        """Load weight.

        Args:
            weight_path (str): Path to weight file.

        Returns:
            None
        """
        return YOLO(weight_path)

    def validate_classes(self, classes: list) -> list:
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

    def detect(self, frame_img: np.ndarray, classes: list) -> any:
        """Detect object in frame.

        Args:
            frame_img (np.ndarray): Frame image.
            classes (list[str]): List of class names.

        Returns:
            any: Detections.
        """
        if len(classes) == 1:
            class_name = classes[0]
        class_ids = [self.english_to_class_id.get(c) for c in classes]

        if self.validate_classes(class_ids):
            # object is detectable from the model
            detected_objects = self.model.predict(
                source=frame_img, classes=class_ids
            )[0]

            num_detections = len(detected_objects.boxes)

            if num_detections == 0:
                # No object detected
                confidence_from_model = None

            else:
                confidence_from_model = list(
                    detected_objects.boxes.conf.cpu().detach().numpy()
                )

        else:
            class_name = None
            confidence_from_model = None
            detected_objects = None
            num_detections = 0

        detected_object = DetectedObject(
            name=class_name,
            model_name=self.model_name,
            confidence_of_all_obj=confidence_from_model,
            probability_of_all_obj=[],
            all_obj_detected=detected_objects,
            number_of_detection=num_detections,
            is_detected=bool(num_detections > 0),
            bounding_box_of_all_obj=self.get_bounding_boxes(detected_objects),
        )

        if self.calibration_method:
            # calibrate confidence score
            if "8" not in self.model_name:
                logging.warning(
                    "Temperature scaling calibration is only supported for YOLOv8 models."  # noqa: E501
                )
                return self.no_calibration(detected_object)
            return self.calibrate(detected_object)

        return self.no_calibration(detected_object)

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
                probabilities.append(
                    self.calibrate_confidence(
                        confidence=confidence,
                        calibration_func=self.calibrate_function,
                    )
                )
            detected_object.probability_of_all_obj = probabilities
            detected_object.probability = max(probabilities)
        return detected_object

    def calibrate_function(
        self,
        confidence_per_video: float,
        true_threshold: float = 0.60,  # 0.60,
        false_threshold: float = 0.40,  # 0.40,
        k: float = 7.024,
        x0: float = 0.117,
    ) -> float:
        """Mapping probability.

        Args:
            confidence_per_video (float): Confidence per video.
            true_threshold (float, optional): True threshold. Defaults to 0.64
            false_threshold (float, optional): False threshold. Defaults to .38
            k (float, optional): k. Defaults to 7.024
            x0 (float, optional): x0. Defaults to 0.117

        Returns:
            float: Mapped probability.
        """
        if confidence_per_video >= true_threshold:
            return 1.0
        if confidence_per_video < false_threshold:
            return 0.0
        return 1 / (1 + np.exp(-k * (confidence_per_video - x0)))

    def no_calibration(self, detected_object: DetectedObject) -> DetectedObject:
        """No calibration.

        Args:
            detected_object (DetectedObject): Detected object.
        """
        probabilities = []
        if detected_object.is_detected:
            for confidence in detected_object.confidence_of_all_obj:
                probabilities.append(confidence)
            detected_object.probability_of_all_obj = probabilities
            detected_object.probability = max(probabilities)
        return detected_object

    def get_class_id_from_name(self, class_name: str) -> int:
        """Get class id.

        Args:
            class_name (str): Class name.

        Returns:
            int: Class id.
        """
        return self.english_to_class_id.get(class_name)

    @staticmethod
    def available_models() -> None:
        """Get available models."""
        pprint(MODEL_PATH.keys())  # noqa: T203
        return list(MODEL_PATH.keys())
