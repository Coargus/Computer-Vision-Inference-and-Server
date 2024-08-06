"""Faster RCNN model for open vocabulary object detection."""

import logging
from pathlib import Path
from pprint import pprint

import numpy as np
from cog_cv_abstraction.schema.detected_object import DetectedObject
from cog_cv_abstraction.schema.detected_object_set import DetectedObjectSet
from mmdet.apis import inference_detector

from cvias.image.detection.mmdetection import MMDetection

MODEL_PATH = {
    "faster_rcnn_r50_fpn_1x_coco": "https://download.openxlab.org.cn/models/mmdetection/FasterR-CNN/weight/faster-rcnn_r101_fpn_1x_coco",
    "faster-rcnn_x101-64x4d_fpn_ms-3x_coco": "https://download.openxlab.org.cn/models/mmdetection/FasterR-CNN/weight/faster-rcnn_x101-64x4d_fpn_2x_coco",
}


class FasterRCNN(MMDetection):
    """Open vocabulary Faster RCNN model."""

    def __init__(
        self,
        model_name: str = "faster_rcnn_r50_fpn_1x_coco",
        explicit_checkpoint_path: Path | None = None,
        gpu_number: int = 0,
    ) -> None:
        """Initialization."""
        super().__init__(model_name, explicit_checkpoint_path, gpu_number)
        self.model_name = model_name

    def detect(self, frame_img: np.ndarray, classes: list | None = None) -> any:
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
        # Perform inference
        detected_objects = inference_detector(self.model, frame_img)
        confidence_from_model = (
            detected_objects.pred_instances.scores.cpu().numpy()
        )
        bboxes = detected_objects.pred_instances.bboxes.cpu().numpy()
        detected_classes = detected_objects.pred_instances.labels.cpu().numpy()

        num_detections = len(bboxes)
        is_detected = bool(num_detections > 0)
        detected_obj_set = DetectedObjectSet()

        if is_detected:
            for idx, detected_class_id in enumerate(detected_classes):
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
                    bboxes[idx][:4].tolist()
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

    @staticmethod
    def available_models() -> None:
        """Get available models."""
        pprint(MODEL_PATH.keys())  # noqa: T203
        return list(MODEL_PATH.keys())
