"""CVIAS Detection Module."""

from __future__ import annotations

from calibrate_cv.conformal_prediction import (
    get_non_conformity_score,
    process_raw_data_distribution,
)
from cog_cv_abstraction.image.detection.object._base import (
    ObjectDetectionModelBase,
)


class CviasDetectionModel(ObjectDetectionModelBase):
    """CVIAS Detection Model."""

    def __init__(self, calibration_method: str | None = None) -> None:
        """Initialize CVIAS Detection Model."""
        super().__init__()
        self.calibration_method = calibration_method

    def calibrate_model_with_conformal_prediction(
        self,
        data_distribution_path: str,
        sample_number: int = 2000,
        random_state: int = 1,
    ) -> None:
        """Calibrate model with conformal prediction."""
        confidence, prediction, ground_truth = process_raw_data_distribution(
            data_distribution_path=data_distribution_path,
            sample_number=sample_number,
            random_state=random_state,
        )
        self.non_conformity_score = get_non_conformity_score(
            confidence=confidence,
            prediction=prediction,
            ground_truth=ground_truth,
        )
        self.calibration_method = "conformal_prediction"

    def calibrate_confidence(
        self,
        confidence: float,
        calibration_func: callable | None = None,
    ) -> float:
        """Calibrate confidence score."""
        if self.calibration_method == "conformal_prediction":
            from calibrate_cv.conformal_prediction import (
                calibrate_confidence_score,
            )

            return calibrate_confidence_score(
                confidence=confidence,
                non_conformity_score=self.non_conformity_score,
            )

        return calibration_func(confidence)
