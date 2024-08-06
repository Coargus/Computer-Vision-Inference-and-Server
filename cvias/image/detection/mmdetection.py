"""MMDetection module for Computer Vision in Autonomous Systems (CVIAS)."""

from __future__ import annotations

import logging
import subprocess
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import torch
    from cog_cv_abstraction.schema.detected_object import DetectedObject
    from cog_cv_abstraction.schema.detected_object_set import DetectedObjectSet


from cogutil.download import coargus_cache_dir
from cogutil.torch import get_device
from mmdet.apis import init_detector
from mmdet.utils import register_all_modules
from mmengine.config import Config

from cvias.common.utils_mmdet import install_dependencies, load_class_label
from cvias.image.detection import CviasDetectionModel

register_all_modules()

warnings.filterwarnings("ignore")


class MMDetection(CviasDetectionModel):
    """Computer Vision model from MMDetection."""

    def __init__(
        self,
        model_name: str = "YOLOv9e",
        explicit_checkpoint_path: Path | None = None,
        gpu_number: int = 0,
    ) -> None:
        """Initializing MMDetection Module based on CviasDetectionModel."""
        super().__init__()
        logging.info("Trying to install dependencies for MMDetection")
        install_dependencies()
        if explicit_checkpoint_path:
            self.checkpoint, self.config_path = (
                self.retrieve_model_checkpoint_config(explicit_checkpoint_path)
            )
        else:
            self.checkpoint, self.config_path = self.download_model(model_name)

        self.config = Config.fromfile(self.config_path)
        self.device = get_device(gpu_number)
        self.model = self.load_model(
            config_file=self.config_path,
            checkpoint_file=self.checkpoint,
            device=self.device,
        )
        self.class_id_to_english = self.get_label_class()
        self.english_to_class_id = {
            v: k for k, v in self.class_id_to_english.items()
        }

    def retrieve_model_checkpoint_config(
        self, destination_path: Path
    ) -> tuple[Path, Path]:
        """Retrieve model checkpoint and config."""
        try:
            # Get the first .pth file found
            checkpoint = next(destination_path.glob("*.pth")).as_posix()
            try:
                # Get the first .py file found
                config = next(destination_path.glob("*.py")).as_posix()
            except StopIteration as err:
                msg = "No .py file found in the specified directory."
                raise FileNotFoundError(msg) from err
        except StopIteration as err:
            msg = "No .pth file found in the specified directory."
            raise FileNotFoundError(msg) from err
        return checkpoint, config

    def download_model(self, model_name: str) -> None:
        """Download model to cache directory."""
        coargus_path: Path = coargus_cache_dir()
        destination_path: Path = coargus_path / model_name
        try:
            checkpoint = next(destination_path.glob("*.pth"))
            config = next(destination_path.glob("*.py"))
        except StopIteration:
            checkpoint = None
        if checkpoint:
            msg = (f"Model already exists: {destination_path!s}",)
            logging.info(msg)
        else:
            command = [
                "mim",
                "download",
                "mmdet",
                "--config",
                model_name,
                "--dest",
                str(destination_path),
            ]

            try:
                subprocess.run(
                    command,  # noqa: S603
                    check=True,
                    capture_output=True,
                    text=True,
                )

                msg = (f"Model has been downloaded to: {destination_path!s}",)
                logging.info(msg)

                checkpoint, config = self.retrieve_model_checkpoint_config(
                    destination_path
                )

            except subprocess.CalledProcessError as e:
                msg = f"Error occurred during downloading a model: {e.stderr}"
                logging.exception(msg)
        return str(checkpoint), str(config)

    def load_model(
        self, config_file: str, checkpoint_file: str, device: torch.device
    ) -> torch.nn.Module:
        """Load weight.

        Args:
            config_file (str): Config file path.
            checkpoint_file (str): Checkpoint file path.
            device (torch.device): Device.

        Returns:
            Detector Class from MMDetection
        """
        return init_detector(config_file, checkpoint_file, device=device)

    def get_label_class(self) -> dict:
        """Get label class."""
        try:
            return self.model.CLASSES
        except AttributeError:
            msg = "Model does not have attribute CLASSES. Attempting to retrieve from config."  # noqa: E501
            logging.warning(msg)
            try:
                return self.get_classes_from_config(self.config)
            except AttributeError:
                msg = "Unable to retrieve class names from configuration."
                logging.warning(msg)
                try:
                    return load_class_label(self.config.dataset_type)
                except AttributeError as err:
                    msg = "Unable to retrieve class names."
                    raise AttributeError(msg) from err

    def calibrate(self, model: DetectedObject) -> DetectedObject:
        """Calibrate model."""
        raise NotImplementedError

    def detect(
        self, frame_img: np.ndarray, classes: list
    ) -> DetectedObjectSet | DetectedObject:
        """Detect object in frame."""
        raise NotImplementedError
