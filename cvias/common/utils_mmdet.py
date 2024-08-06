"""Utility to manage mmdetection models."""

import logging
import os
import subprocess
from pathlib import Path

import pkg_resources

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

ROOT_DIR = Path(__file__).parent.parent.parent


def mmdet_install() -> None:
    """Install mmdet dependencies."""
    try:
        # Upgrade pip
        subprocess.run(
            ["python3", "-m", "pip", "install", "--upgrade", "pip"],  # noqa: S603, S607 RUF100
            check=True,
        )

        # Install PyTorch
        subprocess.run(
            ["python3", "-m", "pip", "install", "torch==1.13.1"],  # noqa: S603, S607 RUF100
            check=True,
        )

        # Install torchvision
        subprocess.run(
            ["python3", "-m", "pip", "install", "torchvision==0.15.1"],  # noqa: S603, S607 RUF100
            check=True,
        )

        # Install openmim
        subprocess.run(
            ["python3", "-m", "pip", "install", "-U", "openmim"],  # noqa: S603, S607 RUF100
            check=True,
        )

        # Install mmdetection
        subprocess.run(
            ["python3", "-m", "pip", "install", "mmdet==3.0.0"],  # noqa: S603, S607 RUF100
            check=True,
        )

    except subprocess.CalledProcessError as e:
        msg = f"An error occurred: {e}"
        logging.exception(msg)


def check_and_install(package: str) -> None:
    """Check if the package is installed and install it if it is not."""
    try:
        # Check if the package is already installed and meets the version
        pkg_resources.require(package)
        msg = (
            f"{package} is already installed and meets the version requirement."
        )
        logging.info(msg)
    except pkg_resources.DistributionNotFound:
        # The package is not installed
        msg = f"{package} is not installed. Installing now..."
        logging.info(msg)
        os.chdir(ROOT_DIR)
        try:
            mmdet_install()
        except subprocess.CalledProcessError:
            msg = (
                "Failed to install mmdet dependencies.",
                " Command '{e.cmd}' exited with status {e.returncode}",
            )
        logging.exception(msg)
        subprocess.run(["mim", "install", package], check=True)  # noqa: S607, S603
    except pkg_resources.VersionConflict:
        # The installed version does not meet the requirement
        msg = f"{package} version conflict. Updating now..."
        logging.info(msg)
        subprocess.run(["mim", "install", package], check=True)  # noqa: S607, S603


def install_dependencies() -> None:
    """Install the required dependencies."""
    # List of packages to check and install
    packages = ["mmengine", "mmcv<2.1.0,>=2.0.0rc4"]
    for package in packages:
        check_and_install(package)


def load_class_label(dataset_type: str) -> dict:
    """Load class labels for the dataset type."""
    msg = f"Loading class labels for {dataset_type} dataset."
    logging.info(msg)
    if dataset_type == "CocoDataset":
        from .class_mapper.coco import CLASSES

    return CLASSES
