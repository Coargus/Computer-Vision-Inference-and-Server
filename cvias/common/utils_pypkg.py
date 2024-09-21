"""Utility function for python package."""

import json
import logging
import subprocess
import sys
from pathlib import Path

from cogutil.download import coargus_cache_dir

ROOT_PATH = Path(__file__).parent.parent.parent


def import_or_install(module_name: str, attributes: list[str]) -> list:
    """Import a module or install it if not present."""
    try:
        module = __import__(module_name)
        return [getattr(module, attr) for attr in attributes]
    except ImportError:
        install_requirements(module_name)
        module = __import__(module_name)
        return [getattr(module, attr) for attr in attributes]


def install_requirements(module_name: str) -> None:
    """Install requirements from requirements.txt."""
    model_path = coargus_cache_dir() / module_name
    model_metadata = model_path / "metadata.json"
    if not model_metadata.exists():
        module_package_path = (
            ROOT_PATH / "requirements" / module_name / "requirements.txt"
        )
        logging.info(f"Installing required packages for {module_name}.")
        try:
            subprocess.run(  # noqa: S603
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-r",
                    str(module_package_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            # Ensure the directory exists before saving the metadata
            model_metadata.parent.mkdir(
                parents=True, exist_ok=True
            )  # Create parent directories if they don't exist

            with model_metadata.open(
                "w"
            ) as f:  # Open file for writing using Path object
                json.dump({"dependency_install": True}, f)  # Save the JSON data

        except subprocess.CalledProcessError as e:
            logging.exception(
                f"Error occurred: {e.stderr}"
            )  # Print the error output
            logging.exception(
                "Failed to install dependencies. Please check your internet connection and try again."  # noqa: E501
            )
            sys.exit(1)
