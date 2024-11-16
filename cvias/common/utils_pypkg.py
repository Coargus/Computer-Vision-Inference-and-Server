"""Utility function for python package."""

import json
import logging
import os
import sys
from pathlib import Path

from cogutil.download import coargus_cache_dir

ROOT_PATH = Path(__file__).parent.parent.parent


os.environ["PYDEVD_WARN_EVALUATION_TIMEOUT"] = "100"  # 30 seconds instead of 3


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
        logging.info(f"Installing required packages for {module_name}...")

        try:
            # Use pip as a module with progress bar
            import pip

            pip.main(
                [
                    "install",
                    "-r",
                    str(module_package_path),
                    "--progress-bar",
                    "on",
                ]
            )

            # Create metadata after successful installation
            model_metadata.parent.mkdir(parents=True, exist_ok=True)
            with model_metadata.open("w") as f:
                json.dump({"dependency_install": True}, f)

            logging.info(
                f"Successfully installed requirements for {module_name}"
            )

        except Exception:
            logging.exception("Error installing dependencies")
            logging.exception("Failed to install dependencies.")
            sys.exit(1)
