"""ViClip class to get feature vectors of a video clip."""

from __future__ import annotations

import numpy as np
from cog_cv_abstraction.video.embedding._base import (
    ComputerVisionModelVideoEmbeddingBase,
)
from viclip_base import frames2tensor, get_viclip, get_vid_feat


class ViClip(ComputerVisionModelVideoEmbeddingBase):
    """ViClip class to get feature vectors of a video clip."""

    def __init__(self) -> None:
        """Initialize the ViClip model."""
        self.model_cfg = {
        "size": "l",
        "pretrained": "/opt/mars/mnt/ \
            model_weights/viclip/ViClip-InternVid-10M-FLT.pth",
        }

        self.model = get_viclip(
            self.model_cfg["size"],
            self.model_cfg["pretrained"]
        )
        self.clip = self.model["viclip"]
        self.tokenizer = self.model["tokenizer"]

        self.clip = self.clip.to("cuda")

    def get_feature(self, frames: np.ndarray) -> any:
        """Get feature vectors of a video clip.

        Args:
            frames: frames of the video clip

        Returns:
            The feature vectors of the video clip
        """
        frames_tensor = frames2tensor(frames, device="cuda")
        return get_vid_feat(frames_tensor, self.clip).cpu()


if __name__ == "__main__":
    # Example usage
    viclip = ViClip()
    rng = np.random.default_rng()
    frames = rng.random((8, 224, 224, 3))
    feature = viclip.get_feature(frames)
