"""Video ingest: decode + metadata. FR-01, FR-02."""

from dataclasses import dataclass

import numpy as np

from .errors import LowFps, ShortVideo, UnreadableVideo

MIN_FPS = 15.0
MIN_DURATION_S = 10.0


@dataclass
class VideoStream:
    frames: np.ndarray  # uint8 (T, H, W, 3), RGB
    fps: float
    n_frames: int
    width: int
    height: int
    duration_s: float
    path: str


def open_video(path: str) -> VideoStream:
    """Decode a video file and validate it for rPPG work.

    Reads the TRUE frame rate from the container (never assumes 30).
    Raises E_UNREADABLE / E_LOW_FPS / E_SHORT_VIDEO on bad inputs.
    """
    raise NotImplementedError
