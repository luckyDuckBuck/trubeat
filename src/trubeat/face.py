"""Face + landmark tracking via MediaPipe Face Landmarker. FR-03."""

import numpy as np


def track_landmarks(frames: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Run Face Landmarker on every frame.

    Returns:
        landmarks: float32 (T, 478, 2) in pixel coords (NaN where invalid)
        valid: bool (T,) — face confidently tracked in this frame
    Re-detects when tracking is lost; caller warns if valid.mean() < 0.7.
    """
    raise NotImplementedError
