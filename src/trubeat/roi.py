"""ROI definitions + per-frame mean RGB extraction. FR-04, FR-05."""

import numpy as np

# Landmark index lists per ROI (verify by plotting on one frame, Phase 1 M1.2).
# MediaPipe Face Landmarker topology: 478 points. Fill these in during M1.2.
ROI_DEFS_P1: dict[str, list[int]] = {
    "forehead": [],  # e.g. indices around the forehead polygon
    "cheek_l": [],
    "cheek_r": [],
}

# Phase 3: 32 sub-regions tiled across the face (FakeCatcher layout).
ROI_DEFS_P3: dict[str, list[int]] = {}


def extract_roi_means(
    frames: np.ndarray,
    landmarks: np.ndarray,
    valid: np.ndarray,
    roi_defs: dict[str, list[int]],
) -> np.ndarray:
    """Mean R/G/B inside each ROI polygon, per frame.

    Returns:
        traces: float (T, R, 3). Rows for invalid frames are NaN.
    """
    raise NotImplementedError
