"""Signal conditioning + rPPG extraction. FR-06, FR-07."""

import numpy as np
from scipy.signal import butter, detrend, sosfiltfilt


def condition(trace: np.ndarray, fps: float, band: tuple[float, float] = (0.7, 3.0)) -> np.ndarray:
    """Detrend slow drift, then zero-phase SOS band-pass filter.

    Args:
        trace: 1-D signal (T,)
        fps: true frame rate of the video
        band: (low, high) Hz — (0.7, 3.0) for HR, (0.7, 14.0) for detection
    """
    raise NotImplementedError


def extract_ppg(traces: np.ndarray, fps: float, method: str = "pos") -> np.ndarray:
    """Pulse waveform per ROI from (T, R, 3) mean-RGB traces.

    method ∈ {"green", "chrom", "pos"}.
    Returns: ppg float (T, R).
    """
    raise NotImplementedError
