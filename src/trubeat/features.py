"""Window-level feature vectors + Phase-3 PPG maps. FR-10."""

import numpy as np
import pandas as pd


def window_features(
    ppg: np.ndarray, fps: float, win_s: float = 10.0, overlap: float = 0.5
) -> pd.DataFrame:
    """Slide windows over (T, R) ppg → one feature row per window.

    Features: coherence mean/std/min, lag-0 correlation stats, spectral
    entropy, dominant-peak sharpness, per-ROI SNR.
    """
    raise NotImplementedError


def build_ppg_maps(ppg: np.ndarray, fps: float) -> np.ndarray:
    """Phase 3: per-region time signal + power spectrum stacked into 2-D maps.

    Returns: float (R, F, T) — the CNN input from the FakeCatcher paper
    (32 face regions × frequency × time).
    """
    raise NotImplementedError
