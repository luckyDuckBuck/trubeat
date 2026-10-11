"""Multi-region pulse consistency analysis. FR-09."""

import numpy as np


def coherence_features(ppg_win: np.ndarray, fps: float) -> dict:
    """Pairwise magnitude-squared coherence between ROI signals in a window.

    Args:
        ppg_win: float (W, R) — one window of per-ROI pulse signals
    Returns: dict with coherence mean/std/min/max over ROI pairs + per-pair values.
    On a real face the regions rise and fall together (high coherence);
    on a fake they are less consistent.
    """
    raise NotImplementedError


def lag0_correlation(ppg_win: np.ndarray) -> dict:
    """Cross-correlation at lag 0 for every ROI pair → mean/std/min."""
    raise NotImplementedError
