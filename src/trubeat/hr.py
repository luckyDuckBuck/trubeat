"""Heart-rate estimation from the pulse waveform. FR-08."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class HRResult:
    bpm: float
    confidence: float  # peak SNR in the pulse band
    spectrum: np.ndarray = field(repr=False)
    freqs: np.ndarray = field(repr=False)


def estimate_hr(
    ppg: np.ndarray, fps: float, band: tuple[float, float] = (0.7, 3.0)
) -> HRResult:
    """FFT peak-pick inside the pulse band → bpm.

    bpm = peak_freq_hz * 60. A 10 s window at 30 fps gives 0.1 Hz bins
    (6 bpm steps); prefer 30 s windows (2 bpm steps) when reporting HR.
    """
    raise NotImplementedError
