"""Heart-rate estimation from the pulse waveform. FR-08."""

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import detrend


@dataclass
class HRResult:
    bpm: float
    confidence: float  # peak in-band power / total in-band power
    spectrum: np.ndarray = field(repr=False)  # in-band power spectrum
    freqs: np.ndarray = field(repr=False)  # in-band frequencies (Hz)


def _interp_nans(x: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN gaps so a few dropped frames can't poison
    the spectrum. Leading/trailing NaNs take the nearest valid value."""
    x = x.copy()
    nans = np.isnan(x)
    if np.any(nans) and not np.all(nans):
        idx = np.arange(x.shape[0])
        x[nans] = np.interp(idx[nans], idx[~nans], x[~nans])
    return x


def estimate_hr(
    ppg: np.ndarray, fps: float, band: tuple[float, float] = (0.7, 3.0)
) -> HRResult:
    """FFT peak-pick inside the pulse band → bpm.

    The spectrum's strongest in-band bin is refined with parabolic
    interpolation for sub-bin accuracy; bpm = peak_freq_hz * 60.
    Confidence is the peak bin's power over the total in-band power
    (1.0 for a pure tone, near 0 for noise).

    A 10 s window at 30 fps gives 0.1 Hz bins (6 bpm steps); prefer 30 s
    windows (2 bpm steps) when reporting HR. If no frequency bin falls in
    the band, bpm is NaN with 0 confidence.
    """
    x = np.asarray(ppg, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"estimate_hr expects a 1-D PPG signal, got shape {x.shape}")
    if x.shape[0] < 4:
        raise ValueError("estimate_hr needs at least 4 samples")
    x = _interp_nans(x)
    if not np.all(np.isfinite(x)):
        # Degenerate input (e.g. no valid frames at all): NaN bpm, 0 confidence.
        return HRResult(bpm=float("nan"), confidence=0.0,
                        spectrum=np.array([]), freqs=np.array([]))
    x = detrend(x, type="linear")

    n = x.shape[0]
    power = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    lo, hi = float(band[0]), float(band[1])
    mask = (freqs >= lo) & (freqs <= hi)
    f, p = freqs[mask], power[mask]
    if p.size == 0 or not np.all(np.isfinite(p)):
        return HRResult(bpm=float("nan"), confidence=0.0, spectrum=p, freqs=f)

    k = int(np.argmax(p))
    if 0 < k < p.size - 1:
        y0, y1, y2 = p[k - 1], p[k], p[k + 1]
        denom = y0 - 2.0 * y1 + y2
        delta = 0.5 * (y0 - y2) / denom if denom != 0.0 else 0.0
        delta = max(-1.0, min(1.0, delta))
    else:
        delta = 0.0
    df = float(f[1] - f[0]) if f.size > 1 else 0.0
    f_peak = float(f[k] + delta * df)

    total = float(np.sum(p))
    confidence = float(p[k] / total) if total > 0.0 else 0.0
    return HRResult(bpm=f_peak * 60.0, confidence=confidence, spectrum=p, freqs=f)
