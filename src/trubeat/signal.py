"""Signal conditioning + rPPG extraction. FR-06, FR-07."""

import numpy as np
from scipy.signal import butter, detrend, sosfiltfilt

_BUTTER_ORDER = 4


def _interp_nans(x: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN gaps (frames where the face was not tracked).

    Leading/trailing NaNs are filled with the nearest valid value
    (np.interp boundary behavior). An all-NaN input is returned unchanged.
    """
    x = np.asarray(x, dtype=float).copy()
    nans = np.isnan(x)
    if not np.any(nans):
        return x
    if np.all(nans):
        return x
    idx = np.arange(x.shape[0])
    x[nans] = np.interp(idx[nans], idx[~nans], x[~nans])
    return x


def _safe_mean(c: np.ndarray) -> float:
    """Temporal mean guarded against zero/NaN (avoids division blow-ups)."""
    m = np.nanmean(c)
    if not np.isfinite(m) or m == 0.0:
        return 1.0
    return float(m)


def condition(trace: np.ndarray, fps: float, band: tuple[float, float] = (0.7, 3.0)) -> np.ndarray:
    """Detrend slow drift, then zero-phase SOS band-pass filter.

    NaN gaps (invalid frames) are linearly interpolated first — see
    `_interp_nans`. An all-NaN input comes back all-NaN.

    Args:
        trace: 1-D signal (T,)
        fps: true frame rate of the video
        band: (low, high) Hz — (0.7, 3.0) for HR, (0.7, 14.0) for detection
    """
    x = np.asarray(trace, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"condition expects a 1-D trace, got shape {x.shape}")
    if x.shape[0] == 0:
        raise ValueError("condition got an empty trace")
    lo, hi = float(band[0]), float(band[1])
    nyq = float(fps) / 2.0
    if not (0.0 < lo < hi < nyq):
        raise ValueError(
            f"band {band} must satisfy 0 < lo < hi < Nyquist ({nyq:.2f} Hz)"
        )
    x = _interp_nans(x)
    if not np.all(np.isfinite(x)):
        # No valid samples (e.g. face never tracked): nothing to condition.
        # Propagate NaNs rather than raising inside the filter math.
        return np.full_like(x, np.nan)
    x = detrend(x, type="linear")
    sos = butter(_BUTTER_ORDER, [lo, hi], btype="bandpass", fs=fps, output="sos")
    try:
        return sosfiltfilt(sos, x)
    except ValueError:
        # Signal shorter than the default edge padding: shrink padding to fit.
        return sosfiltfilt(sos, x, padlen=max(x.shape[0] - 1, 0))


def _chrom_one(rgb: np.ndarray, fps: float) -> np.ndarray:
    """CHROM (de Haan & Jeanne, 2013) for one ROI's (T, 3) RGB trace."""
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    rn, gn, bn = r / _safe_mean(r), g / _safe_mean(g), b / _safe_mean(b)
    x = 3.0 * rn - 2.0 * gn
    y = 1.5 * rn + gn - 1.5 * bn
    xf, yf = condition(x, fps), condition(y, fps)
    sx, sy = float(np.std(xf)), float(np.std(yf))
    alpha = sx / sy if sy > 0.0 else 0.0
    return xf - alpha * yf


def _pos_one(rgb: np.ndarray, fps: float) -> np.ndarray:
    """POS (Wang et al., 2017) for one ROI's (T, 3) RGB trace."""
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    rn, gn, bn = r / _safe_mean(r), g / _safe_mean(g), b / _safe_mean(b)
    s1 = gn - bn
    s2 = gn + bn - 2.0 * rn
    s1f, s2f = condition(s1, fps), condition(s2, fps)
    sd1, sd2 = float(np.std(s1f)), float(np.std(s2f))
    return s1f + (sd1 / sd2 if sd2 > 0.0 else 0.0) * s2f


def extract_ppg(traces: np.ndarray, fps: float, method: str = "pos") -> np.ndarray:
    """Pulse waveform per ROI from (T, R, 3) mean-RGB traces.

    method ∈ {"green", "chrom", "pos"}:
      - "green": the G channel as-is (strongest haemoglobin absorption).
        Run `condition()` on the output if standalone conditioning is wanted.
      - "chrom": de Haan 2013 chrominance method (illumination-robust).
      - "pos": Wang 2017 plane-orthogonal-to-skin (illumination/motion-robust).

    Returns: ppg float (T, R).
    """
    t = np.asarray(traces, dtype=float)
    if t.ndim != 3 or t.shape[2] != 3:
        raise ValueError(
            f"extract_ppg expects (T, R, 3) RGB traces, got shape {t.shape}"
        )
    n_rois = t.shape[1]
    if method == "green":
        return t[:, :, 1].copy()
    if method == "chrom":
        out = np.empty((t.shape[0], n_rois))
        for r in range(n_rois):
            out[:, r] = _chrom_one(t[:, r, :], fps)
        return out
    if method == "pos":
        out = np.empty((t.shape[0], n_rois))
        for r in range(n_rois):
            out[:, r] = _pos_one(t[:, r, :], fps)
        return out
    raise ValueError(
        f"unknown rPPG method {method!r}; expected 'green', 'chrom' or 'pos'"
    )
