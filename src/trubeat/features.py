"""Window-level feature vectors + Phase-3 PPG maps. FR-10."""

import numpy as np
import pandas as pd

from .consistency import coherence_features, lag0_correlation

# Window parameters for build_ppg_maps (signature stays (ppg, fps)).
WIN_S = 10.0
OVERLAP = 0.5
# Detection band in Hz. FakeCatcher keeps the residual "noise" above the pulse
# band because it helps spot fakes (docs/04 §5).
DETECT_BAND = (0.7, 14.0)

_EPS = 1e-12

FEATURE_COLUMNS = [
    "win_start_s",
    "win_end_s",
    "coh_mean",
    "coh_std",
    "coh_min",
    "coh_max",
    "lag0_mean",
    "lag0_std",
    "lag0_min",
    "spec_entropy",
    "peak_sharpness",
    "snr_mean",
]


def _band_power(x: np.ndarray, fps: float) -> np.ndarray:
    """Power spectrum of x restricted to DETECT_BAND (DC removed)."""
    n = len(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    power = np.abs(np.fft.rfft(x - x.mean())) ** 2
    mask = (freqs >= DETECT_BAND[0]) & (freqs <= DETECT_BAND[1])
    return power[mask]


def _spectral_entropy(power: np.ndarray) -> float:
    """Normalized spectral entropy of a power spectrum in [0, 1].

    A pure periodic pulse concentrates power in a few bins → entropy near 0;
    broadband noise spreads it → entropy near 1.
    """
    total = power.sum()
    if total <= _EPS or len(power) < 2:
        return 0.0
    p = power / total
    return float(-np.sum(p * np.log(p + _EPS)) / np.log(len(p)))


def _peak_sharpness(power: np.ndarray) -> float:
    """Dominant-peak prominence: max bin power / mean bin power in band."""
    if len(power) == 0:
        return 0.0
    return float(power.max() / (power.mean() + _EPS))


def _snr(power: np.ndarray) -> float:
    """Per-signal SNR estimate: peak bin power / mean bin power in band."""
    return _peak_sharpness(power)


def _window_bounds(n_samples: int, fps: float, win_s: float, overlap: float):
    """Yield (start_idx, end_idx) sample windows; empty if none fit."""
    win_n = int(round(win_s * fps))
    step_n = int(round(win_n * (1.0 - overlap)))
    if win_n < 2 or step_n < 1 or n_samples < win_n:
        return
    yield from (
        (s, s + win_n) for s in range(0, n_samples - win_n + 1, step_n)
    )


def window_features(
    ppg: np.ndarray, fps: float, win_s: float = 10.0, overlap: float = 0.5
) -> pd.DataFrame:
    """Slide windows over (T, R) ppg → one feature row per window.

    Columns (documented):
      - ``win_start_s``, ``win_end_s``: window boundaries in seconds.
      - ``coh_mean``, ``coh_std``, ``coh_min``, ``coh_max``: magnitude-squared
        coherence aggregated over ROI pairs (see
        :func:`trubeat.consistency.coherence_features`). High on real faces
        whose regions rise and fall together.
      - ``lag0_mean``, ``lag0_std``, ``lag0_min``: Pearson r at lag 0 over ROI
        pairs (see :func:`trubeat.consistency.lag0_correlation`).
      - ``spec_entropy``: normalized spectral entropy ([0, 1]) of the
        mean-face signal's in-band power spectrum — low for a clean periodic
        pulse, high for noise.
      - ``peak_sharpness``: dominant in-band peak power / mean in-band power
        of the mean-face signal — high when one heartbeat frequency dominates.
      - ``snr_mean``: mean over ROIs of per-ROI peak/mean in-band power.

    Spectral features use the mean-face signal (average of ROI signals) and
    the DETECT_BAND (0.7–14 Hz). Returns an empty DataFrame with the exact
    schema when no full window fits.
    """
    ppg = np.asarray(ppg, dtype=float)
    if ppg.ndim != 2:
        raise ValueError(f"ppg must be 2-D (T, R), got shape {ppg.shape}")
    n_samples, n_roi = ppg.shape
    rows = []
    for start, end in _window_bounds(n_samples, fps, win_s, overlap):
        w = ppg[start:end]
        coh = coherence_features(w, fps)
        lag0 = lag0_correlation(w)
        mean_face = w.mean(axis=1)
        p_mean = _band_power(mean_face, fps)
        snrs = [_snr(_band_power(w[:, r], fps)) for r in range(n_roi)]
        rows.append(
            {
                "win_start_s": start / fps,
                "win_end_s": end / fps,
                "coh_mean": coh["coh_mean"],
                "coh_std": coh["coh_std"],
                "coh_min": coh["coh_min"],
                "coh_max": coh["coh_max"],
                "lag0_mean": lag0["lag0_mean"],
                "lag0_std": lag0["lag0_std"],
                "lag0_min": lag0["lag0_min"],
                "spec_entropy": _spectral_entropy(p_mean),
                "peak_sharpness": _peak_sharpness(p_mean),
                "snr_mean": float(np.mean(snrs)) if snrs else 0.0,
            }
        )
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


def _band_n_bins(win_n: int, fps: float) -> int:
    freqs = np.fft.rfftfreq(win_n, d=1.0 / fps)
    return int(((freqs >= DETECT_BAND[0]) & (freqs <= DETECT_BAND[1])).sum())


def build_ppg_maps(ppg: np.ndarray, fps: float) -> np.ndarray:
    """Phase 3: per-region spectra stacked into 2-D PPG maps.

    For each sliding window (WIN_S seconds, OVERLAP fraction) and each face
    region, takes the rfft magnitude restricted to DETECT_BAND (0.7–14 Hz)
    and normalizes it per region (÷ max + eps), so the map encodes spectral
    *shape* rather than amplitude scale.

    Returns: float array of shape (R, F, n_windows) — regions × in-band
    frequency bins × windows. This is the CNN input in the FakeCatcher design
    (docs/04 §4 describes it as (R, F, T); the third axis indexes the
    sliding windows defined by WIN_S/OVERLAP).

    An input shorter than one full window yields shape (R, F, 0).
    """
    ppg = np.asarray(ppg, dtype=float)
    if ppg.ndim != 2:
        raise ValueError(f"ppg must be 2-D (T, R), got shape {ppg.shape}")
    n_samples, n_roi = ppg.shape
    win_n = int(round(WIN_S * fps))
    n_freq = _band_n_bins(win_n, fps) if win_n >= 2 else 0
    bounds = list(_window_bounds(n_samples, fps, WIN_S, OVERLAP))
    if not bounds or n_roi < 1:
        return np.zeros((n_roi, n_freq, 0))
    freqs = np.fft.rfftfreq(win_n, d=1.0 / fps)
    mask = (freqs >= DETECT_BAND[0]) & (freqs <= DETECT_BAND[1])
    maps = []
    for start, end in bounds:
        w = ppg[start:end]  # (win_n, R)
        spec = np.abs(np.fft.rfft(w - w.mean(axis=0, keepdims=True), axis=0))
        band = spec[mask, :]  # (F, R)
        band = band / (band.max(axis=0, keepdims=True) + _EPS)
        maps.append(band.T)  # (R, F)
    return np.stack(maps, axis=-1)  # (R, F, n_windows)
