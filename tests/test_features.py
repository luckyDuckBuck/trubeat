"""Tests for trubeat.features — schema, windows, and PPG maps."""

import numpy as np
import pandas as pd
import pytest

import synth
from trubeat.features import (
    FEATURE_COLUMNS,
    OVERLAP,
    WIN_S,
    build_ppg_maps,
    window_features,
)

FPS = 30.0

EXPECTED_COLUMNS = [
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


def _coherent_ppg(seconds=60.0, n_roi=3, bpm=72.0):
    n = int(seconds * FPS)
    return np.stack(
        [synth.pulse_wave(n, FPS, bpm) for _ in range(n_roi)], axis=1
    )


def _independent_ppg(seconds=60.0):
    n = int(seconds * FPS)
    return np.stack(
        [
            synth.pulse_wave(n, FPS, b, noise_std=0.3, seed=i)
            for i, b in enumerate((72.0, 95.0, 130.0))
        ],
        axis=1,
    )


def test_schema_exact():
    df = window_features(_coherent_ppg(), FPS)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == EXPECTED_COLUMNS
    assert list(df.columns) == FEATURE_COLUMNS


def test_window_count_and_bounds():
    # 60 s @30fps, 10 s windows, 50% overlap → starts 0,5,…,50 → 11 windows.
    df = window_features(_coherent_ppg(60.0), FPS, win_s=10.0, overlap=0.5)
    assert len(df) == 11
    assert df["win_start_s"].iloc[0] == pytest.approx(0.0)
    assert df["win_end_s"].iloc[0] == pytest.approx(10.0)
    assert df["win_start_s"].iloc[1] == pytest.approx(5.0)
    assert df["win_end_s"].iloc[-1] == pytest.approx(60.0)


def test_coherent_windows_high_coherence():
    # 30 s windows: coherence can discriminate (see consistency caveat).
    df = window_features(_coherent_ppg(60.0), FPS, win_s=30.0, overlap=0.5)
    assert (df["coh_mean"] > 0.9).all()
    assert (df["lag0_mean"] > 0.9).all()
    # Clean periodic pulse: low spectral entropy, sharp dominant peak.
    assert (df["spec_entropy"] < 0.5).all()
    assert (df["peak_sharpness"] > 5.0).all()
    assert (df["snr_mean"] > 5.0).all()


def test_independent_windows_low_coherence():
    df = window_features(_independent_ppg(60.0), FPS, win_s=30.0, overlap=0.5)
    assert (df["coh_mean"] < 0.3).all()
    assert (abs(df["lag0_mean"]) < 0.3).all()


def test_no_nan_anywhere():
    df = window_features(_independent_ppg(60.0), FPS)
    df2 = window_features(np.full((1800, 3), 4.0), FPS)  # degenerate constant
    for d in (df, df2):
        assert d.notna().all().all()


def test_short_input_empty_schema():
    df = window_features(np.zeros((100, 3)), FPS)  # 3.3 s < 10 s window
    assert list(df.columns) == EXPECTED_COLUMNS
    assert len(df) == 0


def test_ppg_map_shape():
    n_roi = 4
    maps = build_ppg_maps(_coherent_ppg(60.0, n_roi=n_roi), FPS)
    win_n = int(WIN_S * FPS)
    freqs = np.fft.rfftfreq(win_n, d=1.0 / FPS)
    n_freq = int(((freqs >= 0.7) & (freqs <= 14.0)).sum())
    n_win = len(range(0, 1800 - win_n + 1, int(win_n * (1.0 - OVERLAP))))
    assert maps.shape == (n_roi, n_freq, n_win)


def test_ppg_map_per_region_normalized():
    maps = build_ppg_maps(_coherent_ppg(30.0, n_roi=3), FPS)
    assert maps.shape[2] > 0
    assert np.all(maps >= 0.0)
    # Per-region spectra are normalized: each region/window peaks at 1.0.
    assert np.allclose(maps.max(axis=1), 1.0)


def test_ppg_map_short_input():
    maps = build_ppg_maps(np.zeros((100, 3)), FPS)
    assert maps.shape[0] == 3
    assert maps.shape[2] == 0


def test_window_features_interpolates_nan_gaps():
    """Dropped frames (NaN rows, e.g. green-method path) must not poison windows."""
    n = 750
    t = np.arange(n) / 30.0
    ppg = np.column_stack([np.sin(2 * np.pi * 1.2 * t + p) for p in (0.0, 0.1, 0.2)])
    ppg[100:120] = np.nan  # simulated dropped frames
    feat = window_features(ppg, 30.0)
    assert len(feat) == 4  # 25 s, 10 s windows, 50% overlap
    vals = feat.drop(columns=["win_start_s", "win_end_s"]).to_numpy()
    assert np.all(np.isfinite(vals))


def test_window_features_all_nan_column_gives_zeros():
    """A column with no usable signal at all → finite zero features, never NaN."""
    n = 750
    t = np.arange(n) / 30.0
    ppg = np.column_stack(
        [np.sin(2 * np.pi * 1.2 * t), np.full(n, np.nan), np.sin(2 * np.pi * 1.2 * t)]
    )
    feat = window_features(ppg, 30.0)
    vals = feat.drop(columns=["win_start_s", "win_end_s"]).to_numpy()
    assert np.all(np.isfinite(vals))
