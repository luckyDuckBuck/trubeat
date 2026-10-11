"""Tests for trubeat.signal: conditioning + CHROM/POS extraction."""

import numpy as np
import pytest

from trubeat.signal import condition, extract_ppg

import synth
from synth import pulse_wave

FPS = 30.0


def _rgb_traces(n, fps, bpm, r_gain=1.0, g_gain=2.0, b_gain=0.5,
                base=140.0, r_ramp=0.0, seed=0):
    """(T, R, 3) RGB traces: pulse in every channel (strongest in G),
    plus an optional slow ramp added to the R channel only
    (single-channel illumination variation)."""
    t = np.arange(n) / fps
    pulse = pulse_wave(n, fps, bpm, seed=seed)
    ramp = r_ramp * (t / t[-1])
    traces = np.zeros((n, 2, 3))
    traces[:, :, 0] = (base + r_gain * pulse + ramp)[:, None]  # R
    traces[:, :, 1] = (base + g_gain * pulse)[:, None]         # G
    traces[:, :, 2] = (base + b_gain * pulse)[:, None]         # B
    return traces, pulse


def test_condition_removes_slow_drift():
    n = int(30 * FPS)
    pure = pulse_wave(n, FPS, 72.0)
    drift = 0.08 * np.arange(n)  # slow linear drift, dwarfs the pulse
    out = condition(pure + drift, FPS)
    assert np.corrcoef(out, pure)[0, 1] > 0.99
    assert abs(out.mean()) < 0.05


def test_condition_interpolates_nan_gap():
    n = int(30 * FPS)
    pure = pulse_wave(n, FPS, 72.0)
    gappy = pure.copy()
    gappy[400:405] = np.nan  # a few dropped frames
    out = condition(gappy, FPS)
    ref = condition(pure, FPS)
    assert not np.isnan(out).any()
    # Only the neighbourhood of the gap may deviate, and only slightly.
    assert np.max(np.abs(out - ref)) < 0.15


def test_condition_all_nan_stays_nan():
    out = condition(np.full(300, np.nan), FPS)
    assert np.isnan(out).all()


def test_condition_rejects_bad_band():
    with pytest.raises(ValueError):
        condition(np.ones(300), FPS, band=(3.0, 0.7))  # inverted
    with pytest.raises(ValueError):
        condition(np.ones(300), FPS, band=(0.7, 20.0))  # above Nyquist (15 Hz)
    with pytest.raises(ValueError):
        condition(np.ones((10, 2)), FPS)  # not 1-D


def test_extract_ppg_green_passthrough():
    traces, _ = _rgb_traces(300, FPS, 72.0)
    out = extract_ppg(traces, FPS, method="green")
    assert out.shape == (300, 2)
    np.testing.assert_array_equal(out, traces[:, :, 1])


@pytest.mark.parametrize("method", ["chrom", "pos"])
def test_chrom_pos_recover_pulse_shape(method):
    traces, pure = _rgb_traces(900, FPS, 72.0)
    ppg = extract_ppg(traces, FPS, method=method)
    assert ppg.shape == (900, 2)
    assert np.isfinite(ppg).all()
    # Sign of CHROM/POS output is arbitrary — correlate on the middle section.
    c = np.corrcoef(ppg[60:-60, 0], pure[60:-60])[0, 1]
    assert abs(c) > 0.9, f"{method}: |corr|={abs(c):.3f}"


@pytest.mark.parametrize("method", ["chrom", "pos"])
def test_chrom_pos_cancel_single_channel_ramp(method):
    # A slow ramp added to R only must not destroy the pulse estimate.
    traces, pure = _rgb_traces(900, FPS, 72.0, r_ramp=40.0)
    ppg = extract_ppg(traces, FPS, method=method)
    c = np.corrcoef(ppg[60:-60, 0], pure[60:-60])[0, 1]
    assert abs(c) > 0.9, f"{method}: |corr|={abs(c):.3f}"


def test_extract_ppg_rejects_bad_input():
    with pytest.raises(ValueError):
        extract_ppg(np.zeros((10, 2, 3)), FPS, method="ica")
    with pytest.raises(ValueError):
        extract_ppg(np.zeros((10, 3)), FPS)  # missing channel axis
