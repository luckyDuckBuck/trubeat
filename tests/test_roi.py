"""Tests for roi.extract_roi_means — exact means on controlled inputs."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synth

from trubeat.roi import ROI_DEFS_P1, extract_roi_means


def _two_tone_frames(t=5, h=100, w=100):
    frames = np.zeros((t, h, w, 3), dtype=np.uint8)
    frames[:, :, : w // 2] = (255, 0, 0)  # left half red (RGB)
    frames[:, :, w // 2 :] = (0, 0, 255)  # right half blue (RGB)
    return frames


def test_exact_means_two_tone():
    t = 5
    frames = _two_tone_frames(t)
    lm = np.zeros((t, 478, 2), dtype=np.float32)
    # left box fully inside the red half, right box fully inside blue half
    lm[:, 0] = (10, 10); lm[:, 1] = (40, 10)
    lm[:, 2] = (40, 40); lm[:, 3] = (10, 40)
    lm[:, 4] = (60, 10); lm[:, 5] = (90, 10)
    lm[:, 6] = (90, 40); lm[:, 7] = (60, 40)
    valid = np.ones(t, dtype=bool)
    traces = extract_roi_means(
        frames, lm, valid, {"left_box": [0, 1, 2, 3], "right_box": [4, 5, 6, 7]}
    )
    assert traces.shape == (t, 2, 3)
    np.testing.assert_array_equal(traces[:, 0], np.full((t, 3), (255, 0, 0)))
    np.testing.assert_array_equal(traces[:, 1], np.full((t, 3), (0, 0, 255)))


def test_invalid_frames_yield_nan():
    t = 4
    frames = _two_tone_frames(t)
    lm = np.zeros((t, 478, 2), dtype=np.float32)
    valid = np.zeros(t, dtype=bool)
    traces = extract_roi_means(frames, lm, valid, {"box": [0, 1, 2]})
    assert traces.shape == (t, 1, 3)
    assert np.isnan(traces).all()


def test_degenerate_roi_stays_nan():
    t = 3
    frames = np.full((t, 50, 50, 3), 128, dtype=np.uint8)
    lm = np.zeros((t, 478, 2), dtype=np.float32)
    valid = np.ones(t, dtype=bool)
    traces = extract_roi_means(frames, lm, valid, {"point": [0, 1]})
    assert traces.shape == (t, 1, 3)
    assert np.isnan(traces).all()


def test_p1_defs_sane_and_exact_on_flat_color():
    assert set(ROI_DEFS_P1) == {"forehead", "cheek_l", "cheek_r"}
    for name, idx in ROI_DEFS_P1.items():
        assert len(idx) >= 3, name
        assert min(idx) >= 0 and max(idx) < 478, name
    # flat color: any mask's mean equals the color exactly
    t = 4
    frames = np.full((t, 240, 320, 3), (100, 150, 200), dtype=np.uint8)
    lm, valid = synth.synthetic_landmarks(t)
    traces = extract_roi_means(frames, lm, valid, ROI_DEFS_P1)
    assert traces.shape == (t, 3, 3)
    np.testing.assert_array_equal(traces, np.full((t, 3, 3), (100, 150, 200)))


def test_roi_index_out_of_range_raises():
    t = 2
    frames = np.zeros((t, 40, 40, 3), dtype=np.uint8)
    lm = np.zeros((t, 478, 2), dtype=np.float32)
    valid = np.ones(t, dtype=bool)
    with pytest.raises(ValueError, match="out of range"):
        extract_roi_means(frames, lm, valid, {"bad": [0, 1, 999]})
