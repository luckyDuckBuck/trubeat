"""Tests for ingest.open_video — all synthetic, no real data."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synth

from trubeat.errors import LowFps, ShortVideo, UnreadableVideo
from trubeat.ingest import VideoStream, open_video


def test_open_valid_video(tmp_path):
    p = synth.write_pulse_video(tmp_path / "v.mp4", fps=30.0, duration_s=10.0)
    vs = open_video(str(p))
    assert isinstance(vs, VideoStream)
    assert vs.frames.shape == (300, 240, 320, 3)
    assert vs.frames.dtype == np.uint8
    assert vs.fps == pytest.approx(30.0, abs=0.5)
    assert vs.n_frames == 300
    assert vs.width == 320 and vs.height == 240
    assert vs.duration_s == pytest.approx(10.0, abs=0.2)


def test_frames_are_rgb_not_bgr(tmp_path):
    # synth writes BGR with R=base+10, B=base-20; after BGR->RGB,
    # channel 0 (R) must be brighter than channel 2 (B).
    p = synth.write_pulse_video(tmp_path / "v.mp4")
    vs = open_video(str(p))
    assert vs.frames[..., 0].mean() > vs.frames[..., 2].mean() + 10.0


def test_corrupt_bytes_raise_unreadable(tmp_path):
    p = synth.write_corrupt_video(tmp_path / "bad.mp4")
    with pytest.raises(UnreadableVideo):
        open_video(str(p))


def test_missing_file_raises_unreadable(tmp_path):
    with pytest.raises(UnreadableVideo):
        open_video(str(tmp_path / "does_not_exist.mp4"))


def test_short_video_raises(tmp_path):
    p = synth.write_pulse_video(tmp_path / "short.mp4", fps=30.0, duration_s=3.0)
    with pytest.raises(ShortVideo):
        open_video(str(p))


def test_low_fps_raises(tmp_path):
    p = synth.write_pulse_video(tmp_path / "slow.mp4", fps=10.0, duration_s=10.0)
    with pytest.raises(LowFps):
        open_video(str(p))
