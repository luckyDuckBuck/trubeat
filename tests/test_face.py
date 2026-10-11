"""Tests for face.track_landmarks.

Video tests skip gracefully when the .task model is not downloaded;
the missing-model error path is always tested.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synth  # noqa: E402

from trubeat import face as face_mod  # noqa: E402
from trubeat.ingest import open_video  # noqa: E402

try:
    _MODEL = face_mod.resolve_model_path()
    _HAS_MODEL = _MODEL.is_file()
except RuntimeError:
    _HAS_MODEL = False

needs_model = pytest.mark.skipif(
    not _HAS_MODEL, reason="face_landmarker.task not downloaded to data/models/"
)


@needs_model
def test_no_face_video_all_invalid(tmp_path):
    # The synthetic pulse video contains no face: every frame must be invalid
    # and every landmark NaN, with the exact contract shapes/dtypes.
    p = synth.write_pulse_video(tmp_path / "noface.mp4", fps=30.0, duration_s=10.0)
    vs = open_video(str(p))
    landmarks, valid = face_mod.track_landmarks(vs.frames, vs.fps)
    assert landmarks.shape == (300, 478, 2)
    assert landmarks.dtype == np.float32
    assert valid.shape == (300,)
    assert valid.dtype == bool
    assert not valid.any()
    assert np.isnan(landmarks).all()


def test_missing_model_raises_informative(monkeypatch, tmp_path):
    monkeypatch.setenv("TRUBEAT_FACE_MODEL", str(tmp_path / "missing.task"))
    with pytest.raises(RuntimeError, match="not found"):
        face_mod.track_landmarks(np.zeros((2, 64, 64, 3), np.uint8), 30.0)


def test_rejects_non_uint8_frames():
    with pytest.raises(ValueError, match="uint8"):
        face_mod.track_landmarks(np.zeros((2, 64, 64, 3), np.float32), 30.0)
