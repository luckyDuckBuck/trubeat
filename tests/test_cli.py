"""CLI smoke tests: `trubeat hr` on a synthetic video (mocked face)."""

import json
import sys

import numpy as np

import synth
from trubeat.cli import main


def _ellipse_landmarks(n, h=240, w=320):
    return synth.synthetic_landmarks(n, h=h, w=w)


def test_hr_cli_reports_bpm(tmp_path, monkeypatch, capsys):
    vid = synth.write_pulse_video(tmp_path / "v.mp4", bpm=72.0, duration_s=12.0)

    import trubeat.face as facemod

    def fake_track(frames, fps):
        return _ellipse_landmarks(len(frames), h=frames.shape[1], w=frames.shape[2])

    monkeypatch.setattr(facemod, "track_landmarks", fake_track)
    monkeypatch.setattr(sys, "argv", ["trubeat", "hr", str(vid)])
    main()
    out = json.loads(capsys.readouterr().out)
    assert abs(out["bpm"] - 72.0) <= 4.0, out
    assert out["face_valid_frac"] == 1.0


def test_hr_cli_rejects_garbage(tmp_path, monkeypatch, capsys):
    bad = synth.write_corrupt_video(tmp_path / "bad.mp4")
    monkeypatch.setattr(sys, "argv", ["trubeat", "hr", str(bad)])
    try:
        main()
    except SystemExit as e:
        assert e.code == 1
    else:  # pragma: no cover
        raise AssertionError("expected SystemExit(1)")
    out = json.loads(capsys.readouterr().out)
    assert out["error"] == "E_UNREADABLE"
