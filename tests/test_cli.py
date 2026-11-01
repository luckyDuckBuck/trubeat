"""CLI smoke tests: `trubeat hr` on a synthetic video (mocked face)."""

import json
import sys

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


def test_train_cli_bundle_matches_the_detector_contract(tmp_path, monkeypatch, capsys):
    """docs/11 A6: `trubeat train` fits on the detector columns only (no
    positional window columns) and saves feature_columns/method/window_s,
    so its bundle is interchangeable with the packaged G2 detector."""
    import joblib
    import numpy as np
    import pandas as pd

    from trubeat import cli, detector, features

    rng = np.random.default_rng(0)

    def fake_feat(path, config):
        shift = 2.0 if "fake" in path else 0.0
        return pd.DataFrame(
            rng.normal(shift, 1.0, (4, len(features.FEATURE_COLUMNS))),
            columns=features.FEATURE_COLUMNS,
        )

    monkeypatch.setattr(cli, "_feat_for_video", fake_feat)
    rows = [
        {"path": f"{lab}_{i}.mp4", "label": lab, "split": "train", "split_group": f"g{i}"}
        for i in range(6)
        for lab in ("real", "fake")
    ]
    manifest = tmp_path / "m.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    cfg = tmp_path / "c.yaml"
    cfg.write_text("method: green\nwindow_s: 10.0\n")
    out = tmp_path / "bundle.joblib"
    monkeypatch.setattr(
        "sys.argv",
        ["trubeat", "train", "--manifest", str(manifest), "--config", str(cfg),
         "--output", str(out)],
    )
    cli.main()
    bundle = joblib.load(out)
    assert bundle["feature_columns"] == list(detector.DETECTOR_COLUMNS)
    assert list(bundle["model"].feature_names_in_) == list(detector.DETECTOR_COLUMNS)
    assert "win_start_s" not in bundle["feature_columns"]
    assert bundle["method"] == "green" and bundle["window_s"] == 10.0
