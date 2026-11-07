"""Tests for scripts/build_manifest.py, scripts/cache_traces.py and the
leakage check of the AI-video scripts (train_aigen, eval_multiseg,
compare_aigen_models).

Builds synthetic dataset trees (no real data) and checks the manifests
and caches come out with the right schema, grouping, and splits.
"""

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import synth

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bm = _load("build_manifest")


def _vid(path, bpm=72.0, seed=0):
    return synth.write_pulse_video(path, bpm=bpm, duration_s=12.0, seed=seed)


def _ubfc_tree(root: Path) -> Path:
    raw = root / "ubfc_raw"
    for i, sid in enumerate(["subject01", "subject02"]):
        d = raw / sid
        d.mkdir(parents=True)
        _vid(d / "vid.avi", bpm=72.0 + i, seed=i)
        (d / "ground_truth.txt").write_text(
            "0.0 0.1 0.2\n72.0 72.5 71.8\n0.0 1.0 2.0\n"
        )
    return raw


def test_ubfc_manifest(tmp_path):
    raw = _ubfc_tree(tmp_path)
    out = tmp_path / "ubfc.csv"
    n = bm.build_ubfc_manifest(raw, out)
    assert n == 2
    with open(out) as f:
        rows = list(csv.DictReader(f))
    assert [r["video_id"] for r in rows] == ["ubfc_subject01", "ubfc_subject02"]
    assert all(r["label"] == "real" and r["split"] == "test" for r in rows)
    assert all(r["split_group"] == r["subject_id"] for r in rows)
    assert all(r["gt_path"].endswith("ground_truth.txt") for r in rows)
    assert all(float(r["fps"]) == pytest.approx(30.0) for r in rows)
    # validator accepts it (split column valid)
    import pandas as pd

    from trubeat.eval import validate_manifest

    validate_manifest(pd.read_csv(out))


def _ffpp_tree(root: Path) -> Path:
    raw = root / "ffpp_raw"
    vdir = raw / "original_sequences" / "youtube" / "c23" / "videos"
    vdir.mkdir(parents=True)
    _vid(vdir / "003.mp4", seed=0)
    _vid(vdir / "007.mp4", seed=1)
    df = raw / "manipulated_sequences" / "Deepfakes" / "c23" / "videos"
    df.mkdir(parents=True)
    _vid(df / "003_069.mp4", seed=2)  # target 003, source 069
    f2f = raw / "manipulated_sequences" / "Face2Face" / "c23" / "videos"
    f2f.mkdir(parents=True)
    _vid(f2f / "007_069.mp4", seed=3)
    sdir = raw / "splits"
    sdir.mkdir(parents=True)
    (sdir / "train.json").write_text(json.dumps(["003"]))
    (sdir / "val.json").write_text(json.dumps([]))
    (sdir / "test.json").write_text(json.dumps(["007"]))
    return raw


def test_ffpp_manifest_grouping_and_splits(tmp_path):
    raw = _ffpp_tree(tmp_path)
    out = tmp_path / "ffpp.csv"
    n = bm.build_ffpp_manifest(raw, out)
    assert n == 4
    with open(out) as f:
        rows = {r["video_id"]: r for r in csv.DictReader(f)}
    # target grouping: real 003 and its fake share split_group "003"
    assert rows["ffpp_original_003"]["split_group"] == "003"
    assert rows["ffpp_Deepfakes_003_069"]["split_group"] == "003"
    assert rows["ffpp_original_003"]["split"] == "train"
    assert rows["ffpp_Deepfakes_003_069"]["split"] == "train"
    assert rows["ffpp_original_007"]["split"] == "test"
    assert rows["ffpp_Face2Face_007_069"]["split"] == "test"
    assert rows["ffpp_Deepfakes_003_069"]["label"] == "fake"
    # no leakage possible by construction: group 003 only in train
    import pandas as pd

    from trubeat.eval import validate_manifest

    validate_manifest(pd.read_csv(out))


def test_ffpp_manifest_missing_splits_fails_loudly(tmp_path):
    raw = _ffpp_tree(tmp_path)
    for f in (raw / "splits").glob("*.json"):
        f.unlink()
    with pytest.raises(FileNotFoundError, match="split JSONs not found"):
        bm.build_ffpp_manifest(raw, tmp_path / "ffpp.csv")


def test_cache_traces_roundtrip_and_resume(tmp_path, monkeypatch, capsys):
    ct = _load("cache_traces")
    import trubeat.face as facemod

    vdir = tmp_path / "vids"
    vdir.mkdir()
    paths = [_vid(vdir / f"v{i}.mp4", seed=i) for i in range(2)]

    def fake_track(frames, fps):
        return synth.synthetic_landmarks(len(frames), h=frames.shape[1],
                                         w=frames.shape[2])

    monkeypatch.setattr(facemod, "track_landmarks", fake_track)
    import pandas as pd

    man = tmp_path / "man.csv"
    pd.DataFrame([
        {"video_id": f"v{i}", "split_group": f"g{i}", "label": "real",
         "split": "test", "path": str(p)} for i, p in enumerate(paths)
    ]).to_csv(man, index=False)

    out_dir = tmp_path / "traces"
    monkeypatch.setattr(
        "sys.argv",
        ["cache_traces", "--manifest", str(man), "--out-dir", str(out_dir)],
    )
    ct.main()
    for i in range(2):
        npz = out_dir / f"v{i}.npz"
        js = out_dir / f"v{i}.json"
        assert npz.is_file() and js.is_file()
        d = np.load(npz)
        assert d["traces"].shape[1] == 3  # 3 ROIs
        assert d["ppg"].shape == (d["traces"].shape[0], 3)
        meta = json.loads(js.read_text())
        assert meta["face_valid_frac"] == 1.0
        assert len(meta["cache_key"]) == 16

    # resume: second run skips both
    monkeypatch.setattr(
        "sys.argv",
        ["cache_traces", "--manifest", str(man), "--out-dir", str(out_dir),
         "--resume"],
    )
    ct.main()
    assert "skipped=2" in capsys.readouterr().out


# ---------------- AI-video scripts: the leakage validator runs first (docs/13 B-03)


def _ai_rows(**columns):
    import pandas as pd

    rows = pd.DataFrame({
        "path": ["/d/a.mp4", "/d/b.mp4", "/d/c.mp4", "/d/d.mp4"],
        "label": ["ai", "real", "ai", "real"],
        "generator": ["kling", "real", "kling", "real"],
        "source": ["aigvdbench", "aigvdbench", "genbuster_ood", "genbuster_id"],
        "family": ["aigvd", "real", "ood", "id"],
        "split_group": ["openvid:1", "openvid:1", "genbuster:ood:kling:1", "genbuster:id:real:2"],
        "role": ["pool", "pool", "external", "external"],
    })
    return rows.assign(**columns)


LEAKY_GROUPS = ["openvid:1", "openvid:1", "openvid:1", "genbuster:id:real:2"]


def test_aigen_role_check_refuses_identity_and_source_leakage():
    ta = _load("train_aigen")
    ta.check_roles(_ai_rows())
    with pytest.raises(ValueError, match="identity leakage"):  # one group on both sides
        ta.check_roles(_ai_rows(split_group=LEAKY_GROUPS))
    with pytest.raises(ValueError, match="source leakage"):  # one file on both sides
        ta.check_roles(_ai_rows(path=["/d/a.mp4", "/d/b.mp4", "/d/a.mp4", "/d/d.mp4"]))
    with pytest.raises(ValueError, match="unknown role"):
        ta.check_roles(_ai_rows(role=["pool", "pool", "external", "holdout"]))


def test_train_aigen_validates_before_any_clip_is_processed(tmp_path, monkeypatch):
    import sys

    ta = _load("train_aigen")
    paths = []
    for name in "abcd":
        (tmp_path / f"{name}.mp4").write_bytes(b"x")
        paths.append(str(tmp_path / f"{name}.mp4"))
    # roles come from the source (genbuster -> external); the groups leak
    _ai_rows(path=paths, split_group=LEAKY_GROUPS).drop(columns="role").to_csv(tmp_path / "aigen.csv", index=False)
    monkeypatch.setattr(ta, "normalize_all", lambda *a, **k: pytest.fail("clips processed before the check"))
    monkeypatch.setattr(sys, "argv", [
        "train_aigen.py", "--manifest", str(tmp_path / "aigen.csv"), "--ffpp-manifest", str(tmp_path / "no.csv"),
        "--model-out", str(tmp_path / "m.joblib"), "--report", str(tmp_path / "r.md"),
        "--features", str(tmp_path / "f.pkl")])
    with pytest.raises(ValueError, match="identity leakage"):
        ta.main()
    assert not (tmp_path / "m.joblib").exists()


def test_aigen_evaluation_scripts_validate_before_scoring(tmp_path, monkeypatch):
    """compare_aigen_models and eval_multiseg read a saved features table:
    the check runs before the bundle is even opened (it does not exist)."""
    import sys

    feats = tmp_path / "features.pkl"
    _ai_rows(split_group=LEAKY_GROUPS).to_pickle(feats)
    missing = str(tmp_path / "no_bundle.joblib")
    with pytest.raises(ValueError, match="identity leakage"):
        _load("compare_aigen_models").evaluate(missing, str(feats))
    ms = _load("eval_multiseg")
    monkeypatch.setattr(sys, "argv", ["eval_multiseg.py", "--features", str(feats), "--model", missing,
                                      "--out", str(tmp_path / "r.md")])
    with pytest.raises(ValueError, match="identity leakage"):
        ms.main()
