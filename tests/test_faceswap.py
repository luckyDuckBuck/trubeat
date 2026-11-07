"""Face-swap detector (docs/12): sampling, cropping, scoring, the training
script's manifest and selection helpers. Synthetic data only; no backbone
weights needed (the embedder and face finder are stubbed where used)."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import synth

from trubeat import faceswap as fs


def test_target_indices_spread_over_the_first_max_seconds():
    idx = fs.target_indices(n_total=3000, fps=25.0, n=16, max_seconds=60.0)
    assert len(idx) == 16 and idx[0] == 0 and idx[-1] == 25 * 60 - 1
    assert idx == sorted(set(idx))
    gaps = np.diff(idx)
    assert gaps.max() - gaps.min() <= 1  # evenly spaced


def test_target_indices_short_clip_takes_every_frame():
    assert fs.target_indices(n_total=10, fps=25.0, n=16, max_seconds=120.0) == list(range(10))
    assert fs.target_indices(n_total=0, fps=25.0, n=16, max_seconds=120.0) == []


def test_sample_frames_returns_the_exact_decoded_frames(tmp_path):
    import cv2

    p = synth.write_pulse_video(tmp_path / "v.mp4", fps=30.0, duration_s=4.0)
    s = fs.sample_frames(p, n=8, max_seconds=120.0)
    assert len(s.frames) == 8 and s.n_frames == 120
    cap = cv2.VideoCapture(str(p))
    ref = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        ref.append(f[:, :, ::-1])
    cap.release()
    for frame, i in zip(s.frames, s.index):
        assert np.array_equal(frame, ref[i])  # same pixels as a full sequential decode
    assert s.time_s == sorted(s.time_s)


def test_sample_frames_unreadable_raises(tmp_path):
    from trubeat.errors import UnreadableVideo

    with pytest.raises(UnreadableVideo):
        fs.sample_frames(tmp_path / "nope.mp4", 8, 120.0)


def test_square_box_and_cut_square_pad_outside_the_frame():
    xs, ys = np.array([10.0, 50.0]), np.array([20.0, 40.0])
    cx, cy, side = fs.square_box(xs, ys, 1.5)
    assert (cx, cy, side) == (30.0, 30.0, 60.0)
    rgb = np.full((50, 50, 3), 200, np.uint8)
    crop = fs.cut_square(rgb, cx=5.0, cy=5.0, side=40.0, size=32)
    assert crop.shape == (32, 32, 3)
    assert crop[0, 0].sum() == 0 and crop[-1, -1].sum() > 0  # top-left is padding


def _bundle(dim=4):
    import train_aigen as ta

    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(+1, 0.3, (40, dim)), rng.normal(-1, 0.3, (40, dim))])
    y = np.array(["fake"] * 40 + ["real"] * 40)
    m = ta.make_model(1.0).fit(X, y)
    return {"model": m, "threshold": 0.0, "config": {"n_frames": 6, "min_faces": 4, "crop_size": 32},
            "real_reference_scores": np.sort(-np.abs(rng.normal(2, 1, 50)))}


def test_crop_scores_are_positive_for_fake():
    b = _bundle()
    s = fs.crop_scores(np.array([[1.0] * 4, [-1.0] * 4]), b)
    assert s[0] > 0 > s[1]


class _StubFinder:
    def __init__(self, found):
        self.found, self.k = found, 0

    def largest_face(self, rgb):
        self.k += 1
        h, w = rgb.shape[:2]
        return (w / 2, h / 2, min(h, w) / 2) if self.k <= self.found else None

    def close(self):
        pass


class _StubEmbedder:
    def __init__(self, value):
        self.value = value

    def embed(self, crops):
        v = np.full((len(crops), 4), self.value, np.float32)
        v[-1] *= 2  # the last crop is the most extreme
        return v


def test_analyze_scores_and_reports_the_most_suspicious_crop(tmp_path):
    p = synth.write_pulse_video(tmp_path / "v.mp4", fps=30.0, duration_s=3.0)
    r = fs.analyze(p, _bundle(), embedder=_StubEmbedder(+1.0), finder=_StubFinder(found=6))
    assert r.label == "swap" and r.score > r.threshold and r.n_faces == 6
    assert r.top_crop is not None and r.top_crop.shape == (32, 32, 3)
    assert r.top_time_s == max(r.top_time_s, 0.0) and r.real_percentile == 1.0
    assert "top_crop" not in json.loads(r.to_json())
    # the whole frame that crop came from, and the face box in it (shown in the app)
    assert r.top_frame is not None and r.top_frame.ndim == 3 and len(r.top_box) == 4
    d = json.loads(r.to_json())
    assert "top_frame" not in d and d["top_box"] == list(r.top_box)
    r = fs.analyze(p, _bundle(), embedder=_StubEmbedder(-1.0), finder=_StubFinder(found=6))
    assert r.label == "no_swap" and r.score < r.threshold


def test_analyze_without_enough_faces_does_not_decide(tmp_path):
    p = synth.write_pulse_video(tmp_path / "v.mp4", fps=30.0, duration_s=3.0)
    r = fs.analyze(p, _bundle(), embedder=_StubEmbedder(+1.0), finder=_StubFinder(found=2))
    assert r.label == "no_face" and r.score is None
    assert r.warnings and r.warnings[0].startswith("W_FEW_FACES: a face was found in 2 of 6")


def test_analyze_unreadable_is_unknown(tmp_path):
    r = fs.analyze(tmp_path / "nope.mp4", _bundle(), embedder=_StubEmbedder(1.0), finder=_StubFinder(6))
    assert r.label == "unknown" and r.warnings[0].startswith("E_UNREADABLE")


def test_config_rejects_unknown_feature():
    with pytest.raises(ValueError):
        fs.merged_config({"feature": "patch-only"})
    assert fs.config_hash({}) == fs.config_hash({"torch_threads": 1})  # hash ignores runtime knobs


# ------------------------------------------------------- training helpers ---

def test_manifest_keeps_pairs_in_one_split_and_group(tmp_path):
    import train_faceswap as tf

    root = tmp_path / "ffpp"
    (root / "splits").mkdir(parents=True)
    for split, pairs in (("train", [["000", "003"]]), ("val", [["010", "011"]]), ("test", [["020", "021"]])):
        (root / "splits" / f"{split}.json").write_text(json.dumps(pairs))
    for rel in ("original_sequences/youtube/c23/videos/000.mp4", "original_sequences/youtube/c23/videos/003.mp4",
                "manipulated_sequences/Deepfakes/c23/videos/000_003.mp4",
                "manipulated_sequences/FaceShifter/c23/videos/003_000.mp4",
                "original_sequences/youtube/c23/videos/020.mp4",
                "original_sequences/actors/c23/videos/01__walking.mp4",
                "manipulated_sequences/DeepFakeDetection/c23/videos/01_02__walking__AB12.mp4"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(b"x")
    df = tf.build_manifest(root)
    assert len(df) == 7  # only files that exist
    tr = df[df.split == "train"]
    assert set(tr.family) == {"original", "Deepfakes", "FaceShifter"} and tr.group.nunique() == 1
    dfd = df[df.dataset == "dfd"]
    assert set(dfd.split) == {"external"} and set(dfd.group) == {"dfd:01"}


def test_lomo_auc_scores_each_held_out_family():
    import train_faceswap as tf

    rng = np.random.default_rng(1)
    rows, X = [], []
    for g in range(10):
        for fam, mu in (("original", -1.0), ("Deepfakes", 1.0), ("FaceSwap", 1.0)):
            for _ in range(3):
                rows.append({"family": fam, "label": "real" if fam == "original" else "fake", "group": f"g{g}"})
                X.append(rng.normal(mu, 0.4, 6))
    per = tf.lomo_auc(pd.DataFrame(rows), np.array(X), C=1.0)
    assert set(per) == {"Deepfakes", "FaceSwap"} and min(per.values()) > 0.95


def test_gates_need_every_swap_family():
    import train_faceswap as tf

    m = {"ffpp_Deepfakes_auc": 0.9, "ffpp_FaceSwap_auc": 0.88, "ffpp_FaceShifter_auc": 0.80,
         "ffpp_fpr": (4, 140), "dfd_DFD_auc": 0.72, "dfd_fpr": (10, 150)}
    g = {name.split(" on ")[-1] if " on " in name else name[:2]: ok for name, ok, _ in tf.gates(m)}
    assert g["Deepfakes"] and g["FaceSwap"] and g["FaceShifter"] is False
    assert g["2."] is True and g["3."] is True and g["4."] is None


def test_cnn_bundle_scores_crops_and_checks_its_weights(tmp_path, monkeypatch):
    """Stage 2: a CNN bundle scores each crop; weights are found locally
    and fetch_cnn_weights verifies the SHA-256 (file:// stands in for the
    release URL)."""
    pytest.importorskip("timm")
    pytest.importorskip("safetensors")
    import hashlib

    import timm
    from safetensors.torch import save_file

    m = timm.create_model("efficientnet_b0", pretrained=False, num_classes=1)
    src = tmp_path / "src" / "w.safetensors"
    src.parent.mkdir()
    save_file({k: v.contiguous() for k, v in m.state_dict().items()}, str(src))
    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    bundle = {"kind": "cnn", "threshold": 0.0, "real_reference_scores": np.zeros(3),
              "config": {"n_frames": 6, "min_faces": 4, "crop_size": 64},
              "cnn": {"arch": "efficientnet_b0", "file": "w.safetensors", "sha256": sha,
                      "url": src.as_uri(), "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}}
    monkeypatch.setattr(fs.Path, "home", staticmethod(lambda: tmp_path / "home"))
    monkeypatch.setattr(fs, "model_dirs", lambda: [tmp_path / "home" / ".cache" / "trubeat"])
    assert fs.cnn_weights_path(bundle) is None
    got = fs.fetch_cnn_weights(bundle)
    assert got.is_file() and fs.cnn_weights_path(bundle) == got
    p = synth.write_pulse_video(tmp_path / "v.mp4", fps=30.0, duration_s=3.0)
    r = fs.analyze(p, bundle, finder=_StubFinder(found=6))
    assert r.label in ("swap", "no_swap") and r.n_faces == 6 and np.isfinite(r.score)


def test_cnn_weights_with_a_wrong_hash_are_refused(tmp_path, monkeypatch):
    from trubeat.errors import ModelMissing

    src = tmp_path / "w.safetensors"
    src.write_bytes(b"not the weights")
    monkeypatch.setattr(fs.Path, "home", staticmethod(lambda: tmp_path / "home"))
    monkeypatch.setattr(fs, "model_dirs", lambda: [tmp_path / "home" / ".cache" / "trubeat"])
    bundle = {"cnn": {"file": "w.safetensors", "url": src.as_uri(), "sha256": "0" * 64}}
    with pytest.raises(ModelMissing):
        fs.fetch_cnn_weights(bundle)
    assert fs.cnn_weights_path(bundle) is None


def test_stale_cnn_weights_are_never_paired_with_a_new_bundle(tmp_path, monkeypatch):
    """Review 2026-11-06: a copy from an earlier training run (same file
    name, other weights) is refused; fetch-models replaces it."""
    import hashlib

    from trubeat.errors import ModelMissing

    good = tmp_path / "release" / "w.safetensors"
    good.parent.mkdir()
    good.write_bytes(b"the new weights")
    cache = tmp_path / "home" / ".cache" / "trubeat"
    cache.mkdir(parents=True)
    (cache / "w.safetensors").write_bytes(b"the old weights")
    monkeypatch.setattr(fs.Path, "home", staticmethod(lambda: tmp_path / "home"))
    monkeypatch.setattr(fs, "model_dirs", lambda: [cache])
    bundle = {"config": {}, "cnn": {"arch": "x", "file": "w.safetensors", "url": good.as_uri(),
                                    "sha256": hashlib.sha256(good.read_bytes()).hexdigest(),
                                    "mean": [0.5] * 3, "std": [0.5] * 3}}
    assert fs.cnn_weights_path(bundle) is None
    with pytest.raises(ModelMissing, match="do not match"):
        fs.CNNScorer.from_bundle(bundle)
    assert fs.fetch_cnn_weights(bundle).read_bytes() == b"the new weights"
    assert fs.CNNScorer.from_bundle(bundle).weights == cache / "w.safetensors"


def test_cnn_weights_next_to_the_bundle_are_found(tmp_path, monkeypatch):
    import hashlib

    (tmp_path / "w.safetensors").write_bytes(b"weights")
    monkeypatch.setattr(fs, "model_dirs", lambda: [])  # noqa: PIE807
    bundle = {"_path": str(tmp_path / "faceswap_cnn.joblib"),
              "cnn": {"file": "w.safetensors", "sha256": hashlib.sha256(b"weights").hexdigest()}}
    assert fs.cnn_weights_path(bundle) == (tmp_path / "w.safetensors").resolve()


def test_cnn_weights_are_never_paired_with_a_bundle_without_a_hash(tmp_path, monkeypatch):
    # review 2026-11-07: an unverifiable bundle must not pick up a same-named stale copy
    (tmp_path / "w.safetensors").write_bytes(b"old weights")
    monkeypatch.setattr(fs, "model_dirs", lambda: [])  # noqa: PIE807
    bundle = {"_path": str(tmp_path / "faceswap_cnn.joblib"), "cnn": {"file": "w.safetensors"}}
    assert fs.cnn_weights_path(bundle) is None


def test_clips_under_two_seconds_are_not_judged(tmp_path):
    p = synth.write_pulse_video(tmp_path / "v.mp4", fps=30.0, duration_s=1.0)
    r = fs.analyze(p, _bundle(), embedder=_StubEmbedder(+1.0), finder=_StubFinder(found=30))
    assert r.label == "unknown" and r.score is None and r.warnings[0].startswith("E_SHORT_VIDEO")


def test_manifest_check_refuses_identity_leakage():
    import train_faceswap as tf

    ok = pd.DataFrame({"split": ["train", "train", "test", "external"],
                       "group": ["ffpp:000_003", "ffpp:000_003", "ffpp:020_021", "dfd:01"]})
    tf.check_manifest(ok)
    with pytest.raises(ValueError, match="leakage"):
        tf.check_manifest(ok.assign(split=["train", "test", "test", "external"]))
    with pytest.raises(ValueError, match="external"):
        tf.check_manifest(ok.assign(group=["ffpp:000_003", "ffpp:000_003", "ffpp:020_021", "ffpp:020_021"]))


def _ship(tmp_path, gates_ok, g4_passed, kind="probe", same_file=True):
    import hashlib
    import types

    import joblib
    import train_faceswap as tf

    names = ["1. a on Deepfakes", "1. a on FaceSwap", "1. a on FaceShifter", "2. reals", "3. DFD", "4. others"]
    b = {"threshold": 0.0, "config": {}, "gates": [(n, ok, "v") for n, ok in zip(names, [*gates_ok, None])]}
    src = tmp_path / "faceswap.joblib"
    joblib.dump(b, src)
    g4 = tmp_path / "g4.json"
    g4.write_text(json.dumps({"bundle_sha256": hashlib.sha256(src.read_bytes()).hexdigest() if same_file else "x",
                              "k": 3, "n": 300, "passed": g4_passed}))
    out = tmp_path / "installed" / "faceswap.joblib"
    tf.cmd_ship(types.SimpleNamespace(bundle=str(src), gate4=str(g4), out=str(out), url=""))
    return joblib.load(out)


def test_ship_applies_the_docs12_rules(tmp_path):
    b = _ship(tmp_path, [True, True, True, True, True], True)
    assert b["mode"] == "decisive" and b["gates"][-1][1] is True and b["gates"][-1][2] == "3/300 flagged"
    assert _ship(tmp_path, [True, True, True, True, False], True)["mode"] == "evidence"   # gate 3 fails
    assert _ship(tmp_path, [True, True, True, True, True], False)["mode"] == "evidence"   # gate 4 fails
    with pytest.raises(SystemExit, match="not shipped"):
        _ship(tmp_path, [True, False, True, True, True], True)                           # gate 1 fails
    with pytest.raises(SystemExit, match="not shipped"):
        _ship(tmp_path, [True, True, True, False, True], True)                           # gate 2 fails
    with pytest.raises(SystemExit, match="different bundle"):
        _ship(tmp_path, [True, True, True, True, True], True, same_file=False)


def test_failed_videos_are_extracted_again(tmp_path):
    """Colab run 1 (2026-11-07): every video failed; a rerun must redo them,
    not skip them as done."""
    import train_faceswap as tf

    for b in ("b1", "b2"):
        (tmp_path / b).mkdir()
        np.savez(tmp_path / b / "ok.npz", cls=np.zeros((4, 2), np.float16), error="")
        np.savez(tmp_path / b / "bad.npz", cls=np.zeros((0, 1), np.float16), error="ValueError: boom")
    assert tf.extracted(tmp_path, ["b1", "b2"], None, "ok")
    assert not tf.extracted(tmp_path, ["b1", "b2"], None, "bad")
    assert not tf.extracted(tmp_path, ["b1", "b2"], None, "missing")
    (tmp_path / "crops" / "ok").mkdir(parents=True)
    assert not tf.extracted(tmp_path, ["b1", "b2"], tmp_path / "crops", "ok")  # crops not saved yet
    (tmp_path / "crops" / "ok" / "done").write_text("4")
    assert tf.extracted(tmp_path, ["b1", "b2"], tmp_path / "crops", "ok")


def test_training_script_replaces_a_notebook_plot_backend():
    """Colab exports MPLBACKEND=module://matplotlib_inline...; MediaPipe
    imports pyplot, which then fails in a venv without that module."""
    import os
    import subprocess

    repo = Path(__file__).resolve().parents[1]
    code = ("import sys, os; sys.path[:0] = ['scripts', 'src']; import train_faceswap; "
            "print(os.environ['MPLBACKEND'])")
    env = dict(os.environ, MPLBACKEND="module://matplotlib_inline.backend_inline")
    out = subprocess.run([sys.executable, "-c", code], cwd=repo, env=env, capture_output=True, text=True, check=False)
    assert out.returncode == 0 and out.stdout.strip() == "Agg", out.stderr[-500:]
