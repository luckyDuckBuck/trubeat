"""Tests for trubeat.aigen: the whole-frame AI-generated video detector.

Property tests first (AGENTS.md §5): the trajectory geometry is checked on
embeddings with known shape; the sampler on synthetic videos with known
duration and frame rate. The backbone test runs only when timm and the
DINOv2 weights are already cached (no network in tests).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synth

from trubeat import aigen
from trubeat.errors import ShortVideo, UnreadableVideo

# ----------------------------------------------------------- geometry -------


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def test_straight_path_has_near_zero_curvature():
    """Points evenly spaced on a straight line: every step points the same
    way, so the angle between consecutive steps is ~0."""
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=64), rng.normal(size=64)
    z = np.stack([a + t * (b - a) for t in np.linspace(0, 1, 16)])
    f = aigen.trajectory_features(z)
    assert f["curv_mean"] < 1e-6
    assert f["step_std"] < 1e-9  # constant step size
    assert f["acc_mean"] < 1e-9  # no second-order change


def test_random_path_is_curved_and_volatile():
    """Independent random unit vectors in high dimension: consecutive steps
    are close to orthogonal-ish (mean angle well above a smooth path's)."""
    rng = np.random.default_rng(1)
    z = _unit(rng.normal(size=(16, 384)))
    f = aigen.trajectory_features(z)
    assert 1.0 < f["curv_mean"] < 2.5  # radians
    assert f["acc_mean"] > 0.5


def test_static_segment_is_zero_not_nan():
    z = np.tile(_unit(np.ones((1, 32))), (16, 1))
    f = aigen.trajectory_features(z)
    assert all(np.isfinite(v) for v in f.values())
    assert f["step_mean"] == 0.0 and f["curv_mean"] == 0.0


def test_geometry_columns_are_complete_and_ordered():
    z = _unit(np.random.default_rng(2).normal(size=(16, 8)))
    f = aigen.trajectory_features(z)
    assert list(f) == aigen.GEOMETRY_COLUMNS


def test_trajectory_features_rejects_too_few_frames():
    with pytest.raises(ValueError):
        aigen.trajectory_features(np.zeros((2, 8)))


def test_feature_vector_sets():
    geom = {c: float(i) for i, c in enumerate(aigen.GEOMETRY_COLUMNS)}
    emb = np.arange(5, dtype=np.float32)
    x, cols = aigen.feature_vector(geom, emb, "geometry")
    assert len(x) == len(cols) == len(aigen.GEOMETRY_COLUMNS)
    x, cols = aigen.feature_vector(geom, emb, "embedding")
    assert cols == [f"emb_{i}" for i in range(5)]
    x, cols = aigen.feature_vector(geom, emb, "combined")
    assert len(x) == len(aigen.GEOMETRY_COLUMNS) + 5
    with pytest.raises(ValueError):
        aigen.feature_vector(geom, emb, "nope")


# ----------------------------------------------------------- sampling -------


def test_target_times_fit_inside_the_clip():
    cfg = aigen.merged_config({})
    span = (cfg["n_frames"] - 1) / cfg["sample_fps"]
    for duration in (2.0, 5.0, 8.0, 30.0):
        segs = aigen._target_times(duration, cfg)
        assert 1 <= len(segs) <= cfg["max_segments"]
        for seg in segs:
            assert len(seg) == cfg["n_frames"]
            assert seg[0] >= 0.0 and seg[-1] <= duration + 1e-9
            assert np.allclose(np.diff(seg), 1.0 / cfg["sample_fps"])
            assert seg[-1] - seg[0] == pytest.approx(span)
    assert aigen._target_times(1.0, cfg) == []


@pytest.mark.parametrize("fps", [24.0, 30.0])
def test_sample_segments_shapes_and_duration(tmp_path, fps):
    p = synth.write_pulse_video(tmp_path / "v.mp4", fps=fps, duration_s=6.0)
    frames, meta = aigen.sample_segments(p, {"img_size": 64})
    s, n, h, w, c = frames.shape
    assert (n, h, w, c) == (16, 64, 64, 3)
    assert 1 <= s <= 3
    assert frames.dtype == np.uint8
    assert meta["duration_s"] == pytest.approx(6.0, abs=0.15)
    assert meta["fps_ts"] == pytest.approx(fps, rel=0.05)


def test_sample_segments_short_clip_raises(tmp_path):
    p = synth.write_pulse_video(tmp_path / "short.mp4", fps=30.0, duration_s=1.0)
    with pytest.raises(ShortVideo):
        aigen.sample_segments(p, {"img_size": 32})


def test_sample_segments_missing_or_corrupt(tmp_path):
    with pytest.raises(UnreadableVideo):
        aigen.sample_segments(tmp_path / "nope.mp4")
    bad = synth.write_corrupt_video(tmp_path / "bad.mp4")
    with pytest.raises(UnreadableVideo):
        aigen.sample_segments(bad)


def test_prep_center_crops_any_aspect():
    tall = np.zeros((400, 100, 3), dtype=np.uint8)
    wide = np.zeros((100, 400, 3), dtype=np.uint8)
    assert aigen._prep(tall, 32).shape == (32, 32, 3)
    assert aigen._prep(wide, 32).shape == (32, 32, 3)


# ------------------------------------------------------ analyze (no net) ----


class _StubEmbedder:
    """Deterministic stand-in for the backbone: embeds mean colour + noise."""

    def embed(self, frames):
        rng = np.random.default_rng(int(frames.sum()) % (2**32))
        out = rng.normal(size=(frames.shape[0], 16)).astype(np.float32)
        return out / np.linalg.norm(out, axis=1, keepdims=True)


def _tiny_bundle(feature_set="geometry"):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    d = len(aigen.GEOMETRY_COLUMNS)
    X = np.vstack([rng.normal(0, 1, (20, d)), rng.normal(1, 1, (20, d))])
    y = np.array(["real"] * 20 + ["ai"] * 20)
    model = make_pipeline(StandardScaler(), LogisticRegression()).fit(X, y)
    return {"model": model, "feature_set": feature_set, "threshold": 0.0,
            "config": {"img_size": 32, "cache_dir": None}}


def test_analyze_returns_a_scored_result(tmp_path):
    p = synth.write_pulse_video(tmp_path / "v.mp4", fps=30.0, duration_s=5.0)
    res = aigen.analyze(p, _tiny_bundle(), embedder=_StubEmbedder())
    assert res.label in ("ai", "real")
    assert 0.0 <= res.p_ai <= 1.0
    assert res.n_segments >= 1
    # positive score <=> p_ai above one half (score oriented so + = AI)
    assert (res.score > 0) == (res.p_ai > 0.5)


def test_analyze_short_clip_is_unknown_not_a_crash(tmp_path):
    p = synth.write_pulse_video(tmp_path / "s.mp4", fps=30.0, duration_s=1.0)
    res = aigen.analyze(p, _tiny_bundle(), embedder=_StubEmbedder())
    assert res.label == "unknown" and res.p_ai is None
    assert any("E_SHORT_VIDEO" in w for w in res.warnings)


# ----------------------------------------------- real backbone (if cached) --


def _weights_cached() -> bool:
    try:
        import timm  # noqa: F401
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    hit = try_to_load_from_cache(
        "timm/" + aigen.DEFAULT_CONFIG["backbone"], "model.safetensors"
    )
    return isinstance(hit, str)


@pytest.mark.skipif(not _weights_cached(), reason="DINOv2 weights not cached locally")
def test_embedder_unit_norm_embeddings(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")  # tests never touch the network
    emb = aigen.Embedder({"torch_threads": 2}).embed(
        np.random.default_rng(0).integers(0, 255, (3, 224, 224, 3), dtype=np.uint8)
    )
    assert emb.shape == (3, 384)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-4)
