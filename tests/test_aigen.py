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


def test_target_times_end_at_the_last_frame_on_the_sampling_grid():
    cfg = aigen.merged_config({})
    step = 1.0 / cfg["sample_fps"]
    span = (cfg["n_frames"] - 1) * step
    for duration, fps in ((2.0, 8.0), (5.0, 24.0), (8.0, 30.0), (30.0, 25.0)):
        last = duration - 1.0 / fps  # timestamp of the last decoded frame
        segs = aigen._target_times(duration, cfg, last_frame_s=last)
        assert 1 <= len(segs) <= cfg["max_segments"]
        for seg in segs:
            assert len(seg) == cfg["n_frames"]
            assert seg[0] >= 0.0 and seg[-1] <= last + 1e-6  # never past the last frame
            assert np.allclose(np.diff(seg), step)
            assert seg[-1] - seg[0] == pytest.approx(span)
            assert seg[0] / step == pytest.approx(round(seg[0] / step))  # on the grid
    assert aigen._target_times(1.0, cfg) == []


def test_no_segment_repeats_a_frame(tmp_path):
    """Review finding #3: the last of several segments repeated its final
    frame (targets ran one frame period past the last frame). Encode the
    frame index in brightness and check every sampled frame is distinct."""
    import cv2

    path = tmp_path / "counter.mp4"
    fps, n = 30.0, 300  # 10 s -> 3 segments
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 64))
    for i in range(n):
        vw.write(np.full((64, 64, 3), i % 250, dtype=np.uint8))
    vw.release()
    frames, meta = aigen.sample_segments(path, {"img_size": 32, "max_segments": 3})
    assert meta["n_segments"] == 3
    for seg in frames:
        levels = seg.reshape(seg.shape[0], -1).mean(axis=1)
        assert np.all(np.abs(np.diff(levels)) > 1.0), levels


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
    s = model.decision_function(X[:20])
    ref = np.sort(s if list(model.classes_)[1] == "ai" else -s)
    return {"model": model, "feature_set": feature_set,
            "threshold": float(np.quantile(ref, 0.95)), "real_reference_scores": ref,
            # the synthetic clip is one flat colour: the title-card guard
            # (docs/12 item A) would rightly skip it, so these tests turn it off
            "config": {"img_size": 32, "cache_dir": None, "max_segments": 1, "graphics_guard": False}}


def test_analyze_returns_a_scored_result(tmp_path):
    p = synth.write_pulse_video(tmp_path / "v.mp4", fps=30.0, duration_s=5.0)
    res = aigen.analyze(p, _tiny_bundle(), embedder=_StubEmbedder())
    assert res.label in ("ai", "real")
    assert 0.0 <= res.real_percentile <= 1.0
    assert res.n_segments == 1  # the bundle's segment count, as evaluated
    # the percentile shown can never contradict the label (review #5)
    if res.label == "ai":
        assert res.real_percentile >= 0.9
    else:
        assert res.real_percentile <= 0.96
    # coverage is stated, not mislabelled as a short clip (round-2 audit)
    assert not any(w.startswith("W_SHORT_CLIP") for w in res.warnings)
    assert any(w.startswith("W_PARTIAL_COVERAGE: scored on the first 2 s segment of a 5.0 s")
               for w in res.warnings), res.warnings


def test_analyze_states_short_clip_against_the_bundles_segment_count(tmp_path):
    p = synth.write_pulse_video(tmp_path / "v.mp4", fps=30.0, duration_s=2.5)
    bundle = _tiny_bundle()
    bundle["config"]["max_segments"] = 3
    res = aigen.analyze(p, bundle, embedder=_StubEmbedder())
    assert res.n_segments == 1
    assert any(w.startswith("W_SHORT_CLIP: 1 of the 3 segments") for w in res.warnings)
    assert not any(w.startswith("W_PARTIAL_COVERAGE") for w in res.warnings)


def test_analyze_short_clip_is_unknown_not_a_crash(tmp_path):
    p = synth.write_pulse_video(tmp_path / "s.mp4", fps=30.0, duration_s=1.0)
    res = aigen.analyze(p, _tiny_bundle(), embedder=_StubEmbedder())
    assert res.label == "unknown" and res.real_percentile is None
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


def test_missing_weights_raise_model_missing_not_a_download(monkeypatch):
    """The analysis path never downloads (AGENTS.md §7): with no cached
    weights it raises E_MODEL_MISSING naming the one command that fetches."""
    from trubeat.errors import ModelMissing

    monkeypatch.setattr(aigen, "cached_weights", lambda backbone: None)
    with pytest.raises(ModelMissing) as exc:
        aigen.Embedder()._load()
    assert exc.value.code == "E_MODEL_MISSING"
    assert "trubeat fetch-models" in str(exc.value)


def test_missing_aigen_extra_raises_model_missing_with_install_steps(monkeypatch):
    """Without the optional [aigen] extra (as in CI), the app must say how to
    install the detector, not report a raw ModuleNotFoundError as a failure."""
    import sys

    from trubeat.errors import ModelMissing

    monkeypatch.setitem(sys.modules, "timm", None)  # import timm -> ImportError
    with pytest.raises(ModelMissing) as exc:
        aigen.Embedder()._load()
    assert exc.value.code == "E_MODEL_MISSING"
    assert '.[aigen]' in str(exc.value) and "trubeat fetch-models" in str(exc.value)


def test_find_bundle_returns_none_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(aigen, "BUNDLE_SEARCH_PATHS", [tmp_path / "nope.joblib"])
    assert aigen.find_bundle() is None


# ------------------------------------------------- title-card guard (docs/12 A)

def _write(path, frames, fps=8.0):
    import cv2

    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()
    return path


def _card(seed=0):
    import cv2

    f = np.zeros((240, 320, 3), np.uint8)
    cv2.putText(f, "This footage is not real", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return f


def _texture(seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)


def test_is_graphic_tells_title_cards_from_textured_video():
    cfg = aigen.merged_config({"img_size": 64})
    card = np.stack([_card()] * 4)
    tex = np.stack([_texture(i) for i in range(4)])
    assert aigen.is_graphic(card, cfg)
    assert not aigen.is_graphic(tex, cfg)


def test_guard_skips_an_opening_title_card(tmp_path):
    frames = [_card()] * 24 + [_texture(i) for i in range(40)]  # 3 s card, then 5 s of texture
    p = _write(tmp_path / "v.mp4", frames)
    cfg = aigen.merged_config({"img_size": 64, "max_segments": 1, "cache_dir": None})
    segs, meta = aigen.guarded_segments(p, cfg)
    assert meta["n_graphic_skipped"] == 1 and len(segs) == 1
    assert meta["segment_starts_s"][0] > 1.8  # moved past the card
    assert not aigen.is_graphic(segs[0], cfg)


def test_guard_reports_graphics_only(tmp_path):
    from trubeat.errors import GraphicsOnly

    p = _write(tmp_path / "v.mp4", [_card()] * 40)
    with pytest.raises(GraphicsOnly):
        aigen.guarded_segments(p, aigen.merged_config({"img_size": 64, "max_segments": 1, "cache_dir": None}))


def test_guard_leaves_natural_video_alone(tmp_path):
    p = _write(tmp_path / "v.mp4", [_texture(i) for i in range(40)])
    _, meta = aigen.guarded_segments(p, aigen.merged_config({"img_size": 64, "max_segments": 1, "cache_dir": None}))
    assert meta["n_graphic_skipped"] == 0 and meta["segment_starts_s"] == [0.0]


def test_flat_but_moving_scene_is_not_a_title_card():
    """docs/12 item A revision: smooth AI scenes are flat too, but they move."""
    cfg = aigen.merged_config({"img_size": 64})
    frames = []
    for k in range(8):  # a dark flat field with a bright bar sliding across
        f = np.full((240, 320, 3), 20, np.uint8)
        f[:, 40 * k:40 * k + 40] = 230
        frames.append(f)
    seg = np.stack(frames)
    assert aigen.flat_fraction(seg[0], 4.0) > 0.7  # flat ...
    assert aigen.segment_motion(seg) > 0.5          # ... but moving
    assert not aigen.is_graphic(seg, cfg)
