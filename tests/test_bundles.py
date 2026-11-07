"""The shipped model bundles, loaded and scored (docs/13 D-01, D-05, D-12).

CI-safe first: random unit vectors as wide as each bundle's backbone stand in
for the embeddings, so the shipped models, thresholds and reference scores
are exercised with no weights and no network. The real-backbone tests at the
end run wherever the weights are cached; CI's full job fetches them and does
not allow these tests to skip (tests/conftest.py).
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synth

from trubeat import aigen, bundles
from trubeat import faceswap as fs

MODELS = Path(aigen.__file__).parent / "models"
AIGEN = MODELS / "aigen.joblib"
FACESWAP = MODELS / "faceswap.joblib"
# feature width of each backbone the shipped bundles name (CLS token)
WIDTH = {"vit_small_patch14_dinov2.lvd142m": 384, "vit_base_patch16_clip_224.openai": 768}


def _unit(rng, shape):
    v = rng.normal(size=shape)
    return (v / np.linalg.norm(v, axis=-1, keepdims=True)).astype(np.float32)


class _UnitEmbedder:
    """Stands in for a backbone: one random unit vector per frame or crop."""

    def __init__(self, dim, seed=0):
        self.dim, self.rng = dim, np.random.default_rng(seed)

    def embed(self, images):
        return _unit(self.rng, (len(images), self.dim))


class _CentreFinder:
    """Stands in for the face finder: a centred box in every frame."""

    def largest_face(self, rgb):
        h, w = rgb.shape[:2]
        return (w / 2, h / 2, min(h, w) / 2)

    def close(self):
        pass


def _textured_clip(path, seconds=5.0, fps=8.0):
    """Moving noise: never a title card, so the AI-video guard keeps it."""
    import cv2

    rng = np.random.default_rng(3)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 240))
    for _ in range(int(seconds * fps)):
        vw.write(rng.integers(0, 256, (240, 320, 3), dtype=np.uint8))
    vw.release()
    return path


@pytest.fixture(scope="module")
def ai_bundle():
    return aigen.load_bundle(AIGEN)


@pytest.fixture(scope="module")
def fs_bundle():
    return fs.load_bundle(FACESWAP)


def _check_threshold_and_references(bundle):
    """The threshold is the 1 - target_fpr quantile of the bundle's own
    reference real scores, so the percentile the app shows and the label
    can never disagree at the operating point."""
    thr = float(bundle["threshold"])
    ref = np.asarray(bundle["real_reference_scores"], dtype=np.float64)
    target_fpr = float(bundle["config"]["target_fpr"])
    assert np.isfinite(thr) and len(ref) >= 100 and np.all(np.isfinite(ref))
    assert np.all(np.diff(ref) >= 0)  # sorted, as analyze's percentile assumes
    assert ref[0] < thr < ref[-1]
    assert abs(np.mean(ref < thr) - (1.0 - target_fpr)) <= 1.0 / len(ref)
    return thr, ref


def _check_scores(scores, thr, ref):
    assert np.all(np.isfinite(scores))
    assert np.ptp(scores) > 0  # different inputs, different scores
    # decision values on the threshold's scale (not probabilities, not
    # another unit): random inputs stay within 20 reference spreads
    assert np.all(np.abs(scores - np.median(ref)) < 20 * np.std(ref))
    pct = np.array([np.mean(ref < s) for s in scores])
    assert np.all((pct >= 0.0) & (pct <= 1.0))
    p_thr = np.mean(ref < thr)  # ~ 1 - target_fpr
    assert np.all(pct[scores >= thr] >= p_thr) and np.all(pct[scores < thr] <= p_thr)


# --------------------------------------------- AI-video bundle (D-01) ---


def test_shipped_bundles_load_with_the_installed_sklearn(ai_bundle, fs_bundle):
    import sklearn

    same = bundles._major_minor(sklearn.__version__) == (1, 9)  # the shipped pickles' version
    for b in (ai_bundle, fs_bundle):
        if same:
            assert b[bundles.WARNINGS_KEY] == []
        else:
            assert [w[:17] for w in b[bundles.WARNINGS_KEY]] == ["W_SKLEARN_VERSION"]


def test_aigen_bundle_scores_unit_vectors_of_its_width(ai_bundle):
    cfg = aigen.merged_config(ai_bundle["config"])
    model, cols = ai_bundle["model"], ai_bundle["feature_columns"]
    assert sorted(model.classes_) == ["ai", "real"]
    assert model.n_features_in_ == len(cols)
    dim = sum(c.startswith("emb_") for c in cols)
    assert dim == WIDTH[cfg["backbone"]]
    thr, ref = _check_threshold_and_references(ai_bundle)
    rng = np.random.default_rng(0)
    seg = _unit(rng, (24, int(cfg["n_frames"]), dim))  # 24 segments of n_frames frames
    scores = aigen.segment_scores(seg, ai_bundle)
    assert scores.shape == (24,)
    _check_scores(scores, thr, ref)
    assert np.array_equal(scores, aigen.segment_scores(seg, ai_bundle))  # deterministic


def test_aigen_bundle_analyses_a_clip_end_to_end(tmp_path, ai_bundle):
    """The whole analysis path with the shipped bundle (re-encode when ffmpeg
    exists, title-card guard, segment scores, percentile), backbone stubbed."""
    p = _textured_clip(tmp_path / "clip.mp4")
    dim = sum(c.startswith("emb_") for c in ai_bundle["feature_columns"])
    res = aigen.analyze(p, ai_bundle, embedder=_UnitEmbedder(dim))
    assert res.label in ("ai", "real"), res.warnings
    assert np.isfinite(res.score) and res.threshold == float(ai_bundle["threshold"])
    assert 0.0 <= res.real_percentile <= 1.0
    assert (res.label == "ai") == (res.score >= res.threshold)
    assert res.n_segments == int(ai_bundle["config"]["max_segments"]) and res.feature_set == ai_bundle["feature_set"]


# ---------------------------------------------- face-swap bundle (D-12) ---


def test_faceswap_bundle_scores_unit_vectors_of_its_width(fs_bundle):
    cfg = fs.merged_config(fs_bundle["config"])
    model = fs_bundle["model"]
    assert fs_bundle.get("kind") != "cnn" and sorted(model.classes_) == ["fake", "real"]
    dim = model.n_features_in_
    assert dim == WIDTH[cfg["backbone"]] * (2 if cfg["feature"] == "cls+patch" else 1)
    thr, ref = _check_threshold_and_references(fs_bundle)
    emb = _unit(np.random.default_rng(1), (32, dim))
    scores = fs.crop_scores(emb, fs_bundle)
    assert scores.shape == (32,)
    _check_scores(scores, thr, ref)
    # a clip's score is the mean crop score: scoring the mean embedding agrees (linear model)
    assert float(scores.mean()) == pytest.approx(float(fs.crop_scores(emb.mean(axis=0, keepdims=True),
                                                                      fs_bundle)[0]), abs=1e-6)


def test_faceswap_bundle_analyses_a_clip_end_to_end(tmp_path, fs_bundle):
    cfg = fs.merged_config(fs_bundle["config"])
    p = synth.write_pulse_video(tmp_path / "v.mp4", fps=30.0, duration_s=3.0)
    r = fs.analyze(p, fs_bundle, embedder=_UnitEmbedder(fs_bundle["model"].n_features_in_),
                   finder=_CentreFinder())
    assert r.label in ("swap", "no_swap"), r.warnings
    assert r.n_faces == r.n_sampled == int(cfg["n_frames"])
    assert np.isfinite(r.score) and r.threshold == float(fs_bundle["threshold"])
    assert 0.0 <= r.real_percentile <= 1.0 and (r.label == "swap") == (r.score >= r.threshold)
    assert r.top_crop.shape == (int(cfg["crop_size"]), int(cfg["crop_size"]), 3)


# ------------------------------------------- scikit-learn version (D-05) ---


def test_version_warning_compares_major_minor():
    w = bundles.sklearn_version_warnings
    assert w("b.joblib", {"1.9.1"}, "1.9.4") == []      # a patch release scores the same
    assert w("b.joblib", set(), "1.9.1") == []
    got = w("b.joblib", {"1.8.2", "1.9.1"}, "1.9.1")
    assert got == [("W_SKLEARN_VERSION: b.joblib was saved with scikit-learn 1.8.2 and is read with 1.9.1; "
                    "its scores may differ from the evaluated ones")]
    assert w("b.joblib", {"pre-0.18"}, "1.9.1")         # unknown counts as different


def _tiny_aigen_bundle(**extra):
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, len(aigen.GEOMETRY_COLUMNS)))
    y = np.array(["real", "ai"] * 20)
    return {"model": LogisticRegression().fit(X, y), "feature_set": "geometry", "threshold": 0.0,
            "config": {"img_size": 32, "cache_dir": None, "max_segments": 1, "graphics_guard": False},
            **extra}


def test_recorded_version_mismatch_reaches_both_analyses(tmp_path):
    """A bundle saved with another major.minor: loading keeps W_SKLEARN_VERSION
    and every analysis reports it, before the reason it did not decide."""
    p = tmp_path / "aigen.joblib"
    joblib.dump(_tiny_aigen_bundle(sklearn_version="0.24.2"), p)
    b = aigen.load_bundle(p)
    assert b[bundles.WARNINGS_KEY][0].startswith("W_SKLEARN_VERSION: aigen.joblib was saved with scikit-learn 0.24.2")
    short = synth.write_pulse_video(tmp_path / "short.mp4", fps=30.0, duration_s=1.0)
    res = aigen.analyze(short, b, embedder=_UnitEmbedder(8))
    assert res.label == "unknown"
    assert res.warnings[0].startswith("W_SKLEARN_VERSION") and res.warnings[-1].startswith("E_SHORT_VIDEO")

    q = tmp_path / "faceswap.joblib"
    joblib.dump({"model": None, "threshold": 0.0, "config": {}, "sklearn_version": "0.24.2"}, q)
    r = fs.analyze(tmp_path / "missing.mp4", fs.load_bundle(q), embedder=_UnitEmbedder(8), finder=_CentreFinder())
    assert r.warnings[0].startswith("W_SKLEARN_VERSION") and r.warnings[-1].startswith("E_UNREADABLE")


def test_matching_version_adds_no_warning(tmp_path):
    import sklearn

    p = tmp_path / "aigen.joblib"
    joblib.dump(_tiny_aigen_bundle(sklearn_version=sklearn.__version__), p)
    assert aigen.load_bundle(p)[bundles.WARNINGS_KEY] == []


def test_older_bundle_mismatch_is_read_from_the_pickled_estimators(tmp_path, monkeypatch):
    """Bundles written before 2026-11-07 carry no sklearn_version key: the
    version scikit-learn stored in each estimator is used, and its own
    InconsistentVersionWarning still reaches the log (nothing is swallowed)."""
    import sklearn
    import sklearn.base
    from sklearn.exceptions import InconsistentVersionWarning

    saved = sklearn.__version__
    p = tmp_path / "aigen.joblib"
    joblib.dump(_tiny_aigen_bundle(), p)  # no sklearn_version key
    monkeypatch.setattr(sklearn.base, "__version__", "2.0.0")  # what unpickling compares against
    monkeypatch.setattr(sklearn, "__version__", "2.0.0")
    with pytest.warns(InconsistentVersionWarning):
        b = aigen.load_bundle(p)
    assert b[bundles.WARNINGS_KEY] == [
        (f"W_SKLEARN_VERSION: aigen.joblib was saved with scikit-learn {saved} and is read with 2.0.0; "
         "its scores may differ from the evaluated ones")]


# ------------------------------- real backbones (CI's full job, weights cached) ---


def _cached(backbone):
    try:
        return aigen.cached_weights(backbone) is not None
    except ImportError:  # no huggingface_hub: the [aigen] extra is not installed
        return False


def test_aigen_bundle_with_its_real_backbone(tmp_path, monkeypatch, ai_bundle):
    backbone = aigen.merged_config(ai_bundle["config"])["backbone"]
    if not _cached(backbone):
        pytest.skip(f"{backbone} weights not cached (trubeat fetch-models)")
    pytest.importorskip("timm")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")  # tests never touch the network
    res = aigen.analyze(_textured_clip(tmp_path / "clip.mp4"), ai_bundle)  # the bundle's own Embedder
    assert res.label in ("ai", "real"), res.warnings
    assert np.isfinite(res.score) and 0.0 <= res.real_percentile <= 1.0
    assert (res.label == "ai") == (res.score >= res.threshold)


def test_faceswap_bundle_with_its_real_backbone(tmp_path, monkeypatch, fs_bundle):
    cfg = fs.merged_config(fs_bundle["config"])
    if not _cached(cfg["backbone"]):
        pytest.skip(f"{cfg['backbone']} weights not cached (trubeat fetch-models)")
    pytest.importorskip("timm")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    emb = fs.FaceEmbedder(cfg["backbone"], cfg["feature"], torch_threads=2, img_size=int(cfg["crop_size"]))
    assert emb.dim == fs_bundle["model"].n_features_in_  # the bundle fits its backbone
    crops = np.random.default_rng(2).integers(0, 256, (4, 224, 224, 3), dtype=np.uint8)
    e = emb.embed(crops)
    assert e.shape == (4, emb.dim) and np.allclose(np.linalg.norm(e, axis=1), 1.0, atol=1e-3)
    assert np.all(np.isfinite(fs.crop_scores(e, fs_bundle)))
    p = synth.write_pulse_video(tmp_path / "v.mp4", fps=30.0, duration_s=3.0)
    r = fs.analyze(p, fs_bundle, embedder=emb, finder=_CentreFinder())
    assert r.label in ("swap", "no_swap") and np.isfinite(r.score) and 0.0 <= r.real_percentile <= 1.0
