"""Tests for trubeat.model: training + single-video prediction."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from unittest.mock import MagicMock, patch

from trubeat.hr import HRResult
from trubeat.model import Prediction, _to_label, predict_video, train_classifier


def _separable(n_per=20, seed=0):
    """Two well-separated clusters; 4 groups, each holding both classes."""
    rng = np.random.default_rng(seed)
    X0 = rng.normal(0.0, 0.5, (n_per, 4))
    X1 = rng.normal(5.0, 0.5, (n_per, 4))
    X = pd.DataFrame(np.vstack([X0, X1]), columns=list("abcd"))
    y = np.array([0] * n_per + [1] * n_per)
    groups = np.array([i % 4 for i in range(2 * n_per)])
    return X, y, groups


def test_train_classifier_returns_fitted_pipeline(capsys):
    X, y, groups = _separable()
    pipe = train_classifier(X, y, groups)
    assert isinstance(pipe, Pipeline)
    assert [n for n, _ in pipe.steps] == ["scaler", "clf"]
    # fitted on the full data: perfect separation on train
    assert roc_auc_score(y, pipe.predict_proba(X)[:, 1]) == pytest.approx(1.0)
    # CV ran inside and printed mean AUC
    assert "AUC" in capsys.readouterr().out


def _mock_stages(hr_bpm=72.0, valid_frac=1.0):
    """Patch every pipeline stage; returns the patchers' context."""
    stream = MagicMock()
    stream.frames = np.zeros((300, 64, 64, 3), np.uint8)
    stream.fps = 30.0
    landmarks = np.zeros((300, 478, 2), np.float32)
    valid = np.zeros(300, bool)
    valid[: int(300 * valid_frac)] = True
    hr_res = HRResult(
        bpm=hr_bpm, confidence=0.9, spectrum=np.array([1.0]), freqs=np.array([1.2])
    )
    patches = [
        patch("trubeat.model.ingest"),
        patch("trubeat.model.face"),
        patch("trubeat.model.roi"),
        patch("trubeat.model.signal"),
        patch("trubeat.model.features"),
        patch("trubeat.model.hr"),
    ]
    return patches, stream, landmarks, valid, hr_res


def test_predict_video_majority_vote():
    patches, stream, landmarks, valid, hr_res = _mock_stages()
    feat = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    model = MagicMock()
    model.predict.return_value = np.array(["real", "fake", "real"])
    with patches[0] as m_ing, patches[1] as m_face, patches[2] as m_roi, patches[
        3
    ] as m_sig, patches[4] as m_feat, patches[5] as m_hr:
        m_ing.open_video.return_value = stream
        m_face.track_landmarks.return_value = (landmarks, valid)
        m_roi.extract_roi_means.return_value = np.zeros((300, 3, 3))
        m_sig.extract_ppg.return_value = np.zeros((300, 3))
        m_feat.window_features.return_value = feat
        m_hr.estimate_hr.return_value = hr_res
        pred = predict_video("vid.mp4", model, {})
    assert isinstance(pred, Prediction)
    assert pred.label == "real"  # majority wins
    assert pred.votes == ["real", "fake", "real"]
    assert pred.n_windows == 3
    assert pred.confidence == pytest.approx(2 / 3)  # winning vote fraction
    assert pred.hr_bpm == pytest.approx(72.0)
    assert pred.warnings == []
    d = pred.to_json()
    assert d["label"] == "real" and d["n_windows"] == 3


def test_predict_video_numeric_labels_mapped():
    patches, stream, landmarks, valid, hr_res = _mock_stages()
    model = MagicMock()
    model.predict.return_value = np.array([1, 0, 1])  # 1 == real
    with patches[0] as m_ing, patches[1] as m_face, patches[2] as m_roi, patches[
        3
    ] as m_sig, patches[4] as m_feat, patches[5] as m_hr:
        m_ing.open_video.return_value = stream
        m_face.track_landmarks.return_value = (landmarks, valid)
        m_roi.extract_roi_means.return_value = np.zeros((300, 3, 3))
        m_sig.extract_ppg.return_value = np.zeros((300, 3))
        m_feat.window_features.return_value = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        m_hr.estimate_hr.return_value = hr_res
        pred = predict_video("vid.mp4", model, {})
    assert pred.votes == ["real", "fake", "real"]
    assert pred.label == "real"


def test_predict_video_unstable_track_warning():
    patches, stream, landmarks, valid, hr_res = _mock_stages(valid_frac=0.5)
    model = MagicMock()
    model.predict.return_value = np.array(["fake", "fake"])
    with patches[0] as m_ing, patches[1] as m_face, patches[2] as m_roi, patches[
        3
    ] as m_sig, patches[4] as m_feat, patches[5] as m_hr:
        m_ing.open_video.return_value = stream
        m_face.track_landmarks.return_value = (landmarks, valid)
        m_roi.extract_roi_means.return_value = np.zeros((300, 3, 3))
        m_sig.extract_ppg.return_value = np.zeros((300, 3))
        m_feat.window_features.return_value = pd.DataFrame({"a": [1.0, 2.0]})
        m_hr.estimate_hr.return_value = hr_res
        pred = predict_video("vid.mp4", model, {})
    assert pred.label == "fake"
    assert any("W_UNSTABLE_TRACK" in w for w in pred.warnings)


def test_to_label_mapping():
    assert _to_label("real") == "real"
    assert _to_label("fake") == "fake"
    assert _to_label(1) == "real"
    assert _to_label(0) == "fake"
    assert _to_label(True) == "real"
