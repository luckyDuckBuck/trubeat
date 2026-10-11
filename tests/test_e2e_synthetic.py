"""A12: synthetic end-to-end pipeline validation (no real data needed).

Proves the full chain works before it ever sees a real face:
  array-level: traces → pos → window_features → train_classifier → separation
  video-level: mp4 → ingest → (mocked face) → roi → signal → hr/features → model
The only mocked link is the face detector itself (needs real faces; human gate).
"""

import numpy as np
import pytest

import synth
from trubeat import features, model, roi, signal

FPS = 30.0
N = 750  # 25 s


def _coherent_traces(n=N, fps=FPS, bpm=72.0, seed=0, noise=0.15):
    t = np.arange(n) / fps
    base = np.zeros((n, 3, 3))
    base[:, :, 0] = 120.0
    base[:, :, 1] = 140.0
    base[:, :, 2] = 150.0
    sig = np.sin(2 * np.pi * (bpm / 60.0) * t)
    rng = np.random.default_rng(seed)
    for r in range(3):
        base[:, r, 1] += 8.0 * sig + rng.normal(0, noise, n)
    return base


def _incoherent_traces(n=N, fps=FPS, seed=1):
    t = np.arange(n) / fps
    base = np.zeros((n, 3, 3))
    base[:, :, 0] = 120.0
    base[:, :, 1] = 140.0
    base[:, :, 2] = 150.0
    rng = np.random.default_rng(seed)
    for r, (bpm, phase) in enumerate([(72.0, 0.0), (95.0, 1.1), (61.0, 2.4)]):
        base[:, r, 1] += 8.0 * np.sin(2 * np.pi * (bpm / 60.0) * t + phase)
        base[:, r, 1] += rng.normal(0, 0.15, n)
    return base


def _featurize(traces):
    ppg = signal.extract_ppg(traces, FPS, method="pos")
    return features.window_features(ppg, FPS)


def test_array_level_separation():
    """Coherent vs incoherent ROI pulses must be perfectly separable."""
    sets = [
        (_coherent_traces(bpm=72.0, seed=0), "real", "g0"),
        (_coherent_traces(bpm=80.0, seed=2), "real", "g1"),
        (_incoherent_traces(seed=1), "fake", "g2"),
        (_incoherent_traces(seed=3), "fake", "g3"),
    ]
    parts, y, groups = [], [], []
    for traces, label, g in sets:
        f = _featurize(traces)
        assert len(f) > 0
        parts.append(f)
        y += [label] * len(f)
        groups += [g] * len(f)
    import pandas as pd

    X = pd.concat(parts, ignore_index=True)
    clf = model.train_classifier(X, np.array(y), np.array(groups))
    acc = float(np.mean(clf.predict(X) == np.array(y)))
    assert acc == 1.0, f"synthetic separation failed: acc={acc}"


def _lr_landmarks(n, h=240, w=320):
    """Indices 0-99 → left half, 100-199 → right half (for incoherent videos)."""
    rng = np.random.default_rng(0)
    lm = np.zeros((n, 478, 2), np.float32)
    lm[:, :100, 0] = rng.uniform(w * 0.1, w * 0.45, (n, 100))
    lm[:, :100, 1] = rng.uniform(h * 0.3, h * 0.7, (n, 100))
    lm[:, 100:200, 0] = rng.uniform(w * 0.55, w * 0.9, (n, 100))
    lm[:, 100:200, 1] = rng.uniform(h * 0.3, h * 0.7, (n, 100))
    lm[:, 200:, 0] = w / 2
    lm[:, 200:, 1] = h / 2
    return lm, np.ones(n, bool)


@pytest.fixture()
def mock_face_and_rois(monkeypatch):
    import trubeat.face as facemod
    import trubeat.roi as roimod

    def fake_track(frames, fps):
        return _lr_landmarks(len(frames), h=frames.shape[1], w=frames.shape[2])

    monkeypatch.setattr(facemod, "track_landmarks", fake_track)
    monkeypatch.setattr(
        roimod, "ROI_DEFS_P1",
        {"left": list(range(100)), "right": list(range(100, 200))},
    )


def test_video_level_hr(tmp_path, mock_face_and_rois):
    """Full video path: mp4 → ingest → roi → pos → hr ≈ 72."""
    from trubeat import hr, ingest

    vid = synth.write_pulse_video(tmp_path / "coh.mp4", bpm=72.0, duration_s=25.0)
    stream = ingest.open_video(str(vid))
    landmarks, valid = _lr_landmarks(len(stream.frames),
                                     h=stream.frames.shape[1],
                                     w=stream.frames.shape[2])
    traces = roi.extract_roi_means(stream.frames, landmarks, valid,
                                   {"left": list(range(100)),
                                    "right": list(range(100, 200))})
    ppg = signal.extract_ppg(traces, stream.fps, method="pos")
    res = hr.estimate_hr(np.nanmean(ppg, axis=1), stream.fps)
    assert abs(res.bpm - 72.0) <= 4.0, f"bpm={res.bpm}"


def test_video_level_separation(tmp_path, mock_face_and_rois):
    """Two coherent + two incoherent videos → classifier separates them."""
    import pandas as pd

    specs = [("coh", 72.0, 0, "real", "v0"), ("coh", 80.0, 2, "real", "v1"),
             ("inc", 72.0, 1, "fake", "v2"), ("inc", 72.0, 3, "fake", "v3")]
    parts, y, groups = [], [], []
    for mode, bpm, seed, label, g in specs:
        vid = synth.write_pulse_video(tmp_path / f"{g}.mp4", bpm=bpm,
                                      duration_s=25.0, seed=seed,
                                      mode="coherent" if mode == "coh" else "incoherent")
        from trubeat import ingest

        stream = ingest.open_video(str(vid))
        landmarks, valid = _lr_landmarks(len(stream.frames),
                                         h=stream.frames.shape[1],
                                         w=stream.frames.shape[2])
        traces = roi.extract_roi_means(
            stream.frames, landmarks, valid,
            {"left": list(range(100)), "right": list(range(100, 200))})
        ppg = signal.extract_ppg(traces, stream.fps, method="pos")
        f = features.window_features(ppg, stream.fps)
        assert len(f) > 0, f"no windows for {g}"
        parts.append(f)
        y += [label] * len(f)
        groups += [g] * len(f)
    X = pd.concat(parts, ignore_index=True)
    clf = model.train_classifier(X, np.array(y), np.array(groups))
    acc = float(np.mean(clf.predict(X) == np.array(y)))
    assert acc == 1.0, f"video-level separation failed: acc={acc}"
