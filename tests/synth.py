"""Synthetic fixtures for tests: pulse videos, landmark arrays, PPG signals.

No real faces, no datasets needed.
- A synthetic "real" video: every region pulses coherently at a known bpm.
- A synthetic "fake" video: left/right halves pulse incoherently.
- NOTE: mp4v compression attenuates the synthetic pulse slightly, so
  video-level bpm assertions need tolerance (±4 bpm). Array-level tests
  (pulse_wave direct) are exact.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def pulse_wave(n, fps, bpm, phase=0.0, noise_std=0.0, seed=0):
    """Pure sine pulse wave, length n samples."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fps
    w = np.sin(2 * np.pi * (bpm / 60.0) * t + phase)
    if noise_std:
        w = w + rng.normal(0, noise_std, n)
    return w


def write_pulse_video(path, fps=30.0, duration_s=10.0, bpm=72.0,
                      frame_size=(320, 240), mode="coherent", seed=0,
                      amplitude=10.0, base=140.0):
    """Write an .mp4 whose green channel carries a pulse at `bpm`.

    mode="coherent": whole frame pulses in phase (synthetic REAL).
    mode="incoherent": left/right halves pulse at different rate/phase
        (synthetic FAKE — regions disagree).
    """
    if cv2 is None:
        raise RuntimeError("cv2 is required for synthetic videos")
    w, h = frame_size
    n = int(fps * duration_s)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not vw.isOpened():
        raise RuntimeError(f"could not open VideoWriter for {path}")
    sig_l = pulse_wave(n, fps, bpm, phase=0.0, seed=seed)
    sig_r = pulse_wave(n, fps, bpm * 1.31, phase=np.pi / 2, seed=seed + 1)
    for i in range(n):
        frame = np.zeros((h, w, 3), dtype=np.float32)
        frame[:, :, 0] = base - 20  # B
        frame[:, :, 1] = base       # G (the pulse channel, BGR order)
        frame[:, :, 2] = base + 10  # R
        if mode == "coherent":
            frame[:, :, 1] += amplitude * sig_l[i]
        elif mode == "incoherent":
            frame[:, : w // 2, 1] += amplitude * sig_l[i]
            frame[:, w // 2 :, 1] += amplitude * sig_r[i]
        else:
            raise ValueError(f"unknown mode {mode}")
        vw.write(np.clip(frame, 0, 255).astype(np.uint8))
    vw.release()
    return Path(path)


def write_corrupt_video(path):
    """Bytes that are not a video → ingest must raise E_UNREADABLE."""
    p = Path(path)
    p.write_bytes(b"\x00\x01\x02not a video\xff\xfe")
    return p


def synthetic_landmarks(n_frames, h=240, w=320, seed=0):
    """Random (T,478,2) float32 landmarks inside a central ellipse.

    Returns (landmarks, valid) with valid all-True.
    """
    rng = np.random.default_rng(seed)
    cx, cy, rx, ry = w / 2, h * 0.45, w * 0.28, h * 0.32
    ang = rng.uniform(0, 2 * np.pi, (n_frames, 478))
    rad = np.sqrt(rng.uniform(0, 1, (n_frames, 478)))
    x = cx + rx * rad * np.cos(ang)
    y = cy + ry * rad * np.sin(ang)
    lm = np.stack([x, y], axis=-1).astype(np.float32)
    return lm, np.ones(n_frames, dtype=bool)
