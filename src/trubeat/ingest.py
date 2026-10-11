"""Video ingest: decode + metadata. FR-01, FR-02."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .errors import LowFps, ShortVideo, UnreadableVideo

MIN_FPS = 15.0
MIN_DURATION_S = 10.0


@dataclass
class VideoStream:
    frames: np.ndarray  # uint8 (T, H, W, 3), RGB
    fps: float
    n_frames: int
    width: int
    height: int
    duration_s: float
    path: str


def open_video(path: str) -> VideoStream:
    """Decode a video file and validate it for rPPG work.

    Reads the TRUE frame rate from the container (never assumes 30).
    Raises E_UNREADABLE / E_LOW_FPS / E_SHORT_VIDEO on bad inputs.

    Error mapping:
      - not a file / cannot open / no decodable frames / missing fps
        metadata -> UnreadableVideo (E_UNREADABLE)
      - 0 < fps < MIN_FPS -> LowFps (E_LOW_FPS)
      - duration < MIN_DURATION_S -> ShortVideo (E_SHORT_VIDEO)
    """
    p = Path(path)
    if not p.is_file():
        raise UnreadableVideo(f"not a file: {path}")
    cap = cv2.VideoCapture(str(p))
    try:
        if not cap.isOpened():
            raise UnreadableVideo(f"cannot open video: {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            # Without frame-rate metadata no bpm math is possible.
            raise UnreadableVideo(f"missing frame-rate metadata: {path}")
        frames_bgr: list[np.ndarray] = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames_bgr.append(frame)
        if not frames_bgr:
            raise UnreadableVideo(f"no decodable frames: {path}")
    finally:
        cap.release()
    if fps < MIN_FPS:
        raise LowFps(f"fps={fps:.2f} < MIN_FPS={MIN_FPS}")
    bgr = np.stack(frames_bgr, axis=0)  # (T, H, W, 3) BGR uint8
    frames = np.ascontiguousarray(bgr[:, :, :, ::-1])  # -> RGB
    n, h, w = frames.shape[0], frames.shape[1], frames.shape[2]
    duration_s = n / fps
    if duration_s < MIN_DURATION_S:
        raise ShortVideo(f"duration={duration_s:.2f}s < MIN_DURATION_S={MIN_DURATION_S}s")
    return VideoStream(
        frames=frames,
        fps=fps,
        n_frames=n,
        width=int(w),
        height=int(h),
        duration_s=duration_s,
        path=str(p),
    )
