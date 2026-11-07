"""Video ingest: decode + metadata. FR-01, FR-02."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .errors import LowFps, ShortVideo, UnreadableVideo

MIN_FPS = 15.0
MIN_DURATION_S = 10.0

# Chrome's MediaRecorder writes WebM containers whose metadata fps is
# ~1000 (millisecond-resolution frame timestamps smear into the rate) and
# whose FRAME_COUNT is meaningless, while the real cadence lives on the
# frame timestamps (~25-30 fps). open_video cross-checks metadata against
# CAP_PROP_POS_MSEC and keeps the honest one:
FPS_DISAGREEMENT = 0.2   # trust metadata only within 20% of the timestamps
MAX_TRUSTED_FPS = 240.0  # metadata above this (e.g. ~1000) is never trusted

# Decode-size ceiling. The SHORT side is capped at this (480: FF++ c23's
# canonical height; UBFC's 640x480 stays untouched, so G1 is unaffected) and
# the long side at 16/9 of it, aspect preserved and never upscaled, so the
# decoded buffer stays bounded for landscape and portrait video alike (a
# 1920x1080x927-frame clip is 5.8 GB as uint8 and OOM-kills a 7 GB machine).
# Before 2026-11-07 (docs/13 A-17) the HEIGHT was capped, which shrank
# vertical phone video to 270x480.
DEFAULT_FRAME_HEIGHT = 480

# Ceiling on the decoded buffer. Decoding stops when it is reached and the
# stream says so (``truncated``); a budget that cannot hold even one frame
# raises E_UNREADABLE instead of letting the kernel OOM-kill us.
MAX_DECODED_BYTES = 2_500_000_000


@dataclass
class VideoStream:
    frames: np.ndarray  # uint8 (T, H, W, 3), RGB
    fps: float
    n_frames: int
    width: int   # of the decoded frames (after the region crop and the downscale)
    height: int
    duration_s: float  # of the decoded frames
    path: str
    # True when decoding stopped before the end of the clip (max_seconds or
    # the byte budget); full_duration_s is the container's estimate, if known.
    truncated: bool = False
    full_duration_s: float | None = None


def decode_scale(w: int, h: int, frame_height: int | None) -> float:
    """Downscale factor (<= 1) that caps the short side at ``frame_height``
    and the long side at 16/9 of it."""
    if frame_height is None or frame_height <= 0:
        return 1.0
    k = 1.0
    if min(w, h) > frame_height:
        k = frame_height / float(min(w, h))
    long_cap = frame_height * 16.0 / 9.0
    if max(w, h) * k > long_cap:
        k = long_cap / float(max(w, h))
    return min(1.0, k)


def clip_region(region, w: int, h: int) -> tuple[int, int, int, int] | None:
    """(x0, y0, x1, y1) inside a w x h frame, or None when ``region`` is None.
    Raises UnreadableVideo for a region that does not overlap the frame."""
    if region is None:
        return None
    x0, y0, x1, y1 = (round(float(v)) for v in region)
    x0, y0 = max(0, min(x0, w)), max(0, min(y0, h))
    x1, y1 = max(0, min(x1, w)), max(0, min(y1, h))
    if x1 - x0 < 16 or y1 - y0 < 16:
        raise UnreadableVideo(f"the region {tuple(region)} does not overlap the {w}x{h} frame")
    return x0, y0, x1, y1


@dataclass
class Probe:
    """What a container says about a video, cross-checked against the frame
    timestamps when the metadata cannot be trusted (a MediaRecorder WebM
    reports -2**63 frames at "1000 fps")."""

    fps: float
    n_frames: int
    duration_s: float
    width: int
    height: int
    from_metadata: bool


def probe(path: str | Path, max_seconds: float | None = None) -> Probe:
    """Frame rate, frame count and duration, from metadata when it is sane,
    else from the timestamps of a grab-only pass (up to ``max_seconds``).
    Never raises: an unreadable file gives zeros."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return Probe(0.0, 0, 0.0, 0, 0, True)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if np.isfinite(fps) and 0 < fps <= MAX_TRUSTED_FPS and n > 0:
            return Probe(fps, n, n / fps, w, h, True)
        limit = float(max_seconds) * 1000.0 if max_seconds else float("inf")
        count, first, last = 0, None, None
        while cap.grab():
            t = float(cap.get(cv2.CAP_PROP_POS_MSEC))
            first = t if first is None else first
            if t - first > limit:
                break
            last = t
            count += 1
    finally:
        cap.release()
    rate = _choose_fps(fps if np.isfinite(fps) and fps > 0 else 0.0, count, first, last)
    return Probe(rate, count, count / rate if rate > 0 else 0.0, w, h, False)


def _choose_fps(
    fps_meta: float, n: int, first_ms: float | None, last_ms: float | None
) -> float:
    """Pick the fps that makes duration and Hz math honest.

    Container metadata is trusted unless it disagrees with the frame
    timestamps by more than FPS_DISAGREEMENT, or claims more than
    MAX_TRUSTED_FPS (MediaRecorder's WebM reports ~1000 for a ~28 fps
    tab capture). The timestamp rate needs at least two frames and a
    sane, strictly increasing span; otherwise fall back to metadata.
    """
    valid_ts = (
        n > 1
        and first_ms is not None
        and last_ms is not None
        and last_ms > first_ms
    )
    if not valid_ts:
        return float(fps_meta)
    fps_ts = (n - 1) / ((last_ms - first_ms) / 1000.0)
    if not np.isfinite(fps_ts) or fps_ts <= 0:
        return float(fps_meta)
    disagrees = abs(fps_ts - fps_meta) > FPS_DISAGREEMENT * fps_meta
    if disagrees or fps_meta > MAX_TRUSTED_FPS:
        return float(fps_ts)
    return float(fps_meta)


def open_video(
    path: str,
    frame_height: int | None = DEFAULT_FRAME_HEIGHT,
    progress: Callable[[int, int], None] | None = None,
    max_seconds: float | None = None,
    region: tuple[float, float, float, float] | None = None,
) -> VideoStream:
    """Decode a video file and validate it for rPPG work.

    Determines the TRUE frame rate: container metadata, cross-checked
    against the frame timestamps (see :func:`_choose_fps`), never
    assumes 30.
    Raises E_UNREADABLE / E_LOW_FPS / E_SHORT_VIDEO on bad inputs.

    ``region`` = (x0, y0, x1, y1) in source pixels crops every frame first
    (the browser extension's video element, docs/13 §3.4). Frames are then
    downscaled so the short side is at most ``frame_height`` and the long
    side at most 16/9 of it (aspect preserved, never upscaled); pass
    frame_height=None to keep native resolution. Decoding stops after
    ``max_seconds`` (by frame timestamp) or when the decoded buffer would
    exceed ``MAX_DECODED_BYTES``; the stream then says ``truncated``.

    ``progress(done, total)`` is called every 24 decoded frames when the
    container reports a frame count (the app's progress bar).

    Error mapping:
      - not a file / cannot open / no decodable frames / missing fps
        metadata / a byte budget that cannot hold one frame / a region
        outside the frame -> UnreadableVideo (E_UNREADABLE)
      - 0 < fps < MIN_FPS -> LowFps (E_LOW_FPS)
      - duration < MIN_DURATION_S -> ShortVideo (E_SHORT_VIDEO)
    """
    p = Path(path)
    if not p.is_file():
        raise UnreadableVideo(f"not a file: {p.name}")
    cap = cv2.VideoCapture(str(p))
    truncated = False
    try:
        if not cap.isOpened():
            raise UnreadableVideo(f"cannot open video: {p.name}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            # Without frame-rate metadata no bpm math is possible.
            raise UnreadableVideo(f"missing frame-rate metadata: {p.name}")
        # A MediaRecorder WebM reports -2**63 frames: treat any count <= 0 as unknown.
        n_meta = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        first_ok, first = cap.read()
        if not first_ok:
            raise UnreadableVideo(f"no decodable frames: {p.name}")
        # CAP_PROP_POS_MSEC is the PTS of the frame just decoded (verified
        # on a real MediaRecorder WebM: 0.0 after the first read). The
        # first/last pair decides honest fps when metadata lies.
        first_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
        last_ms = first_ms
        box = clip_region(region, first.shape[1], first.shape[0])
        if box is not None:
            x0, y0, x1, y1 = box
            first = first[y0:y1, x0:x1]
        h, w = first.shape[0], first.shape[1]
        scale = decode_scale(w, h, frame_height)
        if scale < 1.0:
            h, w = max(1, round(h * scale)), max(1, round(w * scale))
        budget = MAX_DECODED_BYTES // (h * w * 3)
        if budget < 1:
            raise UnreadableVideo(
                f"decoded buffer too large: one {w}x{h} frame exceeds MAX_DECODED_BYTES "
                f"({MAX_DECODED_BYTES / 1e9:.2f} GB) for {p.name}")
        limit_ms = None if not max_seconds else float(max_seconds) * 1000.0
        trusted = fps <= MAX_TRUSTED_FPS
        # Frames we may keep: the count, the seconds limit (when the rate is
        # trustworthy) and the byte budget, whichever is smallest.
        cap_frames = budget
        if n_meta > 0:
            cap_frames = min(cap_frames, n_meta)
        if limit_ms is not None and trusted:
            cap_frames = min(cap_frames, int(np.ceil(float(max_seconds) * fps)) + 1)

        def _prep(frame: np.ndarray) -> np.ndarray:
            """Crop, downscale (if needed) and swap BGR -> RGB."""
            if box is not None:
                frame = frame[box[1]:box[3], box[0]:box[2]]
            if scale < 1.0:
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            return frame[:, :, ::-1]

        frames = np.empty((cap_frames, h, w, 3), dtype=np.uint8) if n_meta > 0 else None
        kept: list[np.ndarray] = []
        first_rgb = (cv2.resize(first, (w, h), interpolation=cv2.INTER_AREA) if scale < 1.0 else first)[:, :, ::-1]
        if frames is not None:
            frames[0] = first_rgb
        else:
            kept.append(np.ascontiguousarray(first_rgb))
        n = 1
        while True:
            if n >= cap_frames:
                # stopped by the seconds limit, the count or the budget; only
                # the count is "the end of the clip"
                truncated = n_meta == 0 or n < n_meta
                if truncated and n_meta == 0:
                    truncated = cap.grab()       # anything left after our last frame?
                break
            ok, frame = cap.read()
            if not ok:
                break
            t_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
            if limit_ms is not None and t_ms - first_ms > limit_ms:
                truncated = True
                break
            last_ms = t_ms
            if frames is not None:
                frames[n] = _prep(frame)
            else:
                kept.append(np.ascontiguousarray(_prep(frame)))
            n += 1
            if progress is not None and n % 24 == 0 and n_meta > 0:
                progress(n, min(n_meta, cap_frames))
        if frames is not None:
            if n < len(frames):          # metadata overcounted, or we stopped early
                frames = np.ascontiguousarray(frames[:n])
        else:
            frames = np.stack(kept, axis=0)
            del kept
    finally:
        cap.release()
    fps_meta = fps
    fps = _choose_fps(fps, n, first_ms, last_ms)
    if fps < MIN_FPS:
        raise LowFps(f"fps={fps:.2f} < MIN_FPS={MIN_FPS}")
    duration_s = n / fps
    if duration_s < MIN_DURATION_S:
        raise ShortVideo(f"duration={duration_s:.2f}s < MIN_DURATION_S={MIN_DURATION_S}s")
    full = n_meta / fps_meta if n_meta > 0 and fps_meta <= MAX_TRUSTED_FPS else None
    return VideoStream(
        frames=frames,
        fps=fps,
        n_frames=n,
        width=int(w),
        height=int(h),
        duration_s=duration_s,
        path=str(p),
        truncated=bool(truncated),
        full_duration_s=full,
    )
