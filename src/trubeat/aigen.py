"""Whole-frame AI-generated video detector (docs/11 Phase D).

The rPPG pipeline looks at three skin patches and, on web-quality video,
cannot measure a pulse at all (docs/11, Phase B result). Fully generated
video (text/image-to-video) has no real performer underneath, so its tells
live in the whole frame and in how it evolves. This module measures that:

    sample_segments  -> fixed-rate frames (time-based, fps-independent)
    Embedder         -> frozen vision backbone (DINOv2-S by default), one
                        L2-normalised embedding per frame
    trajectory_features -> geometry of the frame path in embedding space:
                        step size and curvature (ReStraV, Internò et al.,
                        NeurIPS 2025: natural video traces "straighter"
                        paths) and second-order differences (D3, Zheng et
                        al., ICCV 2025: the volatility of the second-order
                        temporal difference)
    feature_vector   -> geometry stats, the mean embedding, or both
    classifier       -> standardised logistic regression (scripts/train_aigen.py)

Everything runs on CPU and locally: the backbone weights are fetched once
from the public timm/Hugging Face hub at first use and cached; analysing a
video makes no network call after that.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .errors import GraphicsOnly, ModelMissing, ShortVideo, UnreadableVideo

DEFAULT_CONFIG: dict = {
    "backbone": "vit_small_patch14_dinov2.lvd142m",
    "img_size": 224,
    "torch_threads": 6,
    "sample_fps": 8.0,
    "n_frames": 16,
    "max_segments": 3,
    "min_duration_s": 2.0,
    "cache_dir": "data/processed/aigen_emb",
    # Title-card guard (docs/12 item A), inference only: a segment whose
    # frames are mostly flat 8x8 blocks AND still (mean grey-level change
    # between consecutive frames below graphics_max_motion) is a title card
    # or graphic; the first natural segment within graphics_search_s is
    # scored instead.
    "graphics_guard": True,
    "graphics_block_std": 4.0,
    "graphics_flat_frac": 0.7,
    "graphics_max_motion": 0.5,
    "graphics_search_s": 30.0,
}

# Geometry statistics computed for each per-step series.
_STATS = ("mean", "std", "min", "p10", "p50", "p90", "max")
_SERIES = ("step", "curv", "acc")
GEOMETRY_COLUMNS = [f"{s}_{k}" for s in _SERIES for k in _STATS] + ["acc_vol_ratio"]

_EPS = 1e-8
_CHUNK = 1 << 20


def merged_config(config: dict | None) -> dict:
    out = dict(DEFAULT_CONFIG)
    out.update(config or {})
    return out


# --------------------------------------------------------------- sampling ---


def _target_times(
    duration_s: float, cfg: dict, last_frame_s: float | None = None
) -> list[np.ndarray]:
    """Sample times (s) for up to max_segments segments.

    Every target lies at or before ``last_frame_s`` (the timestamp of the
    last decoded frame; defaults to ``duration_s``), and segment starts sit
    on the 1/sample_fps grid the reservoir keeps frames on, so each target
    maps to its own decoded frame. (Targets up to ``duration_s``, one frame
    period past the last frame, repeated a frame in the last segment.)
    """
    step = 1.0 / float(cfg["sample_fps"])
    n = int(cfg["n_frames"])
    span = step * (n - 1)
    end = duration_s if last_frame_s is None else last_frame_s
    if duration_s < float(cfg["min_duration_s"]) or end < span - 1e-6:
        return []
    max_seg = int(cfg["max_segments"])
    free = max(0.0, end - span)
    # Evenly spread segment starts over the clip; a clip with room for one
    # segment gets one.
    n_seg = max(1, min(max_seg, int(free // span) + 1))
    raw = [0.0] if n_seg == 1 else np.linspace(0.0, free, n_seg)
    # Floor onto the grid: on-grid and never past `free` (so never past the
    # last frame).
    starts = sorted({np.floor(float(x) / step + 1e-9) * step for x in raw})
    return [s + step * np.arange(n) for s in starts]


def _prep(frame_bgr: np.ndarray, size: int) -> np.ndarray:
    """BGR frame -> RGB, short side resized to `size`, centre-cropped square."""
    import cv2

    h, w = frame_bgr.shape[:2]
    scale = size / float(min(h, w))
    nh, nw = max(size, round(h * scale)), max(size, round(w * scale))
    img = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    y0, x0 = (nh - size) // 2, (nw - size) // 2
    return np.ascontiguousarray(img[y0 : y0 + size, x0 : x0 + size, ::-1])


def _decode_reservoir(
    p: Path, cfg: dict, index_time: bool
) -> tuple[list[float], dict[int, np.ndarray], dict]:
    """One decode pass. Keeps one prepared frame per 1/sample_fps grid point
    (a superset of every segment's frames) plus every frame's timestamp.

    ``kept`` maps grid index k (time k / sample_fps after the first frame)
    to the first frame at or after that time. A grid point a low-fps source
    skips over has no entry; the sampler then reuses the previous frame,
    which is what that source actually showed at that time.

    Timestamps come from CAP_PROP_POS_MSEC (the PTS of the frame just
    grabbed), or from frame index / metadata fps when ``index_time``.
    """
    import cv2

    cap = cv2.VideoCapture(str(p))
    try:
        if not cap.isOpened():
            raise UnreadableVideo(f"cannot open video: {p}")
        fps_meta = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        info = {
            "fps_meta": fps_meta,
            "n_meta": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        }
        size = int(cfg["img_size"])
        interval_ms = 1000.0 / float(cfg["sample_fps"])
        stamps: list[float] = []
        kept: dict[int, np.ndarray] = {}
        t0: float | None = None
        k = 0  # next grid index to fill
        idx = 0
        while cap.grab():
            if index_time:
                ts = idx * 1000.0 / fps_meta
            else:
                ts = float(cap.get(cv2.CAP_PROP_POS_MSEC))
            if t0 is None:
                t0 = ts  # streams may start at a non-zero PTS
            stamps.append(ts)
            if ts - t0 >= k * interval_ms - 1e-6:
                ok, frame = cap.retrieve()
                if ok:
                    kept[k] = _prep(frame, size)
                    # skip grid points this frame already passed (low fps)
                    k = int(np.floor((ts - t0) / interval_ms + 1e-9)) + 1
            idx += 1
    finally:
        cap.release()
    return stamps, kept, info


def sample_segments(
    path: str | Path, config: dict | None = None, starts_s: list[float] | None = None
) -> tuple[np.ndarray, dict]:
    """Decode fixed-rate frame segments from a video.

    Frames are chosen by *presentation timestamp*, not by frame index, so
    variable-frame-rate files and Chrome MediaRecorder WebMs (whose fps
    metadata reads ~1000) sample at the true rate. When a container's
    timestamps are unusable (not increasing), frame index / metadata fps is
    used instead. Only one frame per sampling interval is converted and
    kept, so memory stays small regardless of clip length.

    Returns (frames uint8 (S, N, size, size, 3), meta dict).
    Raises UnreadableVideo, or ShortVideo when no full segment fits.
    """
    cfg = merged_config(config)
    p = Path(path)
    if not p.is_file():
        raise UnreadableVideo(f"not a file: {path}")
    stamps, kept, info = _decode_reservoir(p, cfg, index_time=False)
    if not stamps:
        raise UnreadableVideo(f"no decodable frames: {path}")
    if len(stamps) > 1 and not stamps[-1] > stamps[0]:
        if info["fps_meta"] <= 0:
            raise UnreadableVideo(f"no usable timestamps or fps metadata: {path}")
        stamps, kept, info = _decode_reservoir(p, cfg, index_time=True)
    rel_s = (np.asarray(stamps) - stamps[0]) / 1000.0
    if len(rel_s) > 1:
        duration_s = float(rel_s[-1] * len(rel_s) / (len(rel_s) - 1))
    else:
        duration_s = 0.0
    last_s = float(rel_s[-1]) if len(rel_s) else 0.0
    if starts_s is None:
        times = _target_times(duration_s, cfg, last_frame_s=last_s)
    else:  # explicit segment starts (the title-card guard's search), on the grid
        step_ = 1.0 / float(cfg["sample_fps"])
        n_ = int(cfg["n_frames"])
        times = [np.floor(s / step_ + 1e-9) * step_ + step_ * np.arange(n_) for s in starts_s
                 if np.floor(s / step_ + 1e-9) * step_ + step_ * (n_ - 1) <= last_s + 1e-6]
    if not times:
        raise ShortVideo(
            f"duration={duration_s:.2f}s is shorter than one "
            f"{cfg['n_frames']}-frame segment at {cfg['sample_fps']} fps"
        )
    grid = sorted(kept)
    step = 1.0 / float(cfg["sample_fps"])
    segments = []
    for seg_t in times:
        frames_seg = []
        for t in seg_t:
            g = round(t / step)
            # the frame kept for grid point g, else the latest one before it
            j = int(np.searchsorted(grid, g, side="right")) - 1
            frames_seg.append(kept[grid[max(j, 0)]])
        segments.append(np.stack(frames_seg))
    meta = {
        "duration_s": duration_s,
        "n_decoded": len(stamps),
        "fps_meta": info["fps_meta"],
        "fps_ts": float((len(rel_s) - 1) / rel_s[-1]) if rel_s[-1] > 0 else 0.0,
        "width": info["width"],
        "height": info["height"],
        "n_segments": len(segments),
        "segment_starts_s": [float(seg_t[0]) for seg_t in times],
    }
    return np.stack(segments), meta


def flat_fraction(frame: np.ndarray, block_std: float) -> float:
    """Share of 8x8 blocks whose grey-level std is below ``block_std``."""
    g = frame.astype(np.float32).mean(axis=2)
    h, w = (g.shape[0] // 8) * 8, (g.shape[1] // 8) * 8
    b = g[:h, :w].reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 64)
    return float((b.std(axis=1) < block_std).mean())


def segment_motion(segment: np.ndarray) -> float:
    """Mean absolute grey-level change between consecutive frames."""
    g = segment.astype(np.float32).mean(axis=3)
    return float(np.abs(np.diff(g, axis=0)).mean()) if len(g) > 1 else 0.0


def is_graphic(segment: np.ndarray, config: dict | None = None) -> bool:
    """A title card / graphic: most of each frame is flat, and the frames
    are still (docs/12 item A; flatness alone also caught smooth AI scenes)."""
    cfg = merged_config(config)
    fr = [flat_fraction(f, float(cfg["graphics_block_std"])) for f in segment]
    return (float(np.median(fr)) >= float(cfg["graphics_flat_frac"])
            and segment_motion(segment) < float(cfg["graphics_max_motion"]))


def guarded_segments(path: str | Path, config: dict | None = None) -> tuple[np.ndarray, dict]:
    """sample_segments with title cards removed: graphic segments are dropped;
    if every segment is graphic, the first natural 2 s segment within
    ``graphics_search_s`` is used. Raises GraphicsOnly when there is none."""
    cfg = merged_config(config)
    frames, meta = sample_segments(path, cfg)
    flags = [is_graphic(s, cfg) for s in frames]
    meta["n_graphic_skipped"] = int(sum(flags))
    if not any(flags):
        return frames, meta
    keep = [(s, st) for s, st, f in zip(frames, meta["segment_starts_s"], flags) if not f]
    if not keep:
        span = (int(cfg["n_frames"]) - 1) / float(cfg["sample_fps"])
        cand = [k * span for k in range(1, int(float(cfg["graphics_search_s"]) // span) + 1)]
        try:
            more, m2 = sample_segments(path, cfg, starts_s=cand) if cand else (np.empty(0), {})
        except ShortVideo:  # no further full segment fits in the clip
            more, m2 = np.empty(0), {}
        for s, st in zip(more, m2.get("segment_starts_s", [])):
            if not is_graphic(s, cfg):
                keep = [(s, st)]
                break
    if not keep:
        raise GraphicsOnly(
            f"every 2 s segment in the first {cfg['graphics_search_s']:.0f} s looks like a title "
            "card or graphic; the AI-video detector was not run"
        )
    meta.update(n_segments=len(keep), segment_starts_s=[st for _, st in keep])
    return np.stack([s for s, _ in keep]), meta


# -------------------------------------------------------------- embedding ---


def _hub_repo(backbone: str) -> str:
    return f"timm/{backbone}"


def cached_weights(backbone: str) -> str | None:
    """Path of the backbone's weights in the local Hugging Face cache, or
    None. A pure cache lookup: no network."""
    from huggingface_hub import try_to_load_from_cache

    hit = try_to_load_from_cache(_hub_repo(backbone), "model.safetensors")
    return hit if isinstance(hit, str) else None


def fetch_weights(backbone: str = DEFAULT_CONFIG["backbone"]) -> str:
    """Download the backbone weights into the local cache (explicit,
    user-initiated; the analysis path itself never downloads)."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(_hub_repo(backbone), "model.safetensors")


class Embedder:
    """Frozen backbone -> one L2-normalised embedding per frame (CPU)."""

    def __init__(self, config: dict | None = None) -> None:
        self.cfg = merged_config(config)
        self._model = None
        self._mean = None
        self._std = None

    def _load(self) -> None:
        try:
            import timm
            import torch
        except ImportError as exc:  # the [aigen] extra is optional
            raise ModelMissing(
                f"the AI-video detector needs PyTorch and timm ({exc.name} is not "
                "installed): install the CPU build of torch, then "
                '`pip install -e ".[aigen]"`, then run `trubeat fetch-models` once'
            ) from exc

        torch.set_num_threads(int(self.cfg["torch_threads"]))
        weights = cached_weights(self.cfg["backbone"])
        if weights is None:
            raise ModelMissing(
                f"backbone weights for {self.cfg['backbone']} are not in the local "
                "cache; run `trubeat fetch-models` once (the only step that downloads)"
            )
        # Load from the local file only: analysis never touches the network.
        model = timm.create_model(
            self.cfg["backbone"],
            pretrained=True,
            pretrained_cfg_overlay={"file": weights},
            img_size=int(self.cfg["img_size"]),
            num_classes=0,
        ).eval()
        dcfg = timm.data.resolve_data_config({}, model=model)
        self._mean = np.asarray(dcfg["mean"], dtype=np.float32).reshape(1, 3, 1, 1)
        self._std = np.asarray(dcfg["std"], dtype=np.float32).reshape(1, 3, 1, 1)
        self._model = model

    def embed(self, frames: np.ndarray) -> np.ndarray:
        """uint8 (N, H, W, 3) RGB -> float32 (N, D), unit L2 norm per row."""
        import torch

        if self._model is None:
            self._load()
        x = frames.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        x = (x - self._mean) / self._std
        with torch.inference_mode():
            out = self._model(torch.from_numpy(np.ascontiguousarray(x))).numpy()
        out = out.astype(np.float32)
        return out / (np.linalg.norm(out, axis=1, keepdims=True) + _EPS)


# --------------------------------------------------------------- geometry ---


def _stats(x: np.ndarray) -> list[float]:
    if x.size == 0:
        return [0.0] * len(_STATS)
    return [
        float(np.mean(x)),
        float(np.std(x)),
        float(np.min(x)),
        float(np.percentile(x, 10)),
        float(np.percentile(x, 50)),
        float(np.percentile(x, 90)),
        float(np.max(x)),
    ]


def trajectory_features(emb: np.ndarray) -> dict[str, float]:
    """Geometry of one segment's embedding path (N, D) -> GEOMETRY_COLUMNS.

    * step: |z[t+1] - z[t]|, how far each frame moves in feature space;
    * curv: angle between consecutive steps, in radians (0 = straight);
    * acc:  |z[t+2] - 2 z[t+1] + z[t]|, the second-order difference;
    * acc_vol_ratio: std(acc) / (mean(step) + eps), the D3-style
      second-order volatility, normalised by how much the clip moves.

    A static segment (no motion) has zero steps; its curvature is defined
    as 0 rather than NaN (AGENTS.md §5), and the step/acc stats read 0.
    """
    z = np.asarray(emb, dtype=np.float64)
    if z.ndim != 2 or z.shape[0] < 3:
        raise ValueError(f"trajectory_features needs (N>=3, D) embeddings, got {z.shape}")
    d = np.diff(z, axis=0)
    step = np.linalg.norm(d, axis=1)
    num = np.sum(d[:-1] * d[1:], axis=1)
    den = step[:-1] * step[1:]
    cos = np.where(den > _EPS, num / np.maximum(den, _EPS), 1.0)
    curv = np.arccos(np.clip(cos, -1.0, 1.0))
    acc = np.linalg.norm(np.diff(z, n=2, axis=0), axis=1)
    vals: list[float] = []
    for series in (step, curv, acc):
        vals.extend(_stats(series))
    vals.append(float(np.std(acc) / (np.mean(step) + _EPS)) if step.size else 0.0)
    return dict(zip(GEOMETRY_COLUMNS, vals, strict=True))


def video_descriptor(seg_emb: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    """(S, N, D) segment embeddings -> (geometry averaged over segments,
    mean embedding over every frame)."""
    feats = [trajectory_features(seg) for seg in seg_emb]
    geom = {c: float(np.mean([f[c] for f in feats])) for c in GEOMETRY_COLUMNS}
    mean_emb = seg_emb.reshape(-1, seg_emb.shape[-1]).mean(axis=0)
    return geom, mean_emb.astype(np.float32)


def feature_vector(
    geom: dict[str, float], mean_emb: np.ndarray, feature_set: str
) -> tuple[np.ndarray, list[str]]:
    """Assemble the classifier input for one video and its column names."""
    emb_cols = [f"emb_{i}" for i in range(len(mean_emb))]
    if feature_set == "geometry":
        return np.array([geom[c] for c in GEOMETRY_COLUMNS]), list(GEOMETRY_COLUMNS)
    if feature_set == "embedding":
        return np.asarray(mean_emb, dtype=float), emb_cols
    if feature_set == "combined":
        return (
            np.concatenate([[geom[c] for c in GEOMETRY_COLUMNS], mean_emb]),
            list(GEOMETRY_COLUMNS) + emb_cols,
        )
    raise ValueError(f"unknown feature_set {feature_set!r}")


# ------------------------------------------------------------------ cache ---


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_key(path: Path, config: dict) -> str:
    cfg = merged_config(config)
    h = hashlib.sha256(_file_hash(path).encode())
    h.update(
        json.dumps(
            {k: cfg[k] for k in ("backbone", "img_size", "sample_fps", "n_frames", "max_segments")},
            sort_keys=True,
        ).encode()
    )
    return h.hexdigest()[:20]


def segment_embeddings(
    path: str | Path, embedder: Embedder, config: dict | None = None, use_cache: bool = True
) -> tuple[np.ndarray, dict]:
    """(S, N, D) embeddings for a video, cached by content + parameters."""
    cfg = merged_config(config)
    p = Path(path)
    cache_file = None
    if use_cache and cfg.get("cache_dir"):
        cache_file = Path(cfg["cache_dir"]) / f"{cache_key(p, cfg)}.npz"
        if cache_file.is_file():
            with np.load(cache_file, allow_pickle=False) as z:
                return z["emb"], json.loads(str(z["meta"]))
    frames, meta = sample_segments(p, cfg)
    s, n = frames.shape[:2]
    emb = embedder.embed(frames.reshape(s * n, *frames.shape[2:])).reshape(s, n, -1)
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_file.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, emb=emb, meta=json.dumps(meta))
        tmp.replace(cache_file)
    return emb, meta


# -------------------------------------------------------------- inference ---


@dataclass
class AIGenResult:
    """The whole-frame detector's answer for one video."""

    # Share of the bundle's reference real clips (out-of-fold, never trained
    # on) that score below this clip. The threshold is the 1 - target_fpr
    # quantile of the same scores, so label "ai" <=> real_percentile above
    # 1 - target_fpr: the number shown can never contradict the label.
    real_percentile: float | None
    score: float | None  # raw decision value (positive = AI-generated)
    threshold: float | None
    label: str  # "ai" | "real" | "unknown"
    n_segments: int = 0
    duration_s: float = 0.0
    feature_set: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self))


# Where a trained bundle is looked up (same order as the G2 detector):
# repo-root overrides first, then the copy packaged with the code.
BUNDLE_SEARCH_PATHS = [
    Path("models/aigen.joblib"),
    Path("data/models/aigen.joblib"),
    Path(__file__).parent / "models" / "aigen.joblib",
]


def find_bundle(path: str | Path | None = None) -> dict | None:
    """Load the AI-video bundle from `path` or the search paths; None if absent."""
    for cand in [Path(path)] if path else BUNDLE_SEARCH_PATHS:
        if cand.is_file():
            bundle = load_bundle(cand)
            bundle.setdefault("_path", str(cand))
            return bundle
    return None


def load_bundle(path: str | Path) -> dict:
    import joblib

    bundle = joblib.load(path)
    for key in ("model", "feature_set", "config", "threshold"):
        if key not in bundle:
            raise ValueError(f"aigen bundle {path} is missing {key!r}")
    return bundle


def normalize_for_inference(path: Path, norm: dict | None, workdir: Path) -> tuple[Path, list[str]]:
    """Re-encode a clip the way training clips were (configs/p5_aigen.yaml
    `normalize`), so the detector sees the same format it was fitted on.
    Falls back to the clip as-is, with a warning, when ffmpeg is missing or
    fails. Local process only; no network."""
    import shutil
    import subprocess

    if not norm:
        return path, []
    if shutil.which("ffmpeg") is None:
        return path, ["W_NO_FFMPEG: analysed as-is; install ffmpeg for training-format input"]
    s = int(norm["short_side"])
    out = workdir / "normalized.mp4"
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(path), "-t", "120",
        "-vf", f"scale='if(gt(iw,ih),-2,{s})':'if(gt(iw,ih),{s},-2)'",
        "-c:v", "libx264", "-crf", str(norm["crf"]), "-preset", str(norm["preset"]),
        "-pix_fmt", "yuv420p", "-an", str(out),
    ]
    try:
        subprocess.run(cmd, check=True, timeout=600, capture_output=True)
        return out, []
    except (subprocess.SubprocessError, OSError) as exc:
        return path, [f"W_NORMALIZE_FAILED: analysed as-is ({type(exc).__name__})"]


def segment_scores(seg_emb: np.ndarray, bundle: dict) -> np.ndarray:
    """Decision value per segment, oriented so positive = AI-generated.

    Training describes each clip by one segment (`train_segments`), so a
    longer clip is scored segment by segment and the scores averaged,
    rather than averaging descriptors the model never saw.
    """
    model = bundle["model"]
    classes = list(model.classes_)
    if sorted(classes) != ["ai", "real"]:
        raise ValueError(f"unexpected classes {classes}")
    xs = []
    for seg in seg_emb:
        geom = trajectory_features(seg)
        x, _ = feature_vector(geom, seg.mean(axis=0), bundle["feature_set"])
        xs.append(x)
    s = model.decision_function(np.vstack(xs))
    return s if classes[1] == "ai" else -s


def analyze(
    path: str | Path,
    bundle: dict,
    embedder: Embedder | None = None,
    use_cache: bool = False,
    normalize: bool = True,
    progress: Callable[[float, str], None] | None = None,
) -> AIGenResult:
    """Score one video with a trained bundle (see scripts/train_aigen.py).

    Never raises for bad input: unreadable or too-short clips come back as
    label "unknown" with the error code in ``warnings``. ``progress(frac,
    stage)`` reports the stage reached (the app's progress bar).
    """
    import tempfile

    def report(frac: float, stage: str) -> None:
        if progress is not None:
            progress(frac, stage)

    cfg = merged_config(bundle["config"])
    cfg["max_segments"] = int(cfg.get("max_segments", 3))
    embedder = embedder or Embedder(cfg)
    thr = float(bundle["threshold"])
    warnings: list[str] = []
    with tempfile.TemporaryDirectory(prefix="trubeat_aigen_") as tmp:
        src = Path(path)
        if normalize and src.is_file():
            report(0.0, "re-encoding the clip")
            src, w = normalize_for_inference(src, cfg.get("normalize"), Path(tmp))
            warnings += w
        report(0.55, "scoring frames with the vision model")
        try:
            if cfg.get("graphics_guard"):
                frames, meta = guarded_segments(src, cfg)
                s_, n_ = frames.shape[:2]
                seg_emb = embedder.embed(frames.reshape(s_ * n_, *frames.shape[2:])).reshape(s_, n_, -1)
            else:
                seg_emb, meta = segment_embeddings(src, embedder, cfg, use_cache=use_cache)
        except (UnreadableVideo, ShortVideo, GraphicsOnly) as exc:
            return AIGenResult(None, None, thr, "unknown", feature_set=bundle["feature_set"],
                               warnings=warnings + [f"{exc.code}: {exc}"])
    scores = segment_scores(seg_emb, bundle)
    score = float(np.mean(scores))
    ref = bundle.get("real_reference_scores")
    pct = float(np.mean(np.asarray(ref) < score)) if ref is not None and len(ref) else None
    n_seg, dur = int(meta.get("n_segments", 0)), float(meta.get("duration_s", 0.0))
    seg_s = (int(cfg["n_frames"]) - 1) / float(cfg["sample_fps"])
    if n_seg < int(cfg["max_segments"]):
        warnings.append(
            f"W_SHORT_CLIP: {n_seg} of the {cfg['max_segments']} segments the model was "
            "evaluated with fit in this clip; lower confidence"
        )
    skipped = int(meta.get("n_graphic_skipped", 0))
    starts = meta.get("segment_starts_s") or [0.0]
    if skipped:
        warnings.append(
            f"W_GRAPHICS_SKIPPED: {skipped} segment{'s' if skipped != 1 else ''} looked like a title "
            f"card or graphic and {'were' if skipped != 1 else 'was'} not scored"
            + (f"; scored the {seg_s:.0f} s segment at {starts[0]:.1f} s instead" if n_seg == 1 else "")
        )
    if dur > n_seg * seg_s + 1.0:
        # The bundle is scored the way it was evaluated (train_segments); a
        # long clip is therefore judged on part of its length only.
        if n_seg == 1:
            where = "the first" if starts[0] < seg_s / 2 else f"the {starts[0]:.1f} s"
        else:
            where = f"{n_seg} evenly spaced"
        warnings.append(
            (f"W_PARTIAL_COVERAGE: scored on {where} {seg_s:.0f} s segment"
             f"{'' if n_seg == 1 else 's'} of a {dur:.1f} s clip") if not where.endswith(" s") else
            f"W_PARTIAL_COVERAGE: scored on one {seg_s:.0f} s segment (starting at "
            f"{starts[0]:.1f} s) of a {dur:.1f} s clip"
        )
    return AIGenResult(
        real_percentile=pct,
        score=score,
        threshold=thr,
        label="ai" if score >= thr else "real",
        n_segments=int(meta.get("n_segments", 0)),
        duration_s=float(meta.get("duration_s", 0.0)),
        feature_set=bundle["feature_set"],
        warnings=warnings,
    )
