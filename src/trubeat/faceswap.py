"""Face-swap detector (docs/12 §3): face crops from sampled frames, a frozen
vision backbone, a linear classifier.

The same functions run in training (``scripts/train_faceswap.py``, the Colab
notebook) and in the app, so both see identical crops:

1. :func:`sample_frames`: ``n_frames`` frames evenly spaced over the first
   ``max_seconds``, decoded at native resolution (only those frames kept).
2. :func:`face_crops`: MediaPipe Face Landmarker (image mode) finds faces;
   the largest one is cut out as a square box ``crop_scale`` times the
   landmark bounding box and resized to ``crop_size``.
3. :class:`FaceEmbedder`: a frozen timm backbone embeds each crop (the CLS
   token, optionally with the mean patch token appended), each part
   L2-normalised.
4. :func:`analyze`: a trained bundle scores the clip as the mean of the
   per-crop decision values (equal to scoring the mean embedding: the
   model is linear) and reports the most suspicious crop as evidence.

Local only: analysis never touches the network (weights come from
``trubeat fetch-models``).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .errors import ModelMissing, ShortVideo, UnreadableVideo
from .facefind import FINDER_DEFAULTS, TwoPassFinder, square_box

DEFAULT_CONFIG: dict = {
    "n_frames": 16,
    "max_seconds": 120.0,
    "min_seconds": 2.0,      # docs/12 §5 B: shorter clips are not judged
    "min_faces": 4,
    "max_faces": 3,
    "crop_scale": 1.3,
    "crop_size": 224,
    "min_face_px": 40,
    "backbone": "vit_small_patch14_dinov2.lvd142m",
    "feature": "cls",
    "torch_threads": 4,
    "target_fpr": 0.03,
    # "landmarker": the round-3 finder (the Face Landmarker on the whole
    # frame); bundles without this key were trained with it and keep it.
    # "twopass": YuNet + the Landmarker on a crop (docs/13 §3.1).
    "finder": "landmarker",
    **FINDER_DEFAULTS,
}
FEATURES = ("cls", "cls+patch")
FINDERS = ("landmarker", "twopass")
BUNDLE_SEARCH_PATHS = [
    Path("models/faceswap.joblib"),
    Path("data/models/faceswap.joblib"),
    Path(__file__).parent / "models" / "faceswap.joblib",
]
_EPS = 1e-8


def _ri(x: float) -> int:
    """Round to the nearest int (numpy<2 float64.__round__ returns a float)."""
    return int(np.rint(x))

ProgressFn = Callable[[float, str], None]


def merged_config(config: dict | None) -> dict:
    out = dict(DEFAULT_CONFIG)
    out.update(config or {})
    if out["feature"] not in FEATURES:
        raise ValueError(f"feature must be one of {FEATURES}, got {out['feature']!r}")
    if out["finder"] not in FINDERS:
        raise ValueError(f"finder must be one of {FINDERS}, got {out['finder']!r}")
    return out


# Runtime knobs that never change a score (left out of the hash).
_RUNTIME_KEYS = ("torch_threads",)


def config_hash(config: dict) -> str:
    """Hash of every setting that changes a crop, a feature or the
    threshold (docs/13 B-08): sampling, finder, crop geometry, backbone,
    feature, minimum length and the target false-alarm rate."""
    cfg = merged_config(config)
    keys = sorted(k for k in DEFAULT_CONFIG if k not in _RUNTIME_KEYS)
    return hashlib.sha256(json.dumps({k: cfg[k] for k in keys}, sort_keys=True).encode()).hexdigest()[:8]


# ----------------------------------------------------------------- frames ---

@dataclass
class SampledFrames:
    frames: list[np.ndarray]   # RGB uint8, native resolution
    index: list[int]           # frame index of each sampled frame
    time_s: list[float]        # timestamp of each sampled frame
    fps: float
    n_frames: int
    duration_s: float
    width: int
    height: int


def target_indices(n_total: int, fps: float, n: int, max_seconds: float) -> list[int]:
    """``n`` frame indices evenly spaced over the first ``max_seconds``
    (all frames when the clip is shorter than ``n`` frames)."""
    if n_total <= 0:
        return []
    last = n_total - 1
    if fps > 0:
        last = min(last, int(max_seconds * fps) - 1)
    last = max(last, 0)
    if last + 1 <= n:
        return list(range(last + 1))
    return sorted({_ri(x) for x in np.linspace(0, last, n)})


def sample_frames(path: str | Path, n: int, max_seconds: float, region: tuple | None = None) -> SampledFrames:
    """Decode only the ``n`` target frames (native resolution, RGB).

    Non-target frames are grabbed, not converted, so memory stays at ``n``
    frames whatever the clip length. ``region`` (x0, y0, x1, y1, source
    pixels; the extension's video element) crops each kept frame, so face
    boxes refer to the cropped frame. Raises UnreadableVideo.
    """
    import cv2

    from .ingest import clip_region

    p = Path(path)
    if not p.is_file():
        raise UnreadableVideo(f"not a file: {p.name}")
    cap = cv2.VideoCapture(str(p))
    try:
        if not cap.isOpened():
            raise UnreadableVideo(f"cannot open video: {p.name}")
        box = (clip_region(region, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
                           int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)) if region is not None else None)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if n_total <= 0:
            # No frame count: a MediaRecorder WebM (the browser extension's
            # capture) reports -2**63 frames at "1000 fps". Count from zero,
            # only as far as sampling can reach (by timestamp, so a long file
            # is not decoded in full), and take the frame rate from the
            # timestamps the way the pulse ingest does.
            from .ingest import _choose_fps

            n_total, first_ms, last_ms = 0, None, None
            while cap.grab():
                t_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
                first_ms = t_ms if first_ms is None else first_ms
                last_ms = t_ms
                n_total += 1
                if t_ms - first_ms > max_seconds * 1000.0:
                    break
            fps = _choose_fps(fps, n_total, first_ms, last_ms) if n_total else fps
            cap.release()
            cap = cv2.VideoCapture(str(p))
            if not cap.isOpened():
                raise UnreadableVideo(f"cannot reopen video: {p.name}")
        want = target_indices(n_total, fps, n, max_seconds)
        if not want:
            raise UnreadableVideo(f"no decodable frames: {p.name}")
        wanted, frames, index, times = set(want), [], [], []
        width = height = 0
        for i in range(want[-1] + 1):
            if not cap.grab():
                break
            if i in wanted:
                ok, bgr = cap.retrieve()
                if not ok:
                    continue
                if box is not None:
                    bgr = bgr[box[1]:box[3], box[0]:box[2]]
                height, width = bgr.shape[:2]
                frames.append(np.ascontiguousarray(bgr[:, :, ::-1]))
                index.append(i)
                times.append(float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0 or (i / fps if fps else 0.0))
    finally:
        cap.release()
    if not frames:
        raise UnreadableVideo(f"no decodable frames: {p.name}")
    duration = n_total / fps if fps > 0 else 0.0
    return SampledFrames(frames, index, times, fps, n_total, duration, width, height)


# ------------------------------------------------------------------ faces ---

@dataclass
class FaceCrop:
    crop: np.ndarray    # (crop_size, crop_size, 3) uint8 RGB
    frame_index: int
    time_s: float
    box: tuple[int, int, int, int]   # x0, y0, side, side in native pixels


def cut_square(rgb: np.ndarray, cx: float, cy: float, side: float, size: int) -> np.ndarray:
    """Square crop centred on (cx, cy), zero-padded where it leaves the
    frame, resized to ``size`` (area interpolation when shrinking)."""
    import cv2

    h, w = rgb.shape[:2]
    s = max(1, _ri(side))
    x0, y0 = _ri(cx - s / 2.0), _ri(cy - s / 2.0)
    out = np.zeros((s, s, 3), dtype=np.uint8)
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(w, x0 + s), min(h, y0 + s)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = rgb[sy0:sy1, sx0:sx1]
    interp = cv2.INTER_AREA if s > size else cv2.INTER_LINEAR
    return cv2.resize(out, (size, size), interpolation=interp)


class FaceFinder:
    """The face finder the config names (``finder``): the round-3 Face
    Landmarker on the whole frame, or the two-pass finder (docs/13 §3.1).
    ``largest_face(rgb)`` returns the crop box ``(cx, cy, side)`` or None;
    ``last_seen`` is how many faces the last frame showed."""

    def __init__(self, config: dict | None = None) -> None:
        self.cfg = merged_config(config)
        self._lm = None
        self._two = TwoPassFinder(self.cfg) if self.cfg["finder"] == "twopass" else None
        self.last_seen = 0

    def _load(self):
        from . import face

        model_path = face.resolve_model_path()
        if not model_path.is_file():
            raise ModelMissing(f"Face Landmarker model not found: {model_path}")
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
        except ImportError as exc:
            raise ModelMissing("mediapipe is required for the face-swap detector") from exc
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=int(self.cfg["max_faces"]),
            min_face_detection_confidence=float(self.cfg["lm_detection_confidence"]),
            min_face_presence_confidence=float(self.cfg["lm_presence_confidence"]),
        )
        self._mp = mp
        self._lm = mp_vision.FaceLandmarker.create_from_options(opts)

    def largest_face(self, rgb: np.ndarray) -> tuple[float, float, float] | None:
        """(cx, cy, side) of the largest face's crop box, or None."""
        if self._two is not None:
            box = self._two.largest_face(rgb)
            self.last_seen = self._two.last_seen
            return box
        if self._lm is None:
            self._load()
        h, w = rgb.shape[:2]
        res = self._lm.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                                             data=np.ascontiguousarray(rgb)))
        best = None
        for pts in res.face_landmarks or []:
            xs = np.array([p.x for p in pts], dtype=np.float32) * w
            ys = np.array([p.y for p in pts], dtype=np.float32) * h
            box = square_box(xs, ys, float(self.cfg["crop_scale"]))
            if box[2] / float(self.cfg["crop_scale"]) < float(self.cfg["min_face_px"]):
                continue
            if best is None or box[2] > best[2]:
                best = box
        self.last_seen = int(best is not None)
        return best

    def close(self) -> None:
        if self._two is not None:
            self._two.close()
        if self._lm is not None:
            self._lm.close()
            self._lm = None


def face_crops(sampled: SampledFrames, finder: FaceFinder, config: dict | None = None,
               progress: Callable[[int, int], None] | None = None) -> list[FaceCrop]:
    cfg = merged_config(config)
    out = []
    for k, (rgb, idx, t) in enumerate(zip(sampled.frames, sampled.index, sampled.time_s)):
        if progress is not None:
            progress(k, len(sampled.frames))
        box = finder.largest_face(rgb)
        if box is None:
            continue
        cx, cy, side = box
        crop = cut_square(rgb, cx, cy, side, int(cfg["crop_size"]))
        s = _ri(side)
        out.append(FaceCrop(crop, idx, t, (_ri(cx - s / 2), _ri(cy - s / 2), s, s)))
    return out


# -------------------------------------------------------------- embedding ---

class FaceEmbedder:
    """Frozen timm backbone -> per-crop feature (CLS, or CLS + mean patch),
    each part L2-normalised. Loads weights from the local cache only."""

    def __init__(self, backbone: str, feature: str = "cls", torch_threads: int = 4,
                 device: str = "cpu", img_size: int = 224) -> None:
        if feature not in FEATURES:
            raise ValueError(f"feature must be one of {FEATURES}")
        self.backbone, self.feature, self.device = backbone, feature, device
        self.img_size = int(img_size)
        self.torch_threads = int(torch_threads)
        self._model = None

    def _load(self) -> None:
        from . import aigen

        try:
            import timm
            import torch
        except ImportError as exc:
            raise ModelMissing(
                f"the face-swap detector needs PyTorch and timm ({exc.name} is not installed): "
                'install the CPU build of torch, then `pip install -e ".[aigen]"`, '
                "then run `trubeat fetch-models` once"
            ) from exc
        torch.set_num_threads(self.torch_threads)
        weights = aigen.cached_weights(self.backbone)
        if weights is None:
            raise ModelMissing(
                f"backbone weights for {self.backbone} are not in the local cache; "
                "run `trubeat fetch-models` once (the only step that downloads)"
            )
        # Built at the crop size (DINOv2 defaults to 518 px; its position
        # embedding is interpolated, as in the AI-video detector).
        model = timm.create_model(self.backbone, pretrained=True,
                                  pretrained_cfg_overlay={"file": weights},
                                  img_size=self.img_size, num_classes=0).eval()
        dcfg = timm.data.resolve_data_config({}, model=model)
        self._mean = np.asarray(dcfg["mean"], dtype=np.float32).reshape(1, 3, 1, 1)
        self._std = np.asarray(dcfg["std"], dtype=np.float32).reshape(1, 3, 1, 1)
        self._size = self.img_size  # dcfg reports the pretrained size (518 for DINOv2)
        self._prefix = int(getattr(model, "num_prefix_tokens", 1))
        self._model = model.to(self.device)

    @property
    def dim(self) -> int:
        if self._model is None:
            self._load()
        d = int(self._model.num_features)
        return d * (2 if self.feature == "cls+patch" else 1)

    def embed(self, crops: np.ndarray, batch: int = 16) -> np.ndarray:
        """uint8 (N, S, S, 3) RGB crops -> float32 (N, dim)."""
        import cv2
        import torch

        if self._model is None:
            self._load()
        if len(crops) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        outs = []
        for i in range(0, len(crops), batch):
            chunk = crops[i:i + batch]
            if chunk.shape[1] != self._size:
                chunk = np.stack([cv2.resize(c, (self._size, self._size),
                                             interpolation=cv2.INTER_AREA) for c in chunk])
            x = chunk.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
            x = (x - self._mean) / self._std
            with torch.inference_mode():
                tok = self._model.forward_features(torch.from_numpy(np.ascontiguousarray(x)).to(self.device))
                cls = tok[:, 0]
                parts = [cls]
                if self.feature == "cls+patch":
                    parts.append(tok[:, self._prefix:].mean(dim=1))
                parts = [p / (p.norm(dim=1, keepdim=True) + _EPS) for p in parts]
                outs.append(torch.cat(parts, dim=1).float().cpu().numpy())
        return np.concatenate(outs).astype(np.float32)


class CNNScorer:
    """Stage 2 (docs/12 §3): a fine-tuned CNN, crop -> logit (positive =
    face-swap). Weights load from a local file only; ``trubeat fetch-models``
    downloads them once (``cnn_weights_path`` says where they are looked for)."""

    def __init__(self, arch: str, weights: str | Path, mean, std, torch_threads: int = 4,
                 device: str = "cpu") -> None:
        self.arch, self.weights, self.device = arch, Path(weights), device
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 3, 1, 1)
        self.torch_threads = int(torch_threads)
        self._model = None

    @classmethod
    def from_bundle(cls, bundle: dict, device: str = "cpu") -> CNNScorer:
        cnn = bundle["cnn"]
        path = cnn_weights_path(bundle)
        if path is None:
            stale = _weight_candidates(bundle)
            raise ModelMissing(
                (f"the face-swap CNN weights at {stale[0]} do not match the installed bundle "
                 "(SHA-256 differs: a copy from another training run?)" if stale else
                 f"face-swap CNN weights {cnn['file']} are not on this machine")
                + "; run `trubeat fetch-models` once (the only step that downloads)"
            )
        cfg = merged_config(bundle["config"])
        return cls(cnn["arch"], path, cnn["mean"], cnn["std"], int(cfg["torch_threads"]), device)

    def _load(self) -> None:
        try:
            import timm
            import torch
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ModelMissing(
                f"the face-swap detector needs PyTorch and timm ({exc.name} is not installed): "
                'install the CPU build of torch, then `pip install -e ".[aigen]"`'
            ) from exc
        torch.set_num_threads(self.torch_threads)
        model = timm.create_model(self.arch, pretrained=False, num_classes=1)
        state = {k: v.float() for k, v in load_file(str(self.weights)).items()}
        model.load_state_dict(state)
        self._model = model.eval().to(self.device)

    def score(self, crops: np.ndarray, batch: int = 8) -> np.ndarray:
        """uint8 (N, S, S, 3) RGB crops -> float32 (N,) logits."""
        import torch

        if self._model is None:
            self._load()
        out = []
        for i in range(0, len(crops), batch):
            x = crops[i:i + batch].astype(np.float32).transpose(0, 3, 1, 2) / 255.0
            x = (x - self.mean) / self.std
            with torch.inference_mode():
                out.append(self._model(torch.from_numpy(np.ascontiguousarray(x)).to(self.device))
                           .float().cpu().numpy().reshape(-1))
        return np.concatenate(out).astype(np.float32) if out else np.zeros(0, np.float32)


def model_dirs() -> list[Path]:
    """Where stage-2 weights are looked for, in order (after the bundle's own
    directory): the same places bundles are found, then the download cache."""
    import os

    dirs = [Path(d) for d in (os.environ.get("TRUBEAT_MODEL_DIR"),) if d]
    return dirs + [Path("models"), Path("data/models"), Path(__file__).parent / "models",
                   Path.home() / ".cache" / "trubeat"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _weight_candidates(bundle: dict) -> list[Path]:
    """Local files named like the bundle's stage-2 weights: next to the
    bundle file first, then ``model_dirs()``."""
    file = bundle["cnn"]["file"]
    dirs = ([Path(bundle["_path"]).parent] if bundle.get("_path") else []) + model_dirs()
    seen = dict.fromkeys(d.resolve() for d in dirs)
    return [d / file for d in seen if (d / file).is_file()]


def cnn_weights_path(bundle: dict) -> Path | None:
    """The first local copy of the bundle's weights whose SHA-256 matches the
    bundle; a copy left from another training run is skipped, never paired
    with this bundle's threshold. A bundle without a SHA-256 cannot be
    verified, so no copy is paired with it."""
    want = bundle["cnn"].get("sha256")
    if not want:
        return None
    for p in _weight_candidates(bundle):
        if _sha256(p) == want:
            return p
    return None


def fetch_cnn_weights(bundle: dict) -> Path:
    """Download the stage-2 weights named in the bundle into ~/.cache/trubeat
    and check their SHA-256 (explicit, user-initiated: `trubeat fetch-models`)."""
    import urllib.request

    cnn = bundle["cnn"]
    have = cnn_weights_path(bundle)
    if have is not None:
        return have
    if not cnn.get("url"):
        raise ModelMissing(f"the bundle names no download URL for {cnn['file']}")
    dest = Path.home() / ".cache" / "trubeat" / cnn["file"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(cnn["url"], timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    if cnn.get("sha256") and _sha256(tmp) != cnn["sha256"]:
        tmp.unlink()
        raise ModelMissing(f"downloaded {cnn['file']} does not match its SHA-256")
    tmp.replace(dest)
    return dest


# ------------------------------------------------------------------ bundle ---

def find_bundle(path: str | Path | None = None) -> Path | None:
    if path is not None:
        p = Path(path)
        return p if p.is_file() else None
    for p in BUNDLE_SEARCH_PATHS:
        if p.is_file():
            return p
    return None


def load_bundle(path: str | Path | None = None) -> dict | None:
    import joblib

    p = find_bundle(path)
    if p is None:
        return None
    b = joblib.load(p)
    b["_path"] = str(p)
    return b


def crop_scores(emb: np.ndarray, bundle: dict) -> np.ndarray:
    """Decision value per crop, oriented so positive = face-swap."""
    model = bundle["model"]
    classes = list(model.classes_)
    if sorted(classes) != ["fake", "real"]:
        raise ValueError(f"unexpected classes {classes}")
    s = model.decision_function(emb)
    return s if classes[1] == "fake" else -s


@dataclass
class FaceSwapResult:
    score: float | None
    threshold: float
    label: str                    # "swap" | "no_swap" | "no_face" | "unknown"
    real_percentile: float | None = None
    n_sampled: int = 0
    n_faces: int = 0
    duration_s: float = 0.0
    top_time_s: float | None = None
    warnings: list[str] = field(default_factory=list)
    top_crop: np.ndarray | None = field(default=None, repr=False)
    # the whole sampled frame the top crop came from (native RGB) and the
    # face box in it (x0, y0, side, side; may extend past the frame edge)
    top_frame: np.ndarray | None = field(default=None, repr=False)
    top_box: tuple[int, int, int, int] | None = None

    def to_json(self) -> str:
        d = asdict(self)
        d.pop("top_crop")
        d.pop("top_frame")
        return json.dumps(d)


def analyze(path: str | Path, bundle: dict, embedder: FaceEmbedder | CNNScorer | None = None,
            finder: FaceFinder | None = None, progress: ProgressFn | None = None,
            region: tuple | None = None) -> FaceSwapResult:
    """Score one clip with a trained bundle. Never raises for bad input:
    unreadable clips and clips without enough faces come back as
    ``unknown`` / ``no_face`` with the reason in ``warnings``."""
    cfg = merged_config(bundle["config"])
    thr = float(bundle["threshold"])

    def report(f: float, s: str) -> None:
        if progress is not None:
            progress(f, s)

    report(0.0, "decoding sampled frames")
    try:
        sampled = sample_frames(path, int(cfg["n_frames"]), float(cfg["max_seconds"]), region=region)
    except UnreadableVideo as exc:
        return FaceSwapResult(None, thr, "unknown", warnings=[f"{exc.code}: {exc}"])
    if sampled.fps > 0 and sampled.duration_s < float(cfg["min_seconds"]):
        note = (f"{ShortVideo.code}: the clip is {sampled.duration_s:.1f} s; the face-swap detector "
                f"needs at least {float(cfg['min_seconds']):.0f} s")
        return FaceSwapResult(None, thr, "unknown", n_sampled=len(sampled.frames),
                              duration_s=sampled.duration_s, warnings=[note])
    own_finder = finder is None
    finder = finder or FaceFinder(cfg)
    try:
        crops = face_crops(sampled, finder, cfg, progress=lambda k, n: report(
            0.3 + 0.4 * k / max(n, 1), f"finding the face, frame {k + 1} of {n}"))
    finally:
        if own_finder:
            finder.close()
    base = {"n_sampled": len(sampled.frames), "n_faces": len(crops), "duration_s": sampled.duration_s}
    if len(crops) < int(cfg["min_faces"]):
        note = (f"W_FEW_FACES: a face was found in {len(crops)} of {len(sampled.frames)} sampled "
                f"frames (needs {cfg['min_faces']})")
        return FaceSwapResult(None, thr, "no_face", warnings=[note], **base)
    report(0.75, "scoring the face crops")
    stack = np.stack([c.crop for c in crops])
    if bundle.get("kind") == "cnn":  # stage 2: a fine-tuned CNN scores each crop directly
        scorer = embedder if isinstance(embedder, CNNScorer) else CNNScorer.from_bundle(bundle)
        per = scorer.score(stack)
    else:  # stage 1: frozen backbone + linear model
        embedder = embedder or FaceEmbedder(cfg["backbone"], cfg["feature"], int(cfg["torch_threads"]),
                                            img_size=int(cfg["crop_size"]))
        per = crop_scores(embedder.embed(stack), bundle)
    score = float(per.mean())
    top = int(np.argmax(per))
    ref = bundle.get("real_reference_scores")
    pct = float(np.mean(np.asarray(ref) < score)) if ref is not None and len(ref) else None
    c = crops[top]
    frame = sampled.frames[sampled.index.index(c.frame_index)]
    return FaceSwapResult(score, thr, "swap" if score >= thr else "no_swap", pct,
                          top_time_s=float(c.time_s), top_crop=c.crop, top_frame=frame,
                          top_box=tuple(int(v) for v in c.box), **base)
