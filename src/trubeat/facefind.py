"""Two-pass face finder (docs/13 §3.1): find small faces, then landmark them.

The MediaPipe Face Landmarker sees a whole frame through a short-range
detector at 128x128, so it misses any face under roughly 12-14% of the
frame width (docs/13 §1.2: a 200 px face in a 1920x1080 frame). This
finder keeps the round-3 pass and adds a second one for the faces it
misses:

1. **The round-3 pass:** the Landmarker on the whole frame (IMAGE mode).
   When it finds a face, that face is returned unchanged, so every crop the
   round-3 finder cut is cut identically (docs/13 §6.1 rules 2-3).
2. **Only when it finds none: detect** with OpenCV's YuNet
   (``cv2.FaceDetectorYN``, MIT licence,
   ``models/face_detection_yunet_2023mar.onnx``) on a copy of the frame
   shrunk to at most ``detect_max_side`` pixels (it finds faces down to
   about 10 px), then **confirm and landmark** each candidate, largest
   first, with the Landmarker on a square crop ``landmark_margin`` times
   the detection box, resized to between ``landmark_min_crop`` and
   ``landmark_max_crop`` pixels. A candidate without landmarks is not a
   face.

(The first implementation ran YuNet first; a 6-clip dry run of the
adoption check showed crop-based landmarks differing from whole-frame ones
by more than 5% of the face size in a third of frames, so the round-3 pass
now runs first and decides whenever it can.)

The returned box is the round-3 crop box: a square ``crop_scale`` times the
landmark bounding box. Faces whose landmark box is under ``min_face_px``
(native pixels) are ignored, as before.

Local only: no network; the models ship inside the package.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import ModelMissing

YUNET_FILE = "face_detection_yunet_2023mar.onnx"

FINDER_DEFAULTS: dict = {
    "detect_max_side": 1280,     # YuNet runs on a copy no larger than this (px, longest side)
    "detect_score": 0.6,         # YuNet score floor
    "detect_nms": 0.3,           # YuNet non-maximum suppression IoU
    "landmark_margin": 2.0,      # landmark crop side = margin x detection box side
    "landmark_min_crop": 256,    # small faces are upscaled to at least this crop size
    "landmark_max_crop": 512,    # large faces are downscaled to at most this
    "lm_detection_confidence": 0.5,
    "lm_presence_confidence": 0.5,
}


def resolve_yunet_path() -> Path:
    """``TRUBEAT_YUNET_MODEL``, then the repo's ``data/models/``, then the
    copy packaged with trubeat (the same order as the Face Landmarker)."""
    env = os.environ.get("TRUBEAT_YUNET_MODEL")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    repo = here.parents[2] / "data" / "models" / YUNET_FILE
    if repo.is_file():
        return repo
    return here.parent / "models" / YUNET_FILE


@dataclass
class Detection:
    """A YuNet box in native pixels."""

    x: float
    y: float
    w: float
    h: float
    score: float

    @property
    def side(self) -> float:
        return max(self.w, self.h)

    @property
    def centre(self) -> tuple[float, float]:
        return self.x + self.w / 2.0, self.y + self.h / 2.0


def square_box(xs: np.ndarray, ys: np.ndarray, scale: float) -> tuple[float, float, float]:
    """(centre x, centre y, side) of the square ``scale`` x the bounding box."""
    x0, x1, y0, y1 = float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())
    side = max(x1 - x0, y1 - y0) * scale
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0, side


def crop_window(cx: float, cy: float, side: float) -> tuple[int, int, int]:
    """(x0, y0, s): the integer square ``cut_square`` cuts for this box."""
    s = max(1, int(np.rint(side)))
    return int(np.rint(cx - s / 2.0)), int(np.rint(cy - s / 2.0)), s


def crop_to_native(px: np.ndarray, py: np.ndarray, window: tuple[int, int, int],
                   size: int) -> tuple[np.ndarray, np.ndarray]:
    """Map points in a ``size`` x ``size`` crop back to native pixels."""
    x0, y0, s = window
    k = s / float(size)
    return x0 + px * k, y0 + py * k


def scale_detections(faces: np.ndarray | None, k: float, score_floor: float) -> list[Detection]:
    """YuNet rows (x, y, w, h, 10 landmark coords, score) found on a copy
    shrunk by ``k`` -> native-pixel detections above the score floor,
    largest first."""
    if faces is None or len(faces) == 0:
        return []
    out = []
    for f in np.asarray(faces, dtype=np.float32):
        score = float(f[14])
        if score < score_floor or f[2] <= 0 or f[3] <= 0:
            continue
        out.append(Detection(float(f[0]) / k, float(f[1]) / k, float(f[2]) / k, float(f[3]) / k, score))
    out.sort(key=lambda d: d.side, reverse=True)
    return out


class TwoPassFinder:
    """YuNet detection + Face Landmarker confirmation (see the module doc).

    ``largest_face(rgb)`` has the round-3 ``FaceFinder`` contract:
    ``(cx, cy, side)`` of the crop box in native pixels, or None. After each
    call, ``last_seen`` holds how many faces of at least ``min_face_px``
    the deciding pass saw, and ``last_source`` says which pass found the
    returned face ("landmarker", "yunet" or "").
    """

    def __init__(self, config: dict) -> None:
        self.cfg = {**FINDER_DEFAULTS, **config}
        self._det = None
        self._lm = None
        self._mp = None
        self.last_seen = 0
        self.last_source = ""

    # ------------------------------------------------------------- loading
    def _load(self) -> None:
        import cv2

        from . import face

        yunet = resolve_yunet_path()
        if not yunet.is_file():
            raise ModelMissing(f"YuNet face detector model not found: {yunet}")
        lm_path = face.resolve_model_path()
        if not lm_path.is_file():
            raise ModelMissing(f"Face Landmarker model not found: {lm_path}")
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
        except ImportError as exc:
            raise ModelMissing("mediapipe is required for the face-swap detector") from exc
        self._det = cv2.FaceDetectorYN.create(
            str(yunet), "", (320, 320), float(self.cfg["detect_score"]),
            float(self.cfg["detect_nms"]), 5000)
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(lm_path)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=int(self.cfg.get("max_faces", 3)),
            min_face_detection_confidence=float(self.cfg["lm_detection_confidence"]),
            min_face_presence_confidence=float(self.cfg["lm_presence_confidence"]),
        )
        self._mp = mp
        self._lm = mp_vision.FaceLandmarker.create_from_options(opts)

    # ----------------------------------------------------------- the passes
    def detect(self, rgb: np.ndarray) -> list[Detection]:
        """YuNet on a copy no larger than ``detect_max_side``; native boxes."""
        import cv2

        if self._det is None:
            self._load()
        h, w = rgb.shape[:2]
        k = min(1.0, float(self.cfg["detect_max_side"]) / float(max(h, w)))
        small = (cv2.resize(rgb, (max(1, round(w * k)), max(1, round(h * k))), interpolation=cv2.INTER_AREA)
                 if k < 1.0 else rgb)
        self._det.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self._det.detect(np.ascontiguousarray(small[:, :, ::-1]))  # YuNet takes BGR
        return scale_detections(faces, k, float(self.cfg["detect_score"]))

    def _landmark_sets(self, rgb: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """Landmarks (pixel x, pixel y) of every face the Landmarker finds."""
        h, w = rgb.shape[:2]
        res = self._lm.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                                             data=np.ascontiguousarray(rgb)))
        out = []
        for pts in res.face_landmarks or []:
            out.append((np.array([p.x for p in pts], dtype=np.float32) * w,
                        np.array([p.y for p in pts], dtype=np.float32) * h))
        return out

    def confirm(self, rgb: np.ndarray, det: Detection) -> tuple[np.ndarray, np.ndarray] | None:
        """Landmarks of the face in ``det`` (native px), from a crop around it."""
        from .faceswap import cut_square

        cx, cy = det.centre
        side = det.side * float(self.cfg["landmark_margin"])
        size = int(np.clip(round(side), int(self.cfg["landmark_min_crop"]), int(self.cfg["landmark_max_crop"])))
        crop = cut_square(rgb, cx, cy, side, size)
        sets = self._landmark_sets(crop)
        if not sets:
            return None
        # the face nearest the crop centre is the detected one
        c = size / 2.0
        xs, ys = min(sets, key=lambda s: (float(s[0].mean()) - c) ** 2 + (float(s[1].mean()) - c) ** 2)
        return crop_to_native(xs, ys, crop_window(cx, cy, side), size)

    def largest_face(self, rgb: np.ndarray) -> tuple[float, float, float] | None:
        if self._lm is None:
            self._load()
        scale = float(self.cfg["crop_scale"])
        min_px = float(self.cfg["min_face_px"])
        # Pass 1, round-3 behaviour: the Landmarker on the whole frame.
        boxes = [b for b in (square_box(xs, ys, scale) for xs, ys in self._landmark_sets(rgb))
                 if b[2] / scale >= min_px]
        if boxes:
            self.last_seen, self.last_source = len(boxes), "landmarker"
            return max(boxes, key=lambda b: b[2])
        # Pass 2, only for faces pass 1 missed: YuNet, then landmarks on a crop.
        dets = [d for d in self.detect(rgb) if d.side >= min_px]
        self.last_seen, self.last_source = len(dets), ""
        for det in dets[: int(self.cfg.get("max_faces", 3))]:
            lm = self.confirm(rgb, det)
            if lm is None:
                continue
            box = square_box(lm[0], lm[1], scale)
            if box[2] / scale < min_px:
                continue
            self.last_source = "yunet"
            return box
        return None

    def close(self) -> None:
        if self._lm is not None:
            self._lm.close()
            self._lm = None
        self._det = None
