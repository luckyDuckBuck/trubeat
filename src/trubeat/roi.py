"""ROI definitions + per-frame mean RGB extraction. FR-04, FR-05."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# Index sets below come from the raffahitipeuw/health rPPG project
# (rppg_core.py, documented in its PHASE2_REALDATA_UNBLOCK_REPORT.md), which
# reports them as anatomically validated against MediaPipe Tasks
# FaceLandmarker (478-pt topology). Left/right follow that project's naming;
# the cheek sets are mirror images (right ~= left + 229/230), consistent with
# FaceMesh symmetry. Verify visually via save_roi_placement_plot() against
# MediaPipe's published FaceMesh diagram before trusting a new topology.
ROI_DEFS_P1: dict[str, list[int]] = {
    "forehead": [109, 67, 108, 151, 337, 297, 338],
    "cheek_l": [50, 101, 118, 117, 116, 123, 147, 187, 207, 206, 205, 36, 142, 126, 100, 203],
    "cheek_r": [280, 330, 347, 346, 345, 352, 376, 411, 427, 426, 425, 266, 371, 355, 329, 423],
}

# Phase 3: 32 sub-regions tiled across the face (FakeCatcher layout).
ROI_DEFS_P3: dict[str, list[int]] = {}


def extract_roi_means(
    frames: np.ndarray,
    landmarks: np.ndarray,
    valid: np.ndarray,
    roi_defs: dict[str, list[int]],
) -> np.ndarray:
    """Mean R/G/B inside each ROI polygon, per frame.

    Args:
        frames: uint8 (T, H, W, 3) RGB.
        landmarks: float32 (T, 478, 2) pixel coords (NaN where invalid).
        valid: bool (T,) per-frame face-track flag.
        roi_defs: {roi_name: [landmark indices]}; <3 indices -> NaN column.

    Returns:
        traces: float64 (T, R, 3). Rows for invalid frames are NaN.
    """
    t = int(frames.shape[0])
    h, w = int(frames.shape[1]), int(frames.shape[2])
    names = list(roi_defs.keys())
    traces = np.full((t, len(names), 3), np.nan, dtype=np.float64)
    for j, name in enumerate(names):
        idx = list(roi_defs[name])
        if len(idx) < 3:
            continue  # degenerate ROI: column stays NaN
        if min(idx) < 0 or max(idx) >= landmarks.shape[1]:
            raise ValueError(
                f"ROI '{name}': landmark index out of range "
                f"(got {min(idx)}..{max(idx)}, mesh has {landmarks.shape[1]} points)"
            )
        for i in range(t):
            if not valid[i]:
                continue
            pts = landmarks[i, idx]
            if not np.all(np.isfinite(pts)):
                continue
            poly = np.round(pts).astype(np.int32).reshape((-1, 1, 2))
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [poly], 1)
            sel = mask.astype(bool)
            if not np.any(sel):
                continue
            traces[i, j] = frames[i][sel].mean(axis=0)
    return traces


def save_roi_placement_plot(
    path: str | None = None, h: int = 240, w: int = 320, seed: int = 0
) -> str:
    """Render ROI polygons over a synthetic landmark scatter.

    The scatter carries NO anatomical meaning — it exists only so a human can
    see which landmark indices belong to which ROI and cross-check them
    against MediaPipe's published FaceMesh diagram. Not a placement proof.
    """
    if path is None:
        repo_root = Path(__file__).resolve().parents[2]
        path = str(repo_root / "reports" / "roi_placement.png")
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    pts = np.stack(
        [rng.uniform(0, w, 478), rng.uniform(0, h, 478)], axis=1
    )  # synthetic scatter

    s = 3
    canvas = np.full((h * s + 80, w * s, 3), 255, dtype=np.uint8)
    for x, y in pts:
        cv2.circle(canvas, (int(x * s), int(y * s)), 2, (210, 210, 210), -1)

    colors = {  # BGR
        "forehead": (0, 0, 255),
        "cheek_l": (0, 170, 0),
        "cheek_r": (255, 0, 0),
    }
    for name, idx in ROI_DEFS_P1.items():
        color = colors.get(name, (0, 0, 0))
        poly = (pts[idx] * s).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [poly], True, color, 2)
        for j in idx:
            cv2.putText(
                canvas, str(j),
                (int(pts[j, 0] * s) + 4, int(pts[j, 1] * s) - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA,
            )
        cx, cy = pts[idx].mean(axis=0) * s
        cv2.putText(
            canvas, name, (int(cx) - 28, int(cy)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
        )

    caption = [
        "ROI index visualization on SYNTHETIC scatter (no anatomical meaning).",
        "Cross-check indices vs MediaPipe's published FaceMesh diagram.",
        "Source: raffahitipeuw/health rppg_core.py (validated vs FaceLandmarker).",
    ]
    for k, line in enumerate(caption):
        cv2.putText(
            canvas, line, (10, h * s + 22 + k * 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
        )
    cv2.imwrite(path, canvas)
    return path
