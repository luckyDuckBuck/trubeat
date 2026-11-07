"""docs/13 §6.1: adopt the two-pass face finder only if it finds small faces,
loses no large ones, cuts the same crops where both finders work, and is
fast enough. Runs on the FaceForensics++ test clips already on the laptop
(never committed) and writes reports/facefind.md.

Usage:
    python scripts/check_facefind.py [--manifest runs/faceswap/local_test_manifest.csv]
        [--synthetic-clips 40] [--capture /tmp/gradio/<hash>/truebeat-capture.webm]

Rules (fixed in docs/13 §6.1 before this script ran):
  1. small faces pasted at 30/60/100/200/400 px into 1280x720 and
     1920x1080 scenes: found and landmarked in >= 95% of cases per size;
  2. on the local test reals and fakes: the share of clips with a face in
     >= 12 of 16 sampled frames is >= the round-3 finder's share;
  3. where both find the face, the boxes agree within 5% of the side in
     >= 95% of frames;
  4. detection <= 150 ms per 1080p frame on this CPU.
  (5. the user's extension capture: reported, not a rule; see §6.4.)
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from trubeat import faceswap as fs  # noqa: E402

SIZES = (30, 60, 100, 200, 400)
MIN_FACE_PX = 40   # configs/p6_faceswap.yaml: both finders ignore smaller faces by design
CANVASES = ((1280, 720), (1920, 1080))
REGION = 1.6  # pasted region side = REGION x the face's landmark box side


def wilson(k: int, n: int) -> tuple[float, float]:
    import train_aigen as ta

    return ta.wilson(k, n)


def finders():
    v1 = fs.FaceFinder({"finder": "landmarker"})
    v2 = fs.FaceFinder({"finder": "twopass"})
    return v1, v2


def face_side(box: tuple[float, float, float]) -> float:
    return box[2] / 1.3


def clip_pass(paths: list[str], v1, v2) -> pd.DataFrame:
    """Rules 2-4 on real clips: per frame, did each finder find the face, and
    how far apart are the boxes."""
    rows = []
    for k, path in enumerate(paths):
        s = fs.sample_frames(path, 16, 120.0)
        for idx, rgb in zip(s.index, s.frames):
            b1 = v1.largest_face(rgb)
            t0 = time.perf_counter()
            dets = v2._two.detect(rgb)
            t_det = time.perf_counter() - t0
            b2 = v2.largest_face(rgb)
            agree = None
            if b1 is not None and b2 is not None:
                side = b1[2]
                agree = max(abs(b1[0] - b2[0]), abs(b1[1] - b2[1]), abs(b1[2] - b2[2])) / side
            rows.append({"path": path, "frame": idx, "h": rgb.shape[0], "w": rgb.shape[1],
                         "v1": b1 is not None, "v2": b2 is not None, "agree": agree,
                         "v2_source": v2._two.last_source, "n_dets": len(dets), "det_ms": t_det * 1000})
        if (k + 1) % 25 == 0:
            print(f"  {k + 1}/{len(paths)} clips", flush=True)
    return pd.DataFrame(rows)


def faceless(frame: np.ndarray, cw: int, ch: int, v1, v2) -> np.ndarray | None:
    """``frame`` resized to the canvas with every face blurred out; None when
    a face survives (checked with both finders and a permissive YuNet)."""
    import cv2

    from trubeat import facefind

    det = cv2.FaceDetectorYN.create(str(facefind.resolve_yunet_path()), "", (cw, ch), 0.3, 0.3, 5000)
    bg = cv2.resize(frame, (cw, ch), interpolation=cv2.INTER_AREA)
    fill = bg.reshape(-1, 3).mean(axis=0).astype(np.uint8)   # flat patches: no face can survive them
    for _ in range(3):
        _, faces = det.detect(np.ascontiguousarray(bg[:, :, ::-1]))
        boxes = [] if faces is None else [tuple(float(v) for v in f[:4]) for f in faces]
        lm = v1.largest_face(bg)
        if lm is not None:
            boxes.append((lm[0] - lm[2] / 2, lm[1] - lm[2] / 2, lm[2], lm[2]))
        if not boxes:
            break
        for x, y, w, h in boxes:
            m = 0.6 * max(w, h)
            x0, y0 = max(0, int(x - m)), max(0, int(y - m))
            x1, y1 = min(cw, int(x + w + m)), min(ch, int(y + h + m))
            bg[y0:y1, x0:x1] = fill
    _, faces = det.detect(np.ascontiguousarray(bg[:, :, ::-1]))
    if (faces is not None and len(faces)) or v1.largest_face(bg) is not None or v2.largest_face(bg) is not None:
        return None
    return bg


def synthetic_pass(paths: list[str], v1, v2, rng: np.random.Generator) -> pd.DataFrame:
    """Rule 1: faces of known size pasted into a face-free scene made from
    another clip (its faces blurred out and checked absent)."""
    import cv2

    rows = []
    sources = []
    for path in paths:
        s = fs.sample_frames(path, 3, 120.0)
        frame = s.frames[1]
        box = v1.largest_face(frame)
        if box is not None:
            sources.append((path, frame, box))
    for i, (path, frame, box) in enumerate(sources):
        cx, cy, side = box
        face = face_side(box)
        region = fs.cut_square(frame, cx, cy, face * REGION, max(8, int(round(face * REGION))))
        for cw, ch in CANVASES:
            bg = None
            for j in range(1, len(sources)):      # the next clip whose frame can be made face-free
                bg = faceless(sources[(i + j) % len(sources)][1], cw, ch, v1, v2)
                if bg is not None:
                    break
            if bg is None:
                continue
            for t in SIZES:
                rs = int(round(t * REGION))
                if rs >= min(cw, ch):
                    continue
                patch = cv2.resize(region, (rs, rs), interpolation=cv2.INTER_AREA if rs < region.shape[0]
                                   else cv2.INTER_CUBIC)
                x0 = int(rng.integers(0, cw - rs))
                y0 = int(rng.integers(0, ch - rs))
                canvas = bg.copy()
                canvas[y0:y0 + rs, x0:x0 + rs] = patch
                tx, ty = x0 + rs / 2.0, y0 + rs / 2.0
                for name, f in (("v1", v1), ("v2", v2)):
                    b = f.largest_face(canvas)
                    ok = (b is not None and np.hypot(b[0] - tx, b[1] - ty) <= 0.35 * t
                          and 0.6 * t <= face_side(b) <= 1.6 * t)
                    rows.append({"path": path, "canvas": f"{cw}x{ch}", "size": t, "finder": name, "ok": bool(ok)})
        if (i + 1) % 10 == 0:
            print(f"  synthetic {i + 1}/{len(sources)}", flush=True)
    return pd.DataFrame(rows)


def capture_pass(path: str, v1, v2) -> dict:
    s = fs.sample_frames(path, 16, 120.0)
    n1 = sum(v1.largest_face(f) is not None for f in s.frames)
    n2 = sum(v2.largest_face(f) is not None for f in s.frames)
    seen = []
    for f in s.frames:
        v2.largest_face(f)
        seen.append(v2._two.last_seen)
    dets = [max((d.side for d in v2._two.detect(f)), default=0.0) for f in s.frames]
    return {"frames": len(s.frames), "size": f"{s.width}x{s.height}", "v1": n1, "v2": n2,
            "largest_detection_px": float(np.median(dets))}


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", default=str(REPO / "runs/faceswap/local_test_manifest.csv"))
    ap.add_argument("--synthetic-clips", type=int, default=40)
    ap.add_argument("--capture", default=None, help="the user's extension capture (local only)")
    ap.add_argument("--out", default=str(REPO / "reports/facefind.md"))
    ap.add_argument("--limit", type=int, default=0, help="at most N clips for rules 2-4 (0 = all)")
    a = ap.parse_args()
    df = pd.read_csv(a.manifest)
    df = df[df.path.map(lambda p: Path(p).is_file())].reset_index(drop=True)
    if a.limit:
        df = df.groupby("label", group_keys=False).head(a.limit)
    v1, v2 = finders()
    rng = np.random.default_rng(7)
    t0 = time.time()
    print(f"rules 2-4 on {len(df)} clips", flush=True)
    frames = clip_pass(list(df.path), v1, v2)
    reals = df[df.label == "real"].sample(frac=1.0, random_state=7)
    print(f"rule 1 on {a.synthetic_clips} clips", flush=True)
    syn = synthetic_pass(list(reals.path[: a.synthetic_clips]), v1, v2, rng)
    cap = capture_pass(a.capture, v1, v2) if a.capture and Path(a.capture).is_file() else None

    # rule 1
    piv = syn.groupby(["canvas", "size", "finder"]).ok.agg(["sum", "count"]).reset_index()
    r1_rows, r1 = [], True
    for (canvas, size), g in piv.groupby(["canvas", "size"]):
        d = {r.finder: (int(r["sum"]), int(r["count"])) for _, r in g.iterrows()}
        k2, n2 = d.get("v2", (0, 0))
        k1, n1 = d.get("v1", (0, 0))
        required = size >= MIN_FACE_PX    # docs/13 §6.1 revision: smaller faces are reported only
        ok = n2 > 0 and k2 / n2 >= 0.95
        if required:
            r1 &= ok
        lo, hi = wilson(k2, n2)
        r1_rows.append(f"| {canvas} | {size} px | {k1}/{n1} = {k1 / max(n1, 1):.0%} | "
                       f"{k2}/{n2} = {k2 / max(n2, 1):.0%} [{lo:.0%}, {hi:.0%}] | "
                       f"{('yes' if ok else '**no**') if required else 'reported (< min_face_px)'} |")
    # rule 2
    per_clip = frames.groupby("path").agg(v1=("v1", "sum"), v2=("v2", "sum")).reset_index()
    per_clip = per_clip.merge(df[["path", "label"]], on="path")
    r2_rows, r2 = [], True
    for label, g in per_clip.groupby("label"):
        a1, a2 = int((g.v1 >= 12).sum()), int((g.v2 >= 12).sum())
        b1, b2 = int((g.v1 >= 4).sum()), int((g.v2 >= 4).sum())
        r2 &= a2 >= a1
        r2_rows.append(f"| {label} | {len(g)} | {a1} | {a2} | {b1} | {b2} | {'yes' if a2 >= a1 else '**no**'} |")
    # rule 3
    both = frames.dropna(subset=["agree"])
    k3 = int((both.agree <= 0.05).sum())
    r3 = len(both) > 0 and k3 / len(both) >= 0.95
    # rule 4
    big = frames[frames.w * frames.h >= 1280 * 720]
    det_1080 = float(big.det_ms.mean()) if len(big) else float("nan")
    det_all = float(frames.det_ms.mean())
    r4 = (det_1080 if len(big) else det_all) <= 150.0
    adopt = r1 and r2 and r3 and r4

    sizes = frames.groupby(["w", "h"]).size().sort_values(ascending=False).head(4)
    L = [f"# Face finder v2: adoption check (docs/13 §6.1) — {time.strftime('%Y-%m-%d %H:%M')}", "",
         (f"Data: {len(df)} FaceForensics++ c23 test clips on this laptop ({int((df.label == 'real').sum())} real, "
          f"{int((df.label == 'fake').sum())} fake), 16 sampled frames each ({len(frames)} frames; most common sizes: "
          + ", ".join(f"{w}x{h} ({n})" for (w, h), n in sizes.items()) + "). Round-3 finder = the Face Landmarker on "
          "the whole frame; two-pass = that same pass first, and only when it finds nothing, YuNet on a <= 1280 px copy "
          "followed by the Landmarker on a 2x crop of each detection. Runtime "
          f"{(time.time() - t0) / 60:.0f} min."), "",
         f"**Decision: {'adopt' if adopt else 'do not adopt'} the two-pass finder** "
         f"(rule 1 {'pass' if r1 else 'fail'}, 2 {'pass' if r2 else 'fail'}, 3 {'pass' if r3 else 'fail'}, "
         f"4 {'pass' if r4 else 'fail'}).", "",
         "## Rule 1: small faces pasted into a scene (found and landmarked)", "",
         (f"{syn.path.nunique()} faces from distinct test reals, each pasted as a region {REGION}x its landmark box "
          "into a frame of another clip resized to the canvas, with that frame's own faces filled flat and checked "
          "absent (both finders and YuNet at score 0.3 find nothing); success = the returned box centre within "
          "0.35 x the face size of the true centre and its face size within 0.6-1.6 x. Wilson 95% intervals."), "",
         "| canvas | face | round-3 finder | two-pass | pass (>= 95%) |", "|---|---|---|---|---|", *r1_rows, "",
         "## Rule 2: no regression on the test clips", "",
         "| label | clips | round-3: face in >= 12/16 | two-pass: >= 12/16 | round-3: >= 4/16 | two-pass: >= 4/16 | pass |",
         "|---|---|---|---|---|---|---|", *r2_rows, "",
         "## Rule 3: same crop box where both find the face", "",
         (f"Frames where both found a face: {len(both)}; boxes within 5% of the side (max of centre and side "
          f"differences): **{k3}/{len(both)} = {k3 / max(len(both), 1):.1%}** (rule: >= 95%) -> "
          f"{'pass' if r3 else '**fail**'}. Median difference {float(both.agree.median()) if len(both) else float('nan'):.3f} "
          "of the side."), "",
         f"Two-pass source of the returned face: " + ", ".join(
             f"{k or 'none'} {v}" for k, v in frames.v2_source.value_counts().items()) + ".", "",
         "## Rule 4: cost", "",
         (f"YuNet detection on the sampled frames: mean {det_all:.0f} ms per frame over all sizes; "
          f"{det_1080:.0f} ms on frames of at least 1280x720 ({len(big)} frames). Rule: <= 150 ms per 1080p frame "
          f"-> {'pass' if r4 else '**fail**'}."), ""]
    if cap is not None:
        L += ["## Item 5 (reported, not a rule): the user's extension capture", "",
              (f"The capture as recorded (the whole tab at Chrome's default size, {cap['size']}), {cap['frames']} "
               f"sampled frames: round-3 finder {cap['v1']}/{cap['frames']}, two-pass {cap['v2']}/{cap['frames']} "
               f"(a face in >= 12 of 16: {'yes' if cap['v2'] >= 12 else 'no'}); median largest YuNet detection "
               f"{cap['largest_detection_px']:.0f} px (faces under `min_face_px`, 40 px, are never used). The "
               "extension's own changes (full-resolution capture, crop to the player) are tested in docs/13 §6.4."),
              ""]
    Path(a.out).write_text("\n".join(L) + "\n")
    frames.to_csv(Path(a.out).with_suffix(".frames.csv"), index=False)
    syn.to_csv(Path(a.out).with_suffix(".synthetic.csv"), index=False)
    print("\n".join(L))


if __name__ == "__main__":
    main()
