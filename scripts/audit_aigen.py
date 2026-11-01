"""Second-round audit of the AI-video detector (docs/11 §9). Local only.

Checks that a good-looking evaluation is not an artefact:

1. Metadata shortcut: can trivial container facts (fps, width, height,
   duration, bitrate) separate AI from real clips in the pool, scored the
   same leave-one-generator-out way as the detector? A high number means
   the public datasets carry giveaways the detector *could* be exploiting.
2. Near-duplicate leakage: external (GenBuster) clips whose mean embedding
   is near-identical to a pool clip's (cosine >= --dup-cos), i.e. the same
   source content on both sides of the evaluation.
3. External results with duplicates removed, against the report's numbers.
4. Face-swap coverage: the AI-video detector's flag rate on FF++ c23 test
   fakes per manipulation family (does it also catch face edits?), next
   to its rate on FF++ test reals.

Reads runs/aigen_features.pkl and runs/aigen_detector.joblib written by
scripts/train_aigen.py.

Usage:
    python scripts/audit_aigen.py [--dup-cos 0.97]
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import train_aigen as ta

from trubeat import aigen


def container_meta(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        n = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        w = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0)
        h = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0)
    finally:
        cap.release()
    dur = n / fps if fps > 0 else 0.0
    size = Path(path).stat().st_size
    return {"m_fps": fps, "m_w": w, "m_h": h, "m_aspect": w / h if h else 0.0,
            "m_dur": dur, "m_kbps": (size * 8 / 1000 / dur) if dur > 0 else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--features", default=str(REPO / "runs/aigen_features.pkl"))
    ap.add_argument("--model", default=str(REPO / "runs/aigen_detector.joblib"))
    ap.add_argument("--config", default=str(REPO / "configs/p5_aigen.yaml"))
    ap.add_argument("--dup-cos", type=float, default=0.97)
    ap.add_argument("--out", default=str(REPO / "reports/aigen_audit.md"))
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    data = pd.read_pickle(args.features)
    bundle = joblib.load(args.model)
    cols, thr = bundle["feature_columns"], float(bundle["threshold"])
    cfg = aigen.merged_config(yaml.safe_load(Path(args.config).read_text()))
    pool = data[data.role == "pool"].reset_index(drop=True)
    ext = data[data.role == "external"].reset_index(drop=True)
    L = ["# AI-video detector: second-round audit", ""]

    # 1. metadata shortcut (original files, before normalisation)
    meta = pd.DataFrame([container_meta(p) for p in pool.path])
    mpool = pd.concat([pool, meta], axis=1)
    mcols = list(meta.columns)
    lg = ta.logo_eval(mpool, mcols, C=1.0)
    L += ["## 1. Metadata shortcut (fps, size, aspect, duration, bitrate of the original files)", "",
          (f"Macro leave-one-generator-out AUC from metadata alone: **{lg.auc.mean():.3f}** "
           f"(worst generator {lg.auc.min():.3f}). The detector never sees these values "
           "(every clip is re-encoded to one format and centre-cropped first), but a high "
           "number means the datasets differ in ways unrelated to generation."), ""]
    per = pool.assign(**meta).groupby("label")[mcols].median().round(2)
    L += ["| label | " + " | ".join(mcols) + " |", "|---" * (len(mcols) + 1) + "|"]
    L += [f"| {lab} | " + " | ".join(str(v) for v in row) + " |" for lab, row in per.iterrows()]

    # 2. near-duplicates between pool and external (mean-embedding cosine)
    ecols = [c for c in cols if c.startswith("emb_")] or [
        c for c in data.columns if c.startswith("emb_")]
    P = np.array(pool[ecols], dtype=np.float32)  # writable copy
    E = np.array(ext[ecols], dtype=np.float32)
    P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-8
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-8
    best = np.max(E @ P.T, axis=1)
    ext = ext.assign(max_cos_to_pool=best, dup=best >= args.dup_cos)
    by = ext.groupby(["source", "label"]).dup.agg(["sum", "count"])
    L += ["", f"## 2. Near-duplicate content, external vs pool (cosine >= {args.dup_cos})", "",
          "| external source | label | near-duplicates | clips |", "|---|---|---|---|"]
    L += [f"| {s} | {lab} | {int(r['sum'])} | {int(r['count'])} |" for (s, lab), r in by.iterrows()]

    # 3. external results without the duplicates
    final = bundle["model"]
    for name, sub in (("all external", ext), ("near-duplicates removed", ext[~ext.dup])):
        s = ta.ai_score(final, sub[cols].to_numpy())
        res = ta.compact_eval(sub.assign(score=s, flagged=s >= thr))
        L += ["", f"## 3. External results: {name} (n={len(sub)})", "",
              "| measure | value |", "|---|---|"]
        L += [f"| {k} | {v:.3f} |" for k, v in sorted(res.items())]

    # 4. face-swap coverage: FF++ c23 test fakes through the AI-video detector
    ff = pd.read_csv(REPO / "data/manifests/ffpp_c23_qc.csv")
    ff = ff[(ff.split == "test") & ff.usable.astype(bool)]
    ff = ta.cap_rows(ff.assign(source="ffpp", generator=ff.video_id.str.split("_").str[1]),
                     40, by=("generator", "label"))
    norm = ta.normalize_all(ff.path.tolist(), cfg["normalize"], workers=4)
    feats = ta.featurize_all([norm[p] for p in ff.path if p in norm], cfg, int(cfg["workers"]))
    feats = feats[feats.error == ""]
    ff = ff.assign(feat_path=ff.path.map(norm)).merge(feats, on="feat_path")
    s = ta.ai_score(final, ff[cols].to_numpy())
    ff = ff.assign(flagged=s >= thr)
    L += ["", "## 4. FF++ c23 test clips through the AI-video detector (flag rate)", "",
          "| family | n | flagged as AI |", "|---|---|---|"]
    L += [f"| {g} | {len(x)} | {x.flagged.mean():.3f} |" for g, x in ff.groupby("generator")]
    Path(args.out).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
