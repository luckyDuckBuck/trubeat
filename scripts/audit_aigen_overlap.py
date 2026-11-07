"""Re-measure the shipped AI-video detector without external clips that are
near-copies of its training clips (docs/13 B-03, found 2026-11-07).

The external set (GenBuster-Bench) and the training pool (AIGVDBench,
Rapidata, FF++ originals) were curated by different groups, so the
group-key leakage check cannot see a clip that both benchmarks
redistribute. This script compares every external clip's mean DINOv2
embedding with every pool clip's (the same 384 features the model scores),
lists the external clips at cosine >= --threshold with a pool clip, and
recomputes the external tables of reports/aigen_detector_v2.md with and
without them, using the shipped bundle and its threshold unchanged (no
retraining, no new threshold).

Usage:
    python scripts/audit_aigen_overlap.py [--features runs/aigen_features_v2.pkl]
        [--bundle src/trubeat/models/aigen.joblib] [--threshold 0.99]
        [--out reports/aigen_overlap.md]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import train_aigen as ta


def unit(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float64)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def nearest_pool(ext: pd.DataFrame, pool: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    E, P = unit(ext[cols].to_numpy()), unit(pool[cols].to_numpy())
    best, arg = np.full(len(E), -np.inf), np.zeros(len(E), int)
    for i in range(0, len(P), 2048):
        S = E @ P[i:i + 2048].T
        j = S.argmax(axis=1)
        v = S[np.arange(len(E)), j]
        upd = v > best
        best[upd], arg[upd] = v[upd], j[upd] + i
    return best, arg


def summary(ext: pd.DataFrame) -> dict:
    out = {}
    gb_real = ext[(ext.label == "real") & ext.source.str.startswith("genbuster")]
    ff_real = ext[ext.source == "ffpp_original"]
    for name, reals in (("genbuster_real", gb_real), ("ffpp_real", ff_real)):
        k, n = int(reals.flagged.sum()), len(reals)
        out[f"fpr_{name}"] = {"k": k, "n": n, "rate": k / max(n, 1), "ci": list(ta.wilson(k, n))}
    for tier in ("genbuster_id", "genbuster_ood", "genbuster_wild"):
        fakes = ext[(ext.source == tier) & (ext.label == "ai")]
        k = int(fakes.flagged.sum())
        both = pd.concat([gb_real, fakes])
        y = (both.label == "ai").to_numpy().astype(int)
        out[f"recall_{tier.split('_')[1]}"] = {"k": k, "n": len(fakes), "rate": k / max(len(fakes), 1),
                                              "ci": list(ta.wilson(k, len(fakes))),
                                              "auc": ta.safe_auc(y, both.score.to_numpy())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--features", default=str(REPO / "runs/aigen_features_v2.pkl"))
    ap.add_argument("--bundle", default=str(REPO / "src/trubeat/models/aigen.joblib"))
    ap.add_argument("--threshold", type=float, default=0.99, help="cosine at or above which a clip is a near-copy")
    ap.add_argument("--out", default=str(REPO / "reports/aigen_overlap.md"))
    a = ap.parse_args()
    d = pd.read_pickle(a.features)
    b = joblib.load(a.bundle)
    cols = list(b["feature_columns"])
    pool = d[d.role == "pool"].reset_index(drop=True)
    ext = d[d.role == "external"].reset_index(drop=True)
    s = ta.ai_score(b["model"], ext[cols].to_numpy())
    ext = ext.assign(score=s, flagged=s >= float(b["threshold"]))
    best, arg = nearest_pool(ext, pool, cols)
    ext = ext.assign(nearest_cos=best, nearest_pool=pool.path.to_numpy()[arg],
                     nearest_source=pool.source.to_numpy()[arg], nearest_generator=pool.generator.to_numpy()[arg])
    dup = ext.nearest_cos >= a.threshold
    before, after = summary(ext), summary(ext[~dup])
    hits = ext[dup]
    bsha = hashlib.sha256(Path(a.bundle).read_bytes()).hexdigest()

    def row(key: str, label: str) -> str:
        x, y = before[key], after[key]
        auc = (f" | {x['auc']:.3f} → {y['auc']:.3f}" if "auc" in x else " | —")
        return (f"| {label} | {x['k']}/{x['n']} = {x['rate']:.3f} [{x['ci'][0]:.3f}, {x['ci'][1]:.3f}] | "
                f"{y['k']}/{y['n']} = {y['rate']:.3f} [{y['ci'][0]:.3f}, {y['ci'][1]:.3f}]{auc} |")

    by = hits.groupby(["source", "label", "generator"]).size().reset_index(name="n")
    L = [f"# AI-video detector: external clips that are near-copies of training clips — {time.strftime('%Y-%m-%d %H:%M')}",
         "",
         (f"Bundle `{Path(a.bundle).name}` (SHA-256 `{bsha[:12]}…`, threshold {float(b['threshold']):+.4f}, unchanged). "
          f"Feature table `{Path(a.features).name}`: {len(pool)} pool (training) clips, {len(ext)} external clips. "
          f"A clip is a near-copy when its mean DINOv2 embedding (the {len(cols)} features the model scores) has "
          f"cosine ≥ {a.threshold} with a pool clip."), "",
         (f"**{int(dup.sum())} external clips are near-copies** (cosine median {hits.nearest_cos.median():.4f}, "
          f"min {hits.nearest_cos.min():.4f}). The leakage check by group key could not see them: GenBuster-Bench "
          "and AIGVDBench name their clips differently. Of the other external clips, the most similar pool clip "
          f"has cosine {ext[~dup].nearest_cos.max():.4f} at most (median {ext[~dup].nearest_cos.median():.4f})."), "",
         "| source | label | generator | near-copies |", "|---|---|---|---|"]
    L += [f"| {r.source} | {r.label} | {r.generator} | {r.n} |" for r in by.itertuples(index=False)]
    L += ["", "## External numbers, as reported and without the near-copies", "",
          "| metric | as reported (all external clips) | without near-copies | AUC vs GenBuster reals |",
          "|---|---|---|---|",
          row("fpr_genbuster_real", "false alarms, GenBuster reals"),
          row("fpr_ffpp_real", "false alarms, FF++ c23 test reals"),
          row("recall_id", "recall, GenBuster in-domain tier"),
          row("recall_ood", "recall, GenBuster out-of-domain tier (2024-era tools)"),
          row("recall_wild", "recall, GenBuster wild tier (the newest)"), "",
          ("Rates are over the clips scored; intervals are Wilson 95%. The model, its threshold and every clip's "
           "score are unchanged; only the near-copies are left out of the denominators.")]
    Path(a.out).write_text("\n".join(L) + "\n")
    hits[["path", "source", "label", "generator", "nearest_cos", "nearest_pool", "nearest_source",
          "nearest_generator", "score", "flagged"]].to_csv(Path(a.out).with_suffix(".csv"), index=False)
    Path(a.out).with_suffix(".json").write_text(json.dumps(
        {"bundle_sha256": bsha, "threshold_cos": a.threshold, "n_near_copies": int(dup.sum()),
         "before": before, "after": after, "created": time.strftime("%Y-%m-%d %H:%M")}, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main()
