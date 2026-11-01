"""Try richer clip descriptors for the AI-video detector from CACHED per-frame
embeddings (docs/11 §9, round 2). Local only; no backbone run.

The shipped model describes a clip by the mean of its 16 frame embeddings.
Candidates add temporal statistics per embedding dimension:

  mean        mean over frames (the shipped descriptor; reproduces 0.809)
  mean+std    plus the per-dimension standard deviation over frames
  mean+diff   plus the mean absolute frame-to-frame change
  all         mean + std + diff

Decision rule, fixed before any number is seen (selection uses the pool
only; the external set is reported, never used to choose):
  adopt a candidate only if its pool macro leave-one-family-out AUC beats
  the shipped descriptor's by >= 0.01 AND its worst-family AUC is no more
  than 0.02 below the shipped one's. Otherwise keep the shipped model.

Usage:
    python scripts/explore_aigen_features.py [--out reports/aigen_feature_search.md]
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import train_aigen as ta

from trubeat import aigen

DESCRIPTORS = ("mean", "mean+std", "mean+diff", "all")


def frame_stats(emb: np.ndarray) -> dict[str, np.ndarray]:
    """(S, N, D) cached embeddings -> per-dimension statistics (segment 0;
    training uses one segment per clip)."""
    e = emb[0]
    return {"mean": e.mean(axis=0), "std": e.std(axis=0),
            "diff": np.abs(np.diff(e, axis=0)).mean(axis=0)}


def load_stats(paths: list[str], cfg: dict) -> dict[str, np.ndarray]:
    out = {"mean": [], "std": [], "diff": []}
    for i, p in enumerate(paths, 1):
        f = Path(cfg["cache_dir"]) / f"{aigen.cache_key(Path(p), cfg)}.npz"
        with np.load(f, allow_pickle=False) as z:
            st = frame_stats(z["emb"])
        for k, v in st.items():
            out[k].append(v)
        if i % 2000 == 0:
            ta.log(f"  loaded {i}/{len(paths)}")
    return {k: np.stack(v).astype(np.float32) for k, v in out.items()}


def with_descriptor(df: pd.DataFrame, stats: dict[str, np.ndarray], name: str
                    ) -> tuple[pd.DataFrame, list[str]]:
    parts = {"mean": ["mean"], "mean+std": ["mean", "std"],
             "mean+diff": ["mean", "diff"], "all": ["mean", "std", "diff"]}[name]
    blocks, cols = [], []
    for part in parts:
        blocks.append(stats[part])
        cols += [f"{part}_{i}" for i in range(stats[part].shape[1])]
    X = pd.DataFrame(np.hstack(blocks), columns=cols, index=df.index)
    return pd.concat([df.drop(columns=[c for c in df.columns if c.startswith("emb_")]), X],
                     axis=1), cols


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--features", default=str(REPO / "runs/aigen_features.pkl"))
    ap.add_argument("--model", default=str(REPO / "runs/aigen_detector.joblib"))
    ap.add_argument("--config", default=str(REPO / "configs/p5_aigen.yaml"))
    ap.add_argument("--out", default=str(REPO / "reports/aigen_feature_search.md"))
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    cfg = aigen.merged_config(yaml.safe_load(Path(args.config).read_text()))
    cfg["cache_dir"] = str(REPO / cfg["cache_dir"])
    cfg["max_segments"] = int(cfg.get("train_segments", 1))  # as featurize_all
    data = pd.read_pickle(args.features).reset_index(drop=True)
    shipped = joblib.load(args.model)
    ta.log(f"loading cached frame embeddings for {len(data)} clips")
    stats = load_stats(data.feat_path.tolist(), cfg)
    # sanity: the cached mean must equal the descriptor the model was trained on
    emb_cols = [f"emb_{i}" for i in range(stats["mean"].shape[1])]
    gap = float(np.abs(stats["mean"] - data[emb_cols].to_numpy(dtype=np.float32)).max())
    if gap > 1e-4:
        raise SystemExit(f"cache does not match the saved features (max gap {gap})")

    rows = []
    for name in DESCRIPTORS:
        d, cols = with_descriptor(data, stats, name)
        pool = d[d.role == "pool"].reset_index(drop=True)
        for C in (0.003, 0.01, 0.1):
            lg = ta.logo_eval(pool, cols, C)
            rows.append({"descriptor": name, "C": C, "macro": lg.auc.mean(), "worst": lg.auc.min(),
                         "dims": len(cols)})
            ta.log(f"  {name:10s} C={C:<6} macro {lg.auc.mean():.4f} worst {lg.auc.min():.3f}")
    res = pd.DataFrame(rows)
    base = res[(res.descriptor == "mean") & (res.C == float(shipped["model"][-1].C))].iloc[0]
    best = res.sort_values("macro", ascending=False).iloc[0]
    adopt = (best.descriptor != "mean" and best.macro >= base.macro + 0.01
             and best.worst >= base.worst - 0.02)

    L = [f"# AI-video detector: descriptor search — {time.strftime('%Y-%m-%d %H:%M')}", "",
         ("Selection on the pool only (macro leave-one-generator-family-out AUC), from cached "
          "frame embeddings. Rule fixed in advance (script docstring): adopt only if macro AUC "
          ">= shipped + 0.01 and worst family >= shipped - 0.02."), "",
         "| descriptor | C | dims | macro LOGO AUC | worst family |", "|---|---|---|---|---|"]
    L += [f"| {r.descriptor} | {r.C} | {r.dims} | {r.macro:.4f} | {r.worst:.3f} |"
          for r in res.itertuples()]
    L += ["", (f"Shipped: mean, C={base.C}: {base.macro:.4f} (worst {base.worst:.3f}). "
               f"Best: {best.descriptor}, C={best.C}: {best.macro:.4f} (worst {best.worst:.3f}). "
               f"**Rule says: {'adopt ' + best.descriptor if adopt else 'keep the shipped model'}.**")]

    if adopt:  # external numbers for the adopted candidate, next to the shipped ones
        d, cols = with_descriptor(data, stats, best.descriptor)
        pool = d[d.role == "pool"].reset_index(drop=True)
        ext = d[d.role == "external"].reset_index(drop=True)
        m = ta.make_model(float(best.C)).fit(pool[cols].to_numpy(), pool.label.to_numpy())
        oof = ta.oof_scores(pool, cols, float(best.C))
        thr = float(np.quantile(oof[(pool.label == "real").to_numpy()], 1 - float(cfg["target_fpr"])))
        s = ta.ai_score(m, ext[cols].to_numpy())
        new = ta.compact_eval(ext.assign(score=s, flagged=s >= thr))
        e0 = data[data.role == "external"].reset_index(drop=True)
        s0 = ta.ai_score(shipped["model"], e0[shipped["feature_columns"]].to_numpy())
        old = ta.compact_eval(e0.assign(score=s0, flagged=s0 >= float(shipped["threshold"])))
        L += ["", "## External (reported, not used to choose)", "",
              "| measure | shipped | " + best.descriptor + " |", "|---|---|---|"]
        L += [f"| {k} | {old[k]:.3f} | {new[k]:.3f} |" if isinstance(old[k], float)
              else f"| {k} | {old[k]} | {new[k]} |" for k in sorted(old)]
        joblib.dump({"descriptor": best.descriptor, "C": float(best.C), "threshold": thr,
                     "columns": cols}, REPO / "runs/aigen_descriptor_choice.joblib")
    Path(args.out).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
