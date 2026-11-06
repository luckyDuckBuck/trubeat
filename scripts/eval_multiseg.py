"""Score the whole clip, not its first 2 s (docs/11 §9.6 P2). Local only.

The shipped bundle scores one 2 s segment (the first). This script embeds
up to `--segments` evenly spaced segments per clip (segment 0 is the one
already cached, so only the extra segments run through the backbone) and
compares three ways of describing a clip, all with the shipped model's
recipe (StandardScaler + logistic regression, C from the bundle):

  first      train on segment 0, score segment 0        (shipped)
  avg        train on segment 0, score the mean of all segments
  avg-train  train on the mean of all segments, score the same

For a linear model on mean embeddings, averaging segment scores equals
scoring the mean embedding over all sampled frames, so "avg" is exactly
"average the per-segment scores".

Decision rule, fixed before any number is seen. Selection between avg and
avg-train uses the pool only (macro leave-one-generator-family-out AUC);
the threshold is re-set on out-of-fold pool reals scored the same way, at
the same 5% target. The chosen variant replaces "first" only if all hold:

  1. its pool macro LOGO AUC >= first's (no loss on unseen generators);
  2. docs/11 §9.6 P2 gate, external, at its own threshold: GenBuster-real
     FPR <= 6% and recall on every GenBuster tier >= first's;
  3. FF++ c23 test reals flagged <= 2/135 (as docs/11 §9.7).

The external set gates adoption; it never chooses between candidates.
Segment counts per label are reported because they differ by clip length
(5 s AI clips fit 2 segments, 6 s reals fit 3).

Usage:
    python scripts/eval_multiseg.py [--segments 3] [--workers 3]
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupKFold

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import train_aigen as ta

from trubeat import aigen
from trubeat.errors import TrueBeatError

_EMB: aigen.Embedder | None = None
_CFG1: dict = {}
_CFGK: dict = {}
OUT_DIR = REPO / "data/processed/aigen_emb_multiseg"


def _init(cfg1: dict, cfgk: dict) -> None:
    global _EMB, _CFG1, _CFGK
    warnings.filterwarnings("ignore")
    _CFG1, _CFGK = cfg1, cfgk
    _EMB = aigen.Embedder(cfgk)


def segment_means(path: str) -> tuple[str, np.ndarray | None, str]:
    """(S, D) per-segment mean embeddings, cached by the K-segment key."""
    p = Path(path)
    out = OUT_DIR / f"{aigen.cache_key(p, _CFGK)}.npy"
    if out.is_file():
        return path, np.load(out), ""
    try:
        frames, _ = aigen.sample_segments(p, _CFGK)
        with np.load(Path(_CFG1["cache_dir"]) / f"{aigen.cache_key(p, _CFG1)}.npz",
                      allow_pickle=False) as z:
            seg0 = z["emb"][0]  # (N, D), segment 0 from the training run
        means = [seg0.mean(axis=0)]
        for s in range(1, frames.shape[0]):
            means.append(_EMB.embed(frames[s]).mean(axis=0))
        arr = np.stack(means).astype(np.float32)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        np.save(out, arr)
        return path, arr, ""
    except TrueBeatError as exc:
        return path, None, exc.code
    except Exception as exc:  # noqa: BLE001 - one bad clip must not stop the run
        return path, None, f"E_UNKNOWN: {exc}"


def check_segment0(paths: list[str], cfg1: dict, cfgk: dict, n: int = 4) -> float:
    """Segment 0 of the K-segment sampling must be the cached training
    segment: re-embed it for a few clips and return the worst cosine."""
    emb = aigen.Embedder(cfgk)
    worst = 1.0
    for p in paths[:n]:
        frames, _ = aigen.sample_segments(Path(p), cfgk)
        with np.load(Path(cfg1["cache_dir"]) / f"{aigen.cache_key(Path(p), cfg1)}.npz",
                     allow_pickle=False) as z:
            cached = z["emb"][0].mean(axis=0)
        fresh = emb.embed(frames[0]).mean(axis=0)
        cos = float(cached @ fresh / (np.linalg.norm(cached) * np.linalg.norm(fresh)))
        worst = min(worst, cos)
    return worst


def logo_scores(pool: pd.DataFrame, train_X: np.ndarray, test_X: np.ndarray, C: float,
                n_folds: int = 5) -> tuple[float, float]:
    """ta.logo_eval with separate train / test descriptors (same folds)."""
    reals = pool[pool.label == "real"]
    fold_of = {}
    for k, (_, te) in enumerate(GroupKFold(n_folds).split(reals, groups=reals.split_group)):
        for g in reals.iloc[te].split_group:
            fold_of[g] = k
    fam = pool.generator.map(ta.base_family)
    rf = pool.split_group.map(fold_of)
    aucs = []
    for i, gen in enumerate(sorted(fam[pool.label == "ai"].unique())):
        k = i % n_folds
        te = ((pool.label == "real") & (rf == k)) | ((pool.label == "ai") & (fam == gen))
        blocked = set(pool[te].split_group)
        tr = (((pool.label == "real") & (rf != k)) | ((pool.label == "ai") & (fam != gen))) \
            & ~pool.split_group.isin(blocked)
        m = ta.make_model(C).fit(train_X[tr.to_numpy()], pool.label[tr].to_numpy())
        s = ta.ai_score(m, test_X[te.to_numpy()])
        aucs.append(ta.safe_auc((pool.label[te] == "ai").to_numpy().astype(int), s))
    return float(np.nanmean(aucs)), float(np.nanmin(aucs))


def oof_real_scores(pool: pd.DataFrame, train_X, test_X, C: float) -> np.ndarray:
    s = np.full(len(pool), np.nan)
    for tr, te in GroupKFold(5).split(pool, groups=pool.split_group):
        m = ta.make_model(C).fit(train_X[tr], pool.label.iloc[tr].to_numpy())
        s[te] = ta.ai_score(m, test_X[te])
    return s[(pool.label == "real").to_numpy()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--features", default=str(REPO / "runs/aigen_features.pkl"))
    ap.add_argument("--model", default=str(REPO / "runs/aigen_detector.joblib"))
    ap.add_argument("--config", default=str(REPO / "configs/p5_aigen.yaml"))
    ap.add_argument("--segments", type=int, default=3)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default=str(REPO / "reports/aigen_multiseg.md"))
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    cfg = aigen.merged_config(yaml.safe_load(Path(args.config).read_text()))
    cfg["cache_dir"] = str(REPO / cfg["cache_dir"])
    cfg1 = dict(cfg, max_segments=int(cfg.get("train_segments", 1)))
    cfgk = dict(cfg, max_segments=int(args.segments),
                torch_threads=max(1, int(cfg["train_threads"]) // args.workers))
    data = pd.read_pickle(args.features).reset_index(drop=True)
    bundle = joblib.load(args.model)
    C = float(bundle["model"][-1].C)
    paths = data.feat_path.tolist()

    worst = check_segment0(paths, cfg1, cfgk)
    ta.log(f"segment-0 check: worst cosine {worst:.6f}")
    if worst < 0.9999:
        raise SystemExit("segment 0 of the K-segment sampling is not the cached training segment")

    ta.log(f"embedding up to {args.segments} segments for {len(paths)} clips")
    got: dict[str, np.ndarray] = {}
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(cfg1, cfgk)) as pool:
        for i, (p, arr, err) in enumerate(pool.imap_unordered(segment_means, paths, chunksize=4)):
            if arr is not None:
                got[p] = arr
            if (i + 1) % 250 == 0:
                ta.log(f"  {i + 1}/{len(paths)} ({(i + 1) / (time.time() - t0):.2f} clips/s)")
    data = data[data.feat_path.isin(got)].reset_index(drop=True)
    emb_cols = [f"emb_{i}" for i in range(384)]
    first = data[emb_cols].to_numpy(dtype=np.float64)
    avg = np.stack([got[p].mean(axis=0) for p in data.feat_path]).astype(np.float64)
    nseg = np.array([got[p].shape[0] for p in data.feat_path])
    # sanity: segment 0 of the multi-segment set is the saved descriptor
    s0 = np.stack([got[p][0] for p in data.feat_path])
    if np.abs(s0 - first).max() > 1e-4:
        raise SystemExit("segment-0 means differ from the saved features")

    is_pool = (data.role == "pool").to_numpy()
    pool, ext = data[is_pool].reset_index(drop=True), data[~is_pool].reset_index(drop=True)
    variants = {"first": (first, first), "avg": (first, avg), "avg-train": (avg, avg)}
    rows, ext_res, fitted = [], {}, {}
    for name, (trX, teX) in variants.items():
        macro, worst_fam = logo_scores(pool, trX[is_pool], teX[is_pool], C)
        oof = oof_real_scores(pool, trX[is_pool], teX[is_pool], C)
        thr = float(np.quantile(oof, 1 - float(cfg["target_fpr"])))
        m = ta.make_model(C).fit(trX[is_pool], pool.label.to_numpy())
        fitted[name] = (m, thr, np.sort(oof).astype(np.float32))
        s = ta.ai_score(m, teX[~is_pool])
        ext_res[name] = ta.compact_eval(ext.assign(score=s, flagged=s >= thr))
        rows.append({"variant": name, "macro": macro, "worst": worst_fam, "threshold": thr})
        ta.log(f"  {name:9s} LOGO macro {macro:.4f} worst {worst_fam:.3f} thr {thr:+.3f}")
    res = pd.DataFrame(rows).set_index("variant")
    cands = res.drop(index="first")
    best = cands.macro.idxmax()
    e0, eb = ext_res["first"], ext_res[best]
    ff = ext.source == "ffpp_original"
    ff_X = variants[best][1][~is_pool][ff.to_numpy()]
    ff_flags = int((ta.ai_score(fitted[best][0], ff_X) >= fitted[best][1]).sum())
    gates = [
        ("pool macro LOGO AUC >= first's", cands.macro[best] >= res.macro["first"],
         f"{cands.macro[best]:.4f} vs {res.macro['first']:.4f}"),
        ("GenBuster-real FPR <= 6%", eb["fpr_genbuster_real"] <= 0.06,
         f"{eb['fpr_genbuster_real']:.3f} (first {e0['fpr_genbuster_real']:.3f})"),
        *[(f"{tier} recall >= first's", eb[f"recall_{tier}"] >= e0[f"recall_{tier}"],
           f"{eb[f'recall_{tier}']:.3f} vs {e0[f'recall_{tier}']:.3f}") for tier in ("id", "ood", "wild")],
        ("FF++ c23 test reals flagged <= 2", ff_flags <= 2, f"{ff_flags}/{int(ff.sum())}"),
    ]
    adopt = all(ok for _, ok, _ in gates)

    seg_hist = pd.Series(nseg).value_counts().sort_index()
    seg_by_label = pd.crosstab(data.label, nseg)
    L = [f"# AI-video detector: whole-clip scoring — {time.strftime('%Y-%m-%d %H:%M')}", "",
         (f"Up to {args.segments} evenly spaced 2 s segments per clip; segments per clip: "
          + ", ".join(f"{k}: {v}" for k, v in seg_hist.items()) + ". Model recipe and C "
          f"({C}) from the shipped bundle. Rule fixed in advance (script docstring): the "
          "best of avg / avg-train on the pool replaces first only if every gate holds."), "",
         "## Pool (selection)", "", "| variant | macro LOGO AUC | worst family | threshold (OOF reals, 5%) |",
         "|---|---|---|---|"]
    L += [f"| {v} | {r.macro:.4f} | {r.worst:.3f} | {r.threshold:+.4f} |" for v, r in res.iterrows()]
    L += ["", f"Best candidate on the pool: **{best}**.", "", "| gate | pass | value |", "|---|---|---|"]
    L += [f"| {g} | {'yes' if ok else '**no**'} | {v} |" for g, ok, v in gates]
    L += ["", f"**Rule says: {'adopt ' + best if adopt else 'keep first'}.**", "",
          "## Segments per clip, by label (all clips)", "",
          "| label | " + " | ".join(f"{k} seg" for k in seg_by_label.columns) + " |",
          "|---" * (len(seg_by_label.columns) + 1) + "|"]
    L += [f"| {lab} | " + " | ".join(str(int(v)) for v in r) + " |" for lab, r in seg_by_label.iterrows()]
    L += ["",
          "## External (gates adoption; never chooses between candidates)", "",
          "| measure | " + " | ".join(variants) + " |", "|---" * (len(variants) + 1) + "|"]
    keys = sorted(ext_res["first"])
    for k in keys:
        vals = [ext_res[v][k] for v in variants]
        L.append(f"| {k} | " + " | ".join(str(x) if isinstance(x, int) else f"{x:.3f}"
                                          for x in vals) + " |")
    Path(args.out).write_text("\n".join(L) + "\n")
    joblib.dump({"result": res, "external": ext_res, "adopt": bool(adopt), "best": best, "gates": gates,
                 "segments": int(args.segments)}, REPO / "runs/aigen_multiseg.joblib")
    if adopt:  # candidate bundle: same contract as train_aigen's, K segments scored
        m, thr, ref = fitted[best]
        nb = dict(bundle)
        nb.update(model=m, threshold=thr, real_reference_scores=ref,
                  created=time.strftime("%Y-%m-%d %H:%M"),
                  selection=(f"{bundle.get('selection')}; whole-clip scoring '{best}' over "
                             f"{args.segments} segments adopted by scripts/eval_multiseg.py"))
        nb["config"] = dict(bundle["config"], max_segments=int(args.segments))
        if best == "avg-train":
            nb["config"]["train_segments"] = int(args.segments)
        nb.pop("_path", None)
        joblib.dump(nb, REPO / "runs/aigen_detector_multiseg.joblib")
        ta.log("candidate bundle -> runs/aigen_detector_multiseg.joblib")
    print("\n".join(L))


if __name__ == "__main__":
    main()
