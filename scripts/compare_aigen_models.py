"""Apply the docs/11 §9.7 adoption rule: shipped AI-video model vs a retrain.

Rule (fixed in docs/11 §9.7 before any v2 result existed, commit 3596859):
  1. v2 false alarms at its own out-of-fold threshold: GenBuster reals
     <= 7.3% and FF++ c23 test reals <= 2/135;
  2. GenBuster wild recall >= shipped + 5 points, and no GenBuster tier's
     recall drops by more than 3 points;
  3. pool macro leave-one-family-out AUC over the shipped model's families
     >= shipped - 0.01.

Usage:
    python scripts/compare_aigen_models.py \\
        --old runs/aigen_detector.joblib --old-features runs/aigen_features.pkl \\
        --new runs/aigen_detector_v2.joblib --new-features runs/aigen_features_v2.pkl
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import joblib
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import train_aigen as ta

MAX_GB_FPR, MAX_FF_FLAGS = 0.073, 2
MIN_WILD_GAIN, MAX_TIER_DROP, MAX_LOGO_DROP = 0.05, 0.03, 0.01


def evaluate(bundle_path: str, features_path: str) -> dict:
    b = joblib.load(bundle_path)
    d = pd.read_pickle(features_path)
    cols, thr, C = b["feature_columns"], float(b["threshold"]), float(b["model"][-1].C)
    ext = d[d.role == "external"].reset_index(drop=True)
    s = ta.ai_score(b["model"], ext[cols].to_numpy())
    ext = ext.assign(score=s, flagged=s >= thr)
    pool = d[d.role == "pool"].reset_index(drop=True)
    lg = ta.logo_eval(pool, cols, C).set_index("generator")
    per_gen = (ext[ext.label == "ai"].groupby(["source", "generator"]).flagged
               .agg(["mean", "size"]))
    return {"bundle": b, "ext": ext, "res": ta.compact_eval(ext), "logo": lg,
            "ff_flags": int(ext[ext.source == "ffpp_original"].flagged.sum()),
            "n_ff": int((ext.source == "ffpp_original").sum()), "per_gen": per_gen}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--old", default=str(REPO / "runs/aigen_detector.joblib"))
    ap.add_argument("--old-features", default=str(REPO / "runs/aigen_features.pkl"))
    ap.add_argument("--new", default=str(REPO / "runs/aigen_detector_v2.joblib"))
    ap.add_argument("--new-features", default=str(REPO / "runs/aigen_features_v2.pkl"))
    ap.add_argument("--out", default=str(REPO / "reports/aigen_v2_decision.md"))
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    old, new = evaluate(args.old, args.old_features), evaluate(args.new, args.new_features)
    fams = list(old["logo"].index)
    old_macro = float(old["logo"].auc.mean())
    new_macro_same = float(new["logo"].auc.reindex(fams).mean())
    o, n = old["res"], new["res"]
    tiers = ("id", "ood", "wild")
    checks = [
        (f"GenBuster-real FPR <= {MAX_GB_FPR:.1%}", n["fpr_genbuster_real"] <= MAX_GB_FPR,
         f"{n['fpr_genbuster_real']:.3f} (shipped {o['fpr_genbuster_real']:.3f})"),
        (f"FF++ test reals flagged <= {MAX_FF_FLAGS}", new["ff_flags"] <= MAX_FF_FLAGS,
         f"{new['ff_flags']}/{new['n_ff']} (shipped {old['ff_flags']}/{old['n_ff']})"),
        (f"wild recall >= shipped + {MIN_WILD_GAIN:.0%}",
         n["recall_wild"] >= o["recall_wild"] + MIN_WILD_GAIN,
         f"{n['recall_wild']:.3f} vs {o['recall_wild']:.3f}"),
        *[(f"{t} recall drop <= {MAX_TIER_DROP:.0%}", n[f"recall_{t}"] >= o[f"recall_{t}"] - MAX_TIER_DROP,
           f"{n[f'recall_{t}']:.3f} vs {o[f'recall_{t}']:.3f}") for t in tiers],
        (f"LOGO macro over the shipped model's {len(fams)} families >= shipped - {MAX_LOGO_DROP}",
         new_macro_same >= old_macro - MAX_LOGO_DROP, f"{new_macro_same:.4f} vs {old_macro:.4f}"),
    ]
    adopt = all(ok for _, ok, _ in checks)
    L = [f"# AI-video detector: shipped vs retrain (docs/11 §9.7) — {time.strftime('%Y-%m-%d %H:%M')}",
         "", f"Shipped: `{args.old}` ({old['bundle'].get('trained_on')}).",
         f"Retrain: `{args.new}` ({new['bundle'].get('trained_on')}).", "",
         "| rule | pass | value |", "|---|---|---|"]
    L += [f"| {name} | {'yes' if ok else '**no**'} | {val} |" for name, ok, val in checks]
    L += ["", f"**Decision: {'adopt the retrain' if adopt else 'keep the shipped model'}.**", "",
          "## External measures", "", "| measure | shipped | retrain |", "|---|---|---|"]
    L += [f"| {k} | {o[k]} | {n[k]} |" if isinstance(o[k], int) else f"| {k} | {o[k]:.3f} | {n[k]:.3f} |"
          for k in sorted(o)]
    pg = old["per_gen"].join(new["per_gen"], lsuffix="_old", rsuffix="_new")
    L += ["", "## Recall per external generator", "",
          "| tier | generator | n | shipped | retrain | family in retrain's training? |",
          "|---|---|---|---|---|---|"]
    new_fams = set(new["logo"].index)
    for (src, gen), r in pg.iterrows():
        seen = "yes" if ta.base_family(gen) in new_fams else "no"
        L.append(f"| {src.removeprefix('genbuster_')} | {gen} | {int(r.size_old)} | "
                 f"{r.mean_old:.2f} | {r.mean_new:.2f} | {seen} |")
    L += ["", "## Leave-one-family-out AUC on the pool", "",
          "| family | shipped | retrain |", "|---|---|---|"]
    for fam in sorted(set(fams) | new_fams):
        a = old["logo"].auc.get(fam, float("nan"))
        b = new["logo"].auc.get(fam, float("nan"))
        L.append(f"| {fam} | {a:.3f} | {b:.3f} |")
    Path(args.out).write_text("\n".join(L) + "\n")
    print("\n".join(L))
    joblib.dump({"adopt": adopt, "checks": checks}, REPO / "runs/aigen_v2_decision.joblib")


if __name__ == "__main__":
    main()
