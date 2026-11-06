"""docs/12 item A: does the title-card guard leave real data alone?

Rule fixed in docs/12 §6 before this ran: adopt the guard only if it flags
< 0.5% of the dataset clips' first segments. For external (held-out)
clips it flags, the guarded segment is scored with the shipped bundle and
the held-out numbers are recomputed; they may change by no more than the
flagged share.

Also checks the case that motivated it: the RepresentUs clip's end cards.

Usage:
    python scripts/check_graphics_guard.py [--workers 3] [--clip <video>]
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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import train_aigen as ta

from trubeat import aigen

_CFG: dict = {}


def _init(cfg: dict) -> None:
    warnings.filterwarnings("ignore")
    _CFG.update(cfg)


def first_is_graphic(path: str) -> tuple[str, int, float]:
    """(path, 1 if segment 0 is graphic else 0 / -1 on error, median flat share)."""
    try:
        segs, _ = aigen.sample_segments(path, _CFG)
        fr = [aigen.flat_fraction(f, float(_CFG["graphics_block_std"])) for f in segs[0]]
        return path, int(aigen.is_graphic(segs[0], _CFG)), float(np.median(fr))
    except Exception:  # noqa: BLE001
        return path, -1, float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--features", default=str(REPO / "runs/aigen_features_v2.pkl"))
    ap.add_argument("--model", default=str(REPO / "src/trubeat/models/aigen.joblib"))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--clip", default=None, help="a clip with title cards to check (optional)")
    ap.add_argument("--out", default=str(REPO / "reports/graphics_guard.md"))
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    bundle = joblib.load(args.model)
    cfg = aigen.merged_config(dict(bundle["config"], max_segments=1, cache_dir=None))
    d = pd.read_pickle(args.features).reset_index(drop=True)
    t0 = time.time()
    res = {}
    with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(cfg,)) as pool:
        for k, (p, g, fr) in enumerate(pool.imap_unordered(first_is_graphic, d.feat_path.tolist(), chunksize=8), 1):
            res[p] = (g, fr)
            if k % 1000 == 0:
                ta.log(f"  {k}/{len(d)} ({k / (time.time() - t0):.1f} clips/s)")
    d["graphic"] = d.feat_path.map(lambda p: res[p][0])
    d["flat"] = d.feat_path.map(lambda p: res[p][1])
    ok = d[d.graphic >= 0]
    k, n = int(ok.graphic.sum()), len(ok)
    lo, hi = ta.wilson(k, n)
    rate = k / max(n, 1)
    adopt = rate < 0.005
    by = ok.groupby(["role", "source", "label"]).graphic.agg(["sum", "size"])
    L = [f"# Title-card guard check (docs/12 item A) — {time.strftime('%Y-%m-%d %H:%M')}", "",
         (f"Segment 0 of every clip, after the app's 360p re-encode; graphic = median share of flat "
          f"8x8 blocks (std < {cfg['graphics_block_std']}) >= {cfg['graphics_flat_frac']}."), "",
         f"Flagged: **{k}/{n} = {rate:.2%}** [{lo:.2%}, {hi:.2%}] ({int((d.graphic < 0).sum())} unreadable).",
         "", f"**Rule (< 0.5%): {'adopt the guard' if adopt else 'do not adopt; revise the rule first'}.**",
         "", "| role | source | label | flagged | clips |", "|---|---|---|---|---|"]
    L += [f"| {r} | {s} | {lab} | {int(v['sum'])} | {int(v['size'])} |" for (r, s, lab), v in by.iterrows()]
    flagged = ok[ok.graphic == 1].sort_values("flat", ascending=False)
    L += ["", "Flagged clips (most flat first):", ""]
    L += [f"- {r.source} {r.label} {r.generator}: flat {r.flat:.2f} `{Path(r.path).name}`" for r in flagged.head(30).itertuples()]

    # held-out numbers with the guard: re-score only the flagged external clips
    ext = d[d.role == "external"].reset_index(drop=True)
    cols, thr = bundle["feature_columns"], float(bundle["threshold"])
    s0 = ta.ai_score(bundle["model"], ext[cols].to_numpy())
    s1 = s0.copy()
    emb = None
    for i in np.where(ext.graphic.to_numpy() == 1)[0]:
        emb = emb or aigen.Embedder(cfg)
        try:
            frames, _ = aigen.guarded_segments(ext.feat_path[i], cfg)
            e = emb.embed(frames.reshape(-1, *frames.shape[2:])).reshape(frames.shape[0], frames.shape[1], -1)
            s1[i] = float(np.mean(aigen.segment_scores(e, bundle)))
        except Exception:  # noqa: BLE001 - graphics only: the app would not score it
            s1[i] = -np.inf
    before = ta.compact_eval(ext.assign(score=s0, flagged=s0 >= thr))
    after = ta.compact_eval(ext.assign(score=np.where(np.isfinite(s1), s1, -1e9), flagged=s1 >= thr))
    L += ["", "## Held-out AI-video numbers, without and with the guard", "",
          "| measure | without | with guard |", "|---|---|---|"]
    L += [f"| {m} | {before[m]:.3f} | {after[m]:.3f} |" for m in sorted(before) if isinstance(before[m], float)]

    if args.clip:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:  # the app's re-encode first, as at inference
            src, _ = aigen.normalize_for_inference(Path(args.clip), cfg.get("normalize"), Path(tmp))
            segs, meta = aigen.sample_segments(src, dict(cfg, max_segments=30))
        flags = [aigen.is_graphic(s, cfg) for s in segs]
        L += ["", f"## `{Path(args.clip).name}`", "",
              "Segments flagged as graphic (start s): " +
              (", ".join(f"{st:.1f}" for st, f in zip(meta["segment_starts_s"], flags) if f) or "none") +
              f" of {len(segs)}."]
    Path(args.out).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
