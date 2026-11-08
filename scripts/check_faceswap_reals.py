"""Gate 4 for the face-swap detector: false alarms on real video that is not
FaceForensics++ (GenBuster-Bench and AIGVDBench real clips, already on the
laptop), over the clips the detector judged (a face in at least min_faces
sampled frames).

Two modes:

* ``--manifest data/manifests/faceswap_gate4a.csv`` (docs/13 §6.3 gate 4a):
  the frozen round-3 list, every clip scored, each file's SHA-256 checked
  first. Gate: flagged <= 3% of judged, n judged >= 200, and no single
  source above 6%.
* the round-3 mode (docs/12 §6 gate 4, kept for the record): a fixed
  shuffled order (seed 7) of the AI-video pool's reals, scored until
  ``--target`` clips were judged.

Writes reports/<name>.{md,csv,json}; the JSON (with the bundle's SHA-256 and
per-source counts) feeds the ship step.

Usage:
    python scripts/check_faceswap_reals.py --bundle <bundle> --manifest data/manifests/faceswap_gate4a.csv \\
        --out reports/faceswap_gate4a.md
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import train_aigen as ta

from trubeat import faceswap as fs

_W: dict = {}


def _init(bundle_path: str) -> None:
    warnings.filterwarnings("ignore")
    b = fs.load_bundle(bundle_path)
    if b.get("kind") == "cnn":  # weights next to the bundle (from the cloud), checked against its SHA-256
        scorer = fs.CNNScorer.from_bundle(b)
        scorer.torch_threads = 2
    else:
        cfg = fs.merged_config(b["config"])
        scorer = fs.FaceEmbedder(cfg["backbone"], cfg["feature"], 2, img_size=int(cfg["crop_size"]))
    _W.update(bundle=b, scorer=scorer, finder=fs.FaceFinder(b["config"]))


def score(path: str) -> tuple[str, str, float | None, int, str]:
    r = fs.analyze(path, _W["bundle"], embedder=_W["scorer"], finder=_W["finder"])
    return path, r.label, r.score, r.n_faces, "; ".join(r.warnings)


def sha256(path: Path) -> str:
    from trubeat.claims import file_sha256

    return file_sha256(path)


def frozen_list(manifest: str) -> tuple[list[str], dict[str, str]]:
    m = pd.read_csv(manifest).sort_values("order")
    paths, src, bad = [], {}, []
    for r in m.itertuples(index=False):
        p = REPO / r.path if not Path(r.path).is_absolute() else Path(r.path)
        if not p.is_file() or sha256(p) != r.sha256:
            bad.append(r.path)
            continue
        paths.append(str(p))
        src[str(p)] = r.source
    if bad:
        raise SystemExit(f"{len(bad)} clips of the frozen list are missing or changed (SHA-256): {bad[:3]}")
    return paths, src


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--manifest", default=None, help="the frozen gate-4a list (docs/13 §6.3)")
    ap.add_argument("--features", default=str(REPO / "runs/aigen_features_v2.pkl"), help="round-3 mode")
    ap.add_argument("--target", type=int, default=300, help="round-3 mode: judged clips to reach")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default=str(REPO / "reports/faceswap_gate4.md"))
    args = ap.parse_args()
    if args.manifest:
        paths, src = frozen_list(args.manifest)
        target = None
        what = (f"the frozen gate-4a list `{Path(args.manifest).name}` ({len(paths)} GenBuster-Bench and AIGVDBench "
                "real clips, every file's SHA-256 checked), every clip processed")
    else:
        d = pd.read_pickle(args.features)
        reals = d[(d.label == "real") & (d.source != "ffpp_original")]
        reals = reals.sample(frac=1.0, random_state=7).reset_index(drop=True)
        paths = [p for p in reals.path if Path(p).is_file()]
        src = dict(zip(reals.path, reals.source))
        target = args.target
        what = "GenBuster-Bench + AIGVDBench reals in a fixed shuffled order (seed 7)"
    res, t0 = [], time.time()
    with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(args.bundle,)) as pool:
        for path, label, s, n, w in pool.imap(score, paths, chunksize=4):
            res.append({"path": path, "source": src[path], "label": label, "score": s, "n_faces": n, "warnings": w})
            judged = sum(r["label"] in ("swap", "no_swap") for r in res)
            if len(res) % 100 == 0:
                ta.log(f"  {len(res)} processed, {judged} judged ({time.time() - t0:.0f}s)")
            if target is not None and judged >= target:
                pool.terminate()
                break
    df = pd.DataFrame(res)
    errors = df[df.label == "unknown"]
    face = df[df.label.isin(["swap", "no_swap"])]
    k, n = int((face.label == "swap").sum()), len(face)
    lo, hi = ta.wilson(k, n)
    per = {}
    for s_, g in face.groupby("source"):
        kk = int((g.label == "swap").sum())
        per[s_] = {"k": kk, "n": len(g), "rate": kk / max(len(g), 1), "ci": list(ta.wilson(kk, len(g))),
                   "processed": int((df.source == s_).sum())}
    worst = max((v["rate"] for v in per.values()), default=0.0)
    if args.manifest:
        passed = n >= 200 and k / max(n, 1) <= 0.03 and worst <= 0.06
        rule = "<= 3% of judged, n judged >= 200, no source above 6%"
    else:
        passed = n >= 200 and k / max(n, 1) <= 0.03
        rule = "<= 3%, n >= 200"
    b = fs.load_bundle(args.bundle)
    cfg = fs.merged_config(b["config"])
    bsha = sha256(Path(args.bundle))
    L = [f"# Face-swap detector, gate 4{'a' if args.manifest else ''} — {time.strftime('%Y-%m-%d %H:%M')}", "",
         (f"Bundle `{Path(args.bundle).name}` (SHA-256 `{bsha[:12]}…`; {b.get('kind', 'probe')}, backbone "
          f"{cfg['backbone']}, {cfg['feature']}, finder {cfg['finder']}, threshold {b['threshold']:+.4f})."), "",
         (f"Clips: {what}: {len(df)} processed, **{n} judged** (a face in at least {cfg['min_faces']} of "
          f"{cfg['n_frames']} sampled frames), {len(errors)} not judged because of an error or a too-short clip."), "",
         f"Flagged as face-swap: **{k}/{n} judged = {k / max(n, 1):.1%}** [{lo:.1%}, {hi:.1%}] (Wilson 95%).", "",
         f"**Gate ({rule}): {'pass' if passed else 'fail'}.**", "",
         "| source | processed | judged | flagged (95% CI) |", "|---|---|---|---|"]
    for s_, v in per.items():
        L.append(f"| {s_} | {v['processed']} | {v['n']} | {v['k']}/{v['n']} = {v['rate']:.1%} "
                 f"[{v['ci'][0]:.1%}, {v['ci'][1]:.1%}] |")
    Path(args.out).write_text("\n".join(L) + "\n")
    df.to_csv(Path(args.out).with_suffix(".csv"), index=False)
    # read by the ship step, which checks it belongs to the same bundle file
    Path(args.out).with_suffix(".json").write_text(json.dumps({
        "bundle": args.bundle, "bundle_sha256": bsha, "k": k, "n": n, "rate": k / max(n, 1), "ci": [lo, hi],
        "passed": passed, "rule": rule, "per_source": per, "n_processed": len(df), "n_scored": len(df),
        "n_not_judged_error": len(errors), "frozen_manifest": Path(args.manifest).name if args.manifest else None,
        "frozen_manifest_sha256": sha256(Path(args.manifest)) if args.manifest else None,
        "created": time.strftime("%Y-%m-%d %H:%M")}, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main()
