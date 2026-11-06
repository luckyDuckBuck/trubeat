"""docs/12 §6 gate 4: false alarms of the face-swap detector on real video
that is not FaceForensics++ (GenBuster-Bench and AIGVDBench real clips,
already on the laptop), counting only clips where a face was found.

Gate: flagged <= 3% with at least 200 clips that have a face. Clips are
taken in a fixed shuffled order (seed 7) until --target clips with a face
have been scored.

Writes reports/faceswap_gate4.{md,csv,json}; the JSON feeds
`scripts/train_faceswap.py ship`.

Usage:
    python scripts/check_faceswap_reals.py --bundle runs/faceswap/faceswap.joblib [--target 300]
"""

from __future__ import annotations

import argparse
import hashlib
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
    if b.get("kind") == "cnn":  # weights next to the bundle (from Colab), checked against its SHA-256
        scorer = fs.CNNScorer.from_bundle(b)
        scorer.torch_threads = 2
    else:
        cfg = fs.merged_config(b["config"])
        scorer = fs.FaceEmbedder(cfg["backbone"], cfg["feature"], 2, img_size=int(cfg["crop_size"]))
    _W.update(bundle=b, scorer=scorer, finder=fs.FaceFinder(b["config"]))


def score(path: str) -> tuple[str, str, float | None, int]:
    r = fs.analyze(path, _W["bundle"], embedder=_W["scorer"], finder=_W["finder"])
    return path, r.label, r.score, r.n_faces


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--features", default=str(REPO / "runs/aigen_features_v2.pkl"))
    ap.add_argument("--target", type=int, default=300, help="clips with a face to score")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default=str(REPO / "reports/faceswap_gate4.md"))
    args = ap.parse_args()
    d = pd.read_pickle(args.features)
    reals = d[(d.label == "real") & (d.source != "ffpp_original")]
    reals = reals.sample(frac=1.0, random_state=7).reset_index(drop=True)
    paths = [p for p in reals.path if Path(p).is_file()]
    src = dict(zip(reals.path, reals.source))
    res, t0 = [], time.time()
    with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(args.bundle,)) as pool:
        for path, label, s, n in pool.imap(score, paths, chunksize=4):
            res.append({"path": path, "source": src[path], "label": label, "score": s, "n_faces": n})
            with_face = sum(r["label"] in ("swap", "no_swap") for r in res)
            if len(res) % 100 == 0:
                ta.log(f"  {len(res)} scored, {with_face} with a face ({time.time() - t0:.0f}s)")
            if with_face >= args.target:
                pool.terminate()
                break
    df = pd.DataFrame(res)
    face = df[df.label.isin(["swap", "no_swap"])]
    k, n = int((face.label == "swap").sum()), len(face)
    lo, hi = ta.wilson(k, n)
    passed = n >= 200 and k / max(n, 1) <= 0.03
    b = fs.load_bundle(args.bundle)
    L = [f"# Face-swap detector, gate 4 — {time.strftime('%Y-%m-%d %H:%M')}", "",
         f"Bundle `{args.bundle}` ({b.get('kind', 'probe')}, threshold {b['threshold']:+.4f}).", "",
         (f"Real, non-FaceForensics clips (GenBuster-Bench + AIGVDBench reals) in a fixed shuffled order: "
          f"{len(df)} scored, {n} with a face in at least {fs.merged_config(b['config'])['min_faces']} "
          f"sampled frames."), "",
         f"Flagged as face-swap: **{k}/{n} = {k / max(n, 1):.1%}** [{lo:.1%}, {hi:.1%}].", "",
         f"**Gate 4 (<= 3%, n >= 200): {'pass' if passed else 'fail'}.**", "",
         "| source | with a face | flagged |", "|---|---|---|"]
    for s_, g in face.groupby("source"):
        L.append(f"| {s_} | {len(g)} | {int((g.label == 'swap').sum())} |")
    Path(args.out).write_text("\n".join(L) + "\n")
    df.to_csv(Path(args.out).with_suffix(".csv"), index=False)
    # read by `train_faceswap.py ship`, which checks it belongs to the same bundle file
    Path(args.out).with_suffix(".json").write_text(json.dumps({
        "bundle": args.bundle, "bundle_sha256": hashlib.sha256(Path(args.bundle).read_bytes()).hexdigest(),
        "k": k, "n": n, "rate": k / max(n, 1), "ci": [lo, hi], "passed": passed, "n_scored": len(df),
        "created": time.strftime("%Y-%m-%d %H:%M")}, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main()
