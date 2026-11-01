"""Audit the web app's verdicts on every cached FF++ trace (docs/11 §2).

Replays ``trubeat.app`` from the point after face tracking, using the cached
(T, R, 3) ROI RGB means under ``data/processed/traces/``, so all cached clips
run in seconds. The replay reuses the app's own functions (extract_ppg,
estimate_hr, _mean_snr, window_features, the bundled detector and
verdict_for), so its verdicts match ``analyze_video`` on the same clip.

Also probes the SNR gate on synthetic signals (white noise, random-walk
drift, a weak 72 bpm pulse) at three clip lengths.

Usage:
    python scripts/audit_verdicts.py [--out runs/audit_verdicts.csv]
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from trubeat import app, features, hr, ingest, signal

REPO = Path(__file__).resolve().parents[1]
FAMILIES = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]


def replay(manifest: Path, trace_dir: Path, bundle: dict) -> pd.DataFrame:
    cols, det = bundle["feature_columns"], bundle["model"]
    method, win_s = bundle["method"], float(bundle["window_s"])
    thr = app.VerdictThresholds(detector_threshold=float(bundle["threshold"]))
    rows = []
    for _, m in pd.read_csv(manifest).iterrows():
        npz = trace_dir / f"{m.video_id}.npz"
        if not npz.is_file():
            continue
        z = np.load(npz)
        traces, fps = z["traces"], float(z["fps"])
        fvf = float(json.loads(npz.with_suffix(".json").read_text())["face_valid_frac"])
        dur = traces.shape[0] / fps
        rec = {"video_id": m.video_id, "label": m.label, "split": m.split,
               "family": m.video_id.split("_")[1], "duration_s": dur,
               "face_valid_frac": fvf}
        if dur < ingest.MIN_DURATION_S:
            rec.update(verdict="inconclusive", why="inc:short")
            rows.append(rec)
            continue
        ppg = signal.extract_ppg(traces, fps, method=method)
        per = [hr.estimate_hr(ppg[:, r], fps, band=(0.7, 3.0)).bpm
               for r in range(ppg.shape[1])]
        fin = [float(b) for b in per if np.isfinite(b)]
        agree = float(np.std(fin)) if len(fin) > 1 else None
        snr = app._mean_snr(ppg, fps)
        feat = features.window_features(ppg, fps, win_s=win_s, overlap=0.5)
        score = float(np.mean(det.decision_function(feat[cols]))) if len(feat) else None
        a = app.PulseAnalysis(verdict="", headline="", detail="", per_roi_hr=fin,
                              agreement_bpm=agree, face_valid_frac=fvf,
                              mean_snr=snr, duration_s=dur, detector_score=score,
                              detector_available=score is not None)
        app.verdict_for(a, thr)
        why = a.verdict
        if a.verdict == "inconclusive":
            # "tracked" only appears in the face-tracking gate's message
            # (the SNR message also says "face regions").
            why = ("inc:face" if "tracked" in a.detail else
                   "inc:snr" if "SNR" in a.detail else "inc:other")
        rec.update(verdict=a.verdict, why=why, snr=snr, agreement=agree,
                   score=score, hr_min=min(fin) if fin else None)
        rows.append(rec)
    return pd.DataFrame(rows)


def report(d: pd.DataFrame) -> None:
    print(f"clips replayed: {len(d)}")
    print("\nverdict / reason, all cached clips:")
    print(d.why.value_counts().to_string())
    t = d[d.split == "test"]
    print("\nTEST split, verdict by family:")
    print(pd.crosstab(t.family, t.verdict, margins=True).to_string())
    dec = t[t.verdict != "inconclusive"]
    print(f"\nTEST decisive: {len(dec)}/{len(t)}, accuracy "
          f"{(dec.verdict == dec.label).mean():.3f} "
          f"(always-'real' baseline {(t.label == 'real').mean():.3f})")
    for fam, g in dec.groupby("family"):
        print(f"  {fam:15s} n={len(g):3d}  correct={(g.verdict == g.label).mean():.2f}")
    s = t.dropna(subset=["score"])
    print("\nTEST AUC, fake = positive (per family = that family vs all reals):")
    for col, sign in (("score", -1), ("agreement", 1), ("snr", -1)):
        ss = s.dropna(subset=[col])
        line = f"  {col:10s} pooled={roc_auc_score(ss.label == 'fake', sign * ss[col]):.3f}"
        for fam in FAMILIES:
            g = ss[(ss.family == fam) | (ss.label == "real")]
            line += f"  {fam[:5]}={roc_auc_score(g.label == 'fake', sign * g[col]):.3f}"
        print(line)
    r = d[(d.label == "real") & d.agreement.notna()]
    print(f"\nREAL clips: median per-region HR spread {r.agreement.median():.1f} bpm; "
          f"share <= 5 bpm {(r.agreement <= 5).mean():.2f}; "
          f"share with a region < 50 bpm {(r.hr_min < 50).mean():.2f}")


def snr_probe(fps: float = 25.0, n: int = 300, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)

    def white(T):
        return rng.normal(size=(T, 3))

    def drift(T):
        return np.cumsum(rng.normal(size=(T, 3)), axis=0)

    def pulse(T):
        t = np.arange(T) / fps
        return 0.3 * np.sin(2 * np.pi * 1.2 * t)[:, None] + rng.normal(size=(T, 3))

    print(f"\nSNR gate probe (app._mean_snr, gate = {app.DEFAULT_THRESHOLDS.min_snr:g}):")
    print(f"  {'signal':30s} {'10.9 s':>7s} {'30 s':>7s} {'120 s':>7s}")
    for name, gen in (("white noise, no pulse", white),
                      ("random-walk drift, no pulse", drift),
                      ("weak 72 bpm pulse + noise", pulse)):
        vals = [np.median([app._mean_snr(gen(int(s * fps)), fps) for _ in range(n)])
                for s in (10.88, 30.0, 120.0)]
        print(f"  {name:30s} " + " ".join(f"{v:7.1f}" for v in vals))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--manifest", default=str(REPO / "data/manifests/ffpp_c23_qc.csv"))
    p.add_argument("--traces", default=str(REPO / "data/processed/traces"))
    p.add_argument("--model", default=None, help="detector bundle (default: app search order)")
    p.add_argument("--out", default=str(REPO / "runs/audit_verdicts.csv"))
    args = p.parse_args()
    warnings.filterwarnings("ignore")
    bundle = app.load_detector(args.model)
    if bundle is None or "feature_columns" not in bundle:
        raise SystemExit("no G2 detector bundle with feature_columns found")
    d = replay(Path(args.manifest), Path(args.traces), bundle)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(args.out, index=False)
    report(d)
    snr_probe()
    print(f"\nper-clip rows -> {args.out}")


if __name__ == "__main__":
    main()
