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

# ---------------------------------------------------------------------------
# Frozen copy of the app's verdict logic as audited on 2026-11-01 (commit
# 613c323), so this baseline stays reproducible after the app moved to the
# fused verdict (docs/11 E1). Do not "fix" these: they are the audited
# behaviour, bugs included.
LEGACY_MIN_DURATION_S = 10.0
LEGACY_MIN_FACE_VALID_FRAC = 0.70
LEGACY_MIN_SNR = 10.0


def legacy_mean_snr(ppg: np.ndarray, fps: float) -> float:
    """The audited peak/mean in-band power ratio (0.7-14 Hz), docs/11 F1."""
    n = ppg.shape[0]
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    mask = (freqs >= 0.7) & (freqs <= 14.0)
    snrs = []
    for r in range(ppg.shape[1]):
        power = np.abs(np.fft.rfft(ppg[:, r] - np.nanmean(ppg[:, r]))) ** 2
        band = power[mask]
        if band.size and np.all(np.isfinite(band)):
            snrs.append(float(band.max() / (band.mean() + 1e-12)))
    return float(np.mean(snrs)) if snrs else 0.0


def legacy_verdict(dur, fvf, snr, agree, score, threshold) -> tuple[str, str]:
    """(verdict, reason) exactly as the audited verdict_for decided it."""
    reasons = []
    if dur < LEGACY_MIN_DURATION_S:
        reasons.append("short")
    if fvf <= 0.0:
        reasons.append("no face was tracked")
    elif fvf < LEGACY_MIN_FACE_VALID_FRAC:
        reasons.append("face tracked in only part of the clip")
    if snr < LEGACY_MIN_SNR:
        reasons.append("SNR")
    if agree is None:
        reasons.append("no agreement estimate")
    if reasons:
        return "inconclusive", "; ".join(reasons)
    if score is None:
        return "inconclusive", "no detector"
    return ("real" if score >= threshold else "fake"), ""


def replay(manifest: Path, trace_dir: Path, bundle: dict) -> pd.DataFrame:
    cols, det = bundle["feature_columns"], bundle["model"]
    method, win_s = bundle["method"], float(bundle["window_s"])
    thr = float(bundle["threshold"])
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
        snr = legacy_mean_snr(ppg, fps)
        feat = features.window_features(ppg, fps, win_s=win_s, overlap=0.5)
        score = float(np.mean(det.decision_function(feat[cols]))) if len(feat) else None
        verdict, reason = legacy_verdict(dur, fvf, snr, agree, score, thr)
        why = verdict
        if verdict == "inconclusive":
            why = ("inc:face" if "tracked" in reason else
                   "inc:snr" if "SNR" in reason else "inc:other")
        rec.update(verdict=verdict, why=why, snr=snr, agreement=agree,
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

    print(f"\nSNR gate probe (legacy peak/mean SNR, gate = {LEGACY_MIN_SNR:g}):")
    print(f"  {'signal':30s} {'10.9 s':>7s} {'30 s':>7s} {'120 s':>7s}")
    for name, gen in (("white noise, no pulse", white),
                      ("random-walk drift, no pulse", drift),
                      ("weak 72 bpm pulse + noise", pulse)):
        vals = [np.median([legacy_mean_snr(gen(int(s * fps)), fps) for _ in range(n)])
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
