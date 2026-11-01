"""Score the app on YOUR labelled videos (docs/11 F-1). Local only.

Put clips you know the answer for in either layout:

    data/raw/user_eval/ai/*.mp4      (AI-generated or manipulated)
    data/raw/user_eval/real/*.mp4    (real camera footage)

or list them in a CSV with columns ``path,label`` (label: ai|real), plus
optional ``note``. Then:

    python scripts/eval_user_set.py                       # folder layout
    python scripts/eval_user_set.py --csv my_labels.csv   # CSV

Runs exactly what the web app runs (``trubeat.app.analyze_video``) on each
clip and reports: how often the verdict is right, how often it is
inconclusive, false alarms on real videos and misses on AI videos, each
with a 95% Wilson interval, plus a per-clip table. Nothing leaves the
machine; the videos are never copied, committed or uploaded (AGENTS.md §7).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from trubeat.app import analyze_video

REPO = Path(__file__).resolve().parents[1]
VIDEO_EXT = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"}


def wilson(k: int, n: int) -> str:
    if n == 0:
        return "n/a"
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return f"{k}/{n} = {p:.0%} [{(c - h) / d:.0%}, {(c + h) / d:.0%}]"


def load_items(args) -> pd.DataFrame:
    if args.csv:
        df = pd.read_csv(args.csv)
        if not {"path", "label"} <= set(df.columns):
            raise SystemExit("CSV needs columns path,label")
    else:
        root = Path(args.dir)
        rows = [
            {"path": str(f), "label": label}
            for label in ("ai", "real")
            for f in sorted((root / label).glob("*"))
            if f.suffix.lower() in VIDEO_EXT
        ]
        if not rows:
            raise SystemExit(
                f"no videos under {root}/ai or {root}/real; see this script's docstring"
            )
        df = pd.DataFrame(rows)
    df["label"] = df["label"].str.lower().str.strip()
    bad = set(df.label) - {"ai", "real"}
    if bad:
        raise SystemExit(f"labels must be ai|real, got {sorted(bad)}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default=str(REPO / "data/raw/user_eval"))
    ap.add_argument("--csv", default=None)
    ap.add_argument("--out", default=str(REPO / "reports/user_eval.md"))
    args = ap.parse_args()
    items = load_items(args)
    recs = []
    for i, it in enumerate(items.itertuples(), 1):
        t = time.time()
        a = analyze_video(it.path)
        recs.append({
            "clip": Path(it.path).name, "label": it.label, "verdict": a.verdict,
            "decided_by": a.decided_by or "-", "ai_pct": a.ai_real_percentile,
            "duration_s": round(a.duration_s, 1), "headline": a.headline,
            "why_inconclusive": a.detail if a.verdict == "inconclusive" else "",
        })
        print(f"[{i}/{len(items)}] {Path(it.path).name}: {a.verdict} "
              f"({a.decided_by or 'no decision'}) in {time.time() - t:.0f}s", flush=True)
    df = pd.DataFrame(recs)
    want = {"ai": "fake", "real": "real"}
    decided = df[df.verdict != "inconclusive"]
    ai, real = df[df.label == "ai"], df[df.label == "real"]
    lines = [
        f"# TrueBeat on your labelled videos — {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        f"{len(df)} clips ({len(ai)} AI, {len(real)} real). Local run; nothing uploaded.",
        "",
        "| measure | result (95% CI) |",
        "|---|---|",
        f"| inconclusive | {wilson(int((df.verdict == 'inconclusive').sum()), len(df))} |",
        (
            "| correct, among decided clips | "
            f"{wilson(int((decided.verdict == decided.label.map(want)).sum()), len(decided))} |"
        ),
        f"| AI clips caught | {wilson(int((ai.verdict == 'fake').sum()), len(ai))} |",
        f"| real clips wrongly flagged | {wilson(int((real.verdict == 'fake').sum()), len(real))} |",
        "",
        "## Per clip",
        "",
        "| clip | label | verdict | decided by | AI score above % of real clips | length | why inconclusive |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in df.itertuples():
        p = "n/a" if r.ai_pct is None or pd.isna(r.ai_pct) else f"{r.ai_pct:.0%}"
        lines.append(f"| {r.clip} | {r.label} | {r.verdict} | {r.decided_by} | {p} | "
                     f"{r.duration_s} s | {r.why_inconclusive} |")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:11]))
    print(f"\nfull table -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
