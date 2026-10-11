# 08 — Acceptance Test Plan

## 1. Heart-rate acceptance (Phase 1)

**Dataset:** UBFC-rPPG (42 subjects, oximeter ground truth).

| Metric | How | Target |
|--------|-----|--------|
| MAE (bpm) | mean \|estimated − ground truth\| per video | ≤ 5 |
| RMSE (bpm) | penalizes big misses | ≤ 8 |
| Pearson r | estimated vs. true HR across videos | ≥ 0.85 |
| Bland-Altman | bias ± limits of agreement, plotted | bias within ±3 bpm |

- Report **per-subject** MAE table, not just the mean — one terrible subject
  can hide behind a good average.
- Sanity check on the way in: a synthetic 72 bpm sine through your pipeline
  must read 72 ± frequency resolution.

## 2. Detector acceptance (Phases 2–3)

**Datasets:** FF++ c23 (train/val/test, person-disjoint) → Celeb-DF v2 + self-made
fakes (final exam, evaluated once).

| Metric | How | Target |
|--------|-----|--------|
| Accuracy | person-disjoint test split | reported, not gated |
| AUC | person-disjoint test split | ≥ 0.70 (≥ 0.50 minimum) |
| Precision / Recall | same split | reported with the chosen operating threshold |
| False-positive rate on **real** videos | real-only slice | headline metric (BBC lesson) |
| Cross-dataset AUC | Celeb-DF v2, self-made DFL fakes | reported with CIs; any drop vs. in-dataset is expected and must be explained, not hidden |
| Compression robustness | train clean → test c40, vs. train mixed → test c40 | mixed training must recover most of the drop (Intel finding) |

- **Confidence intervals:** bootstrap (1,000 resamples) on every headline metric.
- **Split protocol** is part of every report: method, grouping key, and the
  validator's leakage check result.
- **Operating threshold** is tuned on validation; the report shows the full ROC,
  never a single cherry-picked accuracy.

## 3. Regression (runs on every change)

A fixed **golden set**: 5 videos (2 real UBFC, 1 real FF++, 2 fake FF++) with
recorded expected outputs. `pytest tests/test_golden.py` fails the build if any
number drifts beyond tolerance. Re-record the golden outputs deliberately, never
by accident.

## 4. System-level checks (Phase 4)

- `trubeat predict` on a 10 s clip finishes in ≤ 60 s on the reference laptop.
- All six error codes (`E_UNREADABLE`, `E_LOW_FPS`, `E_SHORT_VIDEO`, `E_NO_FACE`,
  `W_UNSTABLE_TRACK`, `W_LOW_CONFIDENCE`) trigger correctly on crafted inputs.
- Inference path makes zero network calls (verified with a blocked-network run).
- Docker image builds from scratch and serves `/predict` + the demo page.

## 5. Report template (every evaluation produces this)

```markdown
# Run report — <date> — <config hash>
## Setup
Dataset, split protocol, method, ROI set, band, window, model, seed
## Heart-rate (if applicable)
MAE / RMSE / Pearson / Bland-Altman + per-subject table + plot
## Detection
Accuracy / AUC / Precision / Recall / confusion matrix + ROC plot
False-positive rate on real videos (standalone line)
## Cross-dataset
Per-dataset AUC with bootstrap CIs
## Compression
Clean→compressed vs mixed→compressed table
## Honesty checklist
[ ] split protocol documented  [ ] no leakage (validator green)
[ ] CIs reported  [ ] threshold from validation only
[ ] limitations listed (motion, compression, skin tone, pulse inheritance)
```

## 6. Honesty rules (non-negotiable)

1. Never quote "96%" — that was one older test set; the same paper got ~91%
   on harder data, and a from-scratch build lands below both.
2. Always name the dataset, the split, and the grouping key next to every number.
3. A pulse detected is not a video verified — say what the detector *cannot* do
   (catch pulse-inheriting fakes via global HR) in every report's limitations.
