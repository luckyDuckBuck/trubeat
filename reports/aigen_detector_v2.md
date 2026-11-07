# Whole-frame AI-video detector — 2026-11-01 14:19

Protocol: docs/11 D4/D5 and `scripts/train_aigen.py` docstring. Every clip re-encoded to one format before features ({'short_side': 360, 'crf': 28, 'preset': 'veryfast', 'out_dir': 'data/processed/aigen_norm'}). Threshold from out-of-fold pool reals at FPR 0.05; external data never used for selection or the threshold. Config hash `3f757f1f`.

## Data

- pool: 5370 clips (4013 AI from 38 generators; 1357 real)
- external: 3285 clips (GenBuster-Bench all tiers + FF++ c23 original test split)

## Model selection (macro leave-one-generator-family-out AUC on the pool)

| feature set | C | macro LOGO AUC | worst generator AUC |
|---|---|---|---|
| geometry | 0.01 | 0.4605 | 0.307 |
| geometry | 0.1 | 0.4904 | 0.325 |
| geometry | 1.0 | 0.5182 | 0.370 |
| embedding | 0.01 | 0.8414 | 0.543 |
| embedding | 0.1 | 0.8282 | 0.553 |
| embedding | 1.0 | 0.8241 | 0.554 |
| combined | 0.01 | 0.8407 | 0.535 |
| combined | 0.1 | 0.8280 | 0.544 |
| combined | 1.0 | 0.8250 | 0.549 |

Selected: **embedding, C=0.01**. Pooled out-of-fold AUC (grouped 5-fold, in-distribution): 0.8704. Threshold: +1.1149.

### Leave-one-generator-family-out, selected model (each family unseen in training)

| generator family | n AI | n real | AUC |
|---|---|---|---|
| ltx | 300 | 272 | 0.543 |
| seine | 100 | 272 | 0.547 |
| pyramid-flow | 200 | 271 | 0.624 |
| svd | 100 | 271 | 0.666 |
| easyanimate | 200 | 272 | 0.679 |
| cogvideox | 300 | 272 | 0.713 |
| ipoc | 100 | 271 | 0.756 |
| wan | 379 | 272 | 0.905 |
| marey | 78 | 271 | 0.915 |
| kling | 176 | 272 | 0.919 |
| luma | 258 | 271 | 0.920 |
| hunyuanvideo | 180 | 271 | 0.925 |
| seedance | 73 | 271 | 0.927 |
| veo | 235 | 271 | 0.935 |
| pika | 416 | 272 | 0.943 |
| sora | 260 | 272 | 0.944 |
| vidu | 100 | 271 | 0.949 |
| runway | 279 | 271 | 0.952 |
| mochi | 79 | 271 | 0.960 |
| opensora | 100 | 272 | 0.965 |
| jimeng | 100 | 271 | 0.979 |

## External evaluation (never seen in training, selection or threshold)

| metric | value | 95% CI (Wilson) |
|---|---|---|
| false-positive rate, GenBuster real | 54/1000 = 0.054 | [0.042, 0.070] |
| false-positive rate, FF++ c23 real (talking heads) | 0/135 = 0.000 | [0.000, 0.028] |

| GenBuster tier | n AI | AUC vs GenBuster reals | 95% CI | recall at threshold | 95% CI |
|---|---|---|---|---|---|
| id | 1000 | 0.880 | [0.864, 0.893] | 0.522 | [0.491, 0.553] |
| ood | 1000 | 0.961 | [0.953, 0.969] | 0.831 | [0.807, 0.853] |
| wild | 150 | 0.880 | [0.847, 0.909] | 0.547 | [0.467, 0.624] |

| generator (external) | tier | n | AUC vs GenBuster reals | recall at threshold |
|---|---|---|---|---|
| cogvideox | id | 300 | 0.827 | 0.367 |
| easyanimate | id | 400 | 0.957 | 0.780 |
| hunyuanvideo | id | 100 | 0.808 | 0.320 |
| ltxvideo | id | 200 | 0.840 | 0.340 |
| gen3 | ood | 100 | 0.972 | 0.860 |
| jimeng | ood | 100 | 0.926 | 0.730 |
| kling | ood | 100 | 0.957 | 0.810 |
| luma | ood | 100 | 0.954 | 0.750 |
| pika | ood | 100 | 0.960 | 0.840 |
| sora | ood | 200 | 0.974 | 0.900 |
| vidu | ood | 150 | 0.975 | 0.860 |
| wanx | ood | 150 | 0.957 | 0.820 |
| gen4.5 | wild | 10 | 0.948 | 0.700 |
| hailuo2.3 | wild | 20 | 0.861 | 0.550 |
| kling2.5_turbo | wild | 15 | 0.889 | 0.533 |
| kling2.6 | wild | 20 | 0.893 | 0.650 |
| pixverse_v5 | wild | 10 | 0.868 | 0.600 |
| seedance1.5_pro | wild | 20 | 0.726 | 0.250 |
| seedance2.0 | wild | 15 | 0.883 | 0.533 |
| sora2 | wild | 10 | 0.943 | 0.600 |
| veo3.1 | wild | 10 | 0.946 | 0.800 |
| wan2.5 | wild | 10 | 0.921 | 0.400 |
| wan2.6 | wild | 10 | 0.957 | 0.600 |

## Limits

- Real clips come from OpenVid-HD (AIGVDBench, GenBuster) and FF++ YouTube; other real content (phone video, screen recordings, animation) is not covered.
- AUC and recall are per-clip on curated sets; the user's own labelled set (docs/11 F-1) is the test that matters.
- Generators released after these datasets are untested by construction.
