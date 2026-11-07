# Whole-frame AI-video detector — 2026-11-01 11:37

Protocol: docs/11 D4/D5 and `scripts/train_aigen.py` docstring. Every clip re-encoded to one format before features ({'short_side': 360, 'crf': 28, 'preset': 'veryfast', 'out_dir': 'data/processed/aigen_norm'}). Threshold from out-of-fold pool reals at FPR 0.05; external data never used for selection or the threshold. Config hash `5c84ab9d`.

## Data

- pool: 3878 clips (2521 AI from 26 generators; 1357 real)
- external: 3285 clips (GenBuster-Bench all tiers + FF++ c23 original test split)

## Model selection (macro leave-one-generator-family-out AUC on the pool)

| feature set | C | macro LOGO AUC | worst generator AUC |
|---|---|---|---|
| geometry | 0.01 | 0.5323 | 0.241 |
| geometry | 0.1 | 0.5494 | 0.337 |
| geometry | 1.0 | 0.5689 | 0.423 |
| embedding | 0.01 | 0.8090 | 0.538 |
| embedding | 0.1 | 0.7975 | 0.534 |
| embedding | 1.0 | 0.7937 | 0.533 |
| combined | 0.01 | 0.8075 | 0.527 |
| combined | 0.1 | 0.7959 | 0.522 |
| combined | 1.0 | 0.7930 | 0.525 |

Selected: **embedding, C=0.01**. Pooled out-of-fold AUC (grouped 5-fold, in-distribution): 0.8101. Threshold: +1.2572.

### Leave-one-generator-family-out, selected model (each family unseen in training)

| generator family | n AI | n real | AUC |
|---|---|---|---|
| seine | 100 | 271 | 0.538 |
| ltx | 300 | 272 | 0.552 |
| pyramid-flow | 200 | 272 | 0.637 |
| easyanimate | 200 | 272 | 0.680 |
| svd | 100 | 271 | 0.716 |
| cogvideox | 300 | 272 | 0.722 |
| hunyuanvideo | 22 | 271 | 0.748 |
| ipoc | 100 | 271 | 0.749 |
| wan | 300 | 272 | 0.880 |
| kling | 99 | 272 | 0.909 |
| vidu | 100 | 272 | 0.916 |
| luma | 100 | 271 | 0.930 |
| runway | 200 | 272 | 0.941 |
| sora | 100 | 271 | 0.947 |
| pika | 100 | 271 | 0.954 |
| opensora | 100 | 271 | 0.959 |
| jimeng | 100 | 271 | 0.974 |

## External evaluation (never seen in training, selection or threshold)

| metric | value | 95% CI (Wilson) |
|---|---|---|
| false-positive rate, GenBuster real | 57/1000 = 0.057 | [0.044, 0.073] |
| false-positive rate, FF++ c23 real (talking heads) | 0/135 = 0.000 | [0.000, 0.028] |

| GenBuster tier | n AI | AUC vs GenBuster reals | 95% CI | recall at threshold | 95% CI |
|---|---|---|---|---|---|
| id | 1000 | 0.872 | [0.856, 0.887] | 0.499 | [0.468, 0.530] |
| ood | 1000 | 0.954 | [0.946, 0.963] | 0.783 | [0.756, 0.807] |
| wild | 150 | 0.859 | [0.828, 0.889] | 0.453 | [0.376, 0.533] |

| generator (external) | tier | n | AUC vs GenBuster reals | recall at threshold |
|---|---|---|---|---|
| cogvideox | id | 300 | 0.832 | 0.363 |
| easyanimate | id | 400 | 0.950 | 0.743 |
| hunyuanvideo | id | 100 | 0.767 | 0.260 |
| ltxvideo | id | 200 | 0.828 | 0.335 |
| gen3 | ood | 100 | 0.965 | 0.830 |
| jimeng | ood | 100 | 0.915 | 0.680 |
| kling | ood | 100 | 0.957 | 0.750 |
| luma | ood | 100 | 0.946 | 0.750 |
| pika | ood | 100 | 0.948 | 0.730 |
| sora | ood | 200 | 0.967 | 0.850 |
| vidu | ood | 150 | 0.967 | 0.780 |
| wanx | ood | 150 | 0.953 | 0.813 |
| gen4.5 | wild | 10 | 0.939 | 0.500 |
| hailuo2.3 | wild | 20 | 0.847 | 0.400 |
| kling2.5_turbo | wild | 15 | 0.875 | 0.400 |
| kling2.6 | wild | 20 | 0.852 | 0.500 |
| pixverse_v5 | wild | 10 | 0.833 | 0.600 |
| seedance1.5_pro | wild | 20 | 0.711 | 0.200 |
| seedance2.0 | wild | 15 | 0.839 | 0.333 |
| sora2 | wild | 10 | 0.925 | 0.500 |
| veo3.1 | wild | 10 | 0.947 | 0.600 |
| wan2.5 | wild | 10 | 0.903 | 0.500 |
| wan2.6 | wild | 10 | 0.952 | 0.800 |

## Robustness and input handling (same external subset, threshold fixed)

| input | auc_id | auc_ood | auc_wild | fpr_ffpp_real | fpr_genbuster_real | recall_id | recall_ood | recall_wild |
|---|---|---|---|---|---|---|---|---|
| training normalisation (360p, CRF 28) | 0.857 | 0.981 | 0.882 | 0.000 | 0.000 | 0.400 | 0.744 | 0.453 |
| harsh re-encode (240p, CRF 35) | 0.833 | 0.973 | 0.850 | 0.000 | 0.000 | 0.250 | 0.575 | 0.280 |
| tab-capture-like (VP9 WebM, 720p, 2.5 Mb/s) | 0.864 | 0.983 | 0.897 | 0.000 | 0.000 | 0.438 | 0.838 | 0.520 |
| as-is (no re-encode) | 0.853 | 0.980 | 0.890 | 0.000 | 0.000 | 0.450 | 0.838 | 0.527 |

## Limits

- Real clips come from OpenVid-HD (AIGVDBench, GenBuster) and FF++ YouTube; other real content (phone video, screen recordings, animation) is not covered.
- AUC and recall are per-clip on curated sets; the user's own labelled set (docs/11 F-1) is the test that matters.
- Generators released after these datasets are untested by construction.
