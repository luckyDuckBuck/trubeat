# AI-video detector: shipped vs retrain (docs/11 §9.7) — 2026-11-06 08:20

Shipped: `/home/ayushman/Projects/trubeat/runs/aigen_detector.joblib` (AIGVDBench subset (3721 clips) + FF++ c23 originals train/val (157 reals)).
Retrain: `/home/ayushman/Projects/trubeat/runs/aigen_detector_v2.joblib` (aigvdbench (3721 clips) + ffpp_original (157 clips) + rapidata (1492 clips)).

| rule | pass | value |
|---|---|---|
| GenBuster-real FPR <= 7.3% | yes | 0.054 (shipped 0.057) |
| FF++ test reals flagged <= 2 | yes | 0/135 (shipped 0/135) |
| wild recall >= shipped + 5% | yes | 0.547 vs 0.453 |
| id recall drop <= 3% | yes | 0.522 vs 0.499 |
| ood recall drop <= 3% | yes | 0.831 vs 0.783 |
| wild recall drop <= 3% | yes | 0.547 vs 0.453 |
| LOGO macro over the shipped model's 17 families >= shipped - 0.01 | yes | 0.8194 vs 0.8090 |

**Decision: adopt the retrain.**

## External measures

| measure | shipped | retrain |
|---|---|---|
| auc_id | 0.872 | 0.880 |
| auc_ood | 0.954 | 0.961 |
| auc_wild | 0.859 | 0.880 |
| fpr_ffpp_real | 0.000 | 0.000 |
| fpr_genbuster_real | 0.057 | 0.054 |
| median_score_ai | 1.687 | 1.828 |
| median_score_real | -1.033 | -1.442 |
| n_ai_id | 1000 | 1000 |
| n_ai_ood | 1000 | 1000 |
| n_ai_wild | 150 | 150 |
| n_ffpp_real | 135 | 135 |
| n_genbuster_real | 1000 | 1000 |
| recall_id | 0.499 | 0.522 |
| recall_ood | 0.783 | 0.831 |
| recall_wild | 0.453 | 0.547 |

## Recall per external generator

| tier | generator | n | shipped | retrain | family in retrain's training? |
|---|---|---|---|---|---|
| id | cogvideox | 300 | 0.36 | 0.37 | yes |
| id | easyanimate | 400 | 0.74 | 0.78 | yes |
| id | hunyuanvideo | 100 | 0.26 | 0.32 | yes |
| id | ltxvideo | 200 | 0.34 | 0.34 | yes |
| ood | gen3 | 100 | 0.83 | 0.86 | yes |
| ood | jimeng | 100 | 0.68 | 0.73 | yes |
| ood | kling | 100 | 0.75 | 0.81 | yes |
| ood | luma | 100 | 0.75 | 0.75 | yes |
| ood | pika | 100 | 0.73 | 0.84 | yes |
| ood | sora | 200 | 0.85 | 0.90 | yes |
| ood | vidu | 150 | 0.78 | 0.86 | yes |
| ood | wanx | 150 | 0.81 | 0.82 | yes |
| wild | gen4.5 | 10 | 0.50 | 0.70 | yes |
| wild | hailuo2.3 | 20 | 0.40 | 0.55 | no |
| wild | kling2.5_turbo | 15 | 0.40 | 0.53 | yes |
| wild | kling2.6 | 20 | 0.50 | 0.65 | yes |
| wild | pixverse_v5 | 10 | 0.60 | 0.60 | no |
| wild | seedance1.5_pro | 20 | 0.20 | 0.25 | yes |
| wild | seedance2.0 | 15 | 0.33 | 0.53 | yes |
| wild | sora2 | 10 | 0.50 | 0.60 | yes |
| wild | veo3.1 | 10 | 0.60 | 0.80 | yes |
| wild | wan2.5 | 10 | 0.50 | 0.40 | yes |
| wild | wan2.6 | 10 | 0.80 | 0.60 | yes |

## Leave-one-family-out AUC on the pool

| family | shipped | retrain |
|---|---|---|
| cogvideox | 0.722 | 0.713 |
| easyanimate | 0.680 | 0.679 |
| hunyuanvideo | 0.748 | 0.925 |
| ipoc | 0.749 | 0.756 |
| jimeng | 0.974 | 0.979 |
| kling | 0.909 | 0.919 |
| ltx | 0.552 | 0.543 |
| luma | 0.930 | 0.920 |
| marey | nan | 0.915 |
| mochi | nan | 0.960 |
| opensora | 0.959 | 0.965 |
| pika | 0.954 | 0.943 |
| pyramid-flow | 0.637 | 0.624 |
| runway | 0.941 | 0.952 |
| seedance | nan | 0.927 |
| seine | 0.538 | 0.547 |
| sora | 0.947 | 0.944 |
| svd | 0.716 | 0.666 |
| veo | nan | 0.935 |
| vidu | 0.916 | 0.949 |
| wan | 0.880 | 0.905 |
