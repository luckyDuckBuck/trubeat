# AI-video detector: input robustness — 2026-11-06 08:50

Bundle `runs/aigen_detector_v2.joblib` (created 2026-11-01 14:19, config `3f757f1f`), threshold +1.1149 fixed. External clips only (never used for training, selection or the threshold).

Subset: ffpp_original real 135, genbuster_id ai 160, genbuster_id real 200, genbuster_ood ai 320, genbuster_wild ai 150.

| measure | training normalisation (360p, CRF 28) | harsh re-encode (240p, CRF 35) + app normalisation | tab-capture-like (VP9 WebM, 720p, 2.5 Mb/s) + app normalisation | real-world downscale to 480p + app normalisation | upscale to 1080p + app normalisation | as-is (no re-encode: the app without ffmpeg) |
|---|---|---|---|---|---|---|
| auc_id | 0.849 | 0.807 | 0.847 | 0.847 | 0.846 | 0.867 |
| auc_ood | 0.957 | 0.945 | 0.957 | 0.957 | 0.957 | 0.959 |
| auc_wild | 0.878 | 0.837 | 0.878 | 0.877 | 0.878 | 0.899 |
| fpr_ffpp_real | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| fpr_genbuster_real | 0.055 | 0.035 | 0.055 | 0.045 | 0.050 | 0.055 |
| median_score_ai | 1.796 | 1.050 | 1.775 | 1.730 | 1.744 | 1.981 |
| median_score_real | -1.853 | -2.072 | -1.836 | -1.877 | -1.895 | -1.601 |
| n_ai_id | 160 | 160 | 160 | 160 | 160 | 160 |
| n_ai_ood | 320 | 320 | 320 | 320 | 320 | 320 |
| n_ai_wild | 150 | 150 | 150 | 150 | 150 | 150 |
| n_ffpp_real | 135 | 135 | 135 | 135 | 135 | 135 |
| n_genbuster_real | 200 | 200 | 200 | 200 | 200 | 200 |
| recall_id | 0.444 | 0.294 | 0.419 | 0.419 | 0.438 | 0.531 |
| recall_ood | 0.828 | 0.619 | 0.819 | 0.819 | 0.825 | 0.859 |
| recall_wild | 0.547 | 0.353 | 0.533 | 0.513 | 0.527 | 0.633 |

AUC per tier is against the GenBuster reals in the subset; recall and FPR are at the fixed threshold. `n_*` columns are clip counts.
