# AI-video detector: input robustness — 2026-11-01 13:50

Bundle `/home/ayushman/Projects/trubeat/runs/aigen_detector.joblib` (created 2026-11-01 11:37, config `5c84ab9d`), threshold +1.2572 fixed. External clips only (never used for training, selection or the threshold).

Subset: ffpp_original real 135, genbuster_id ai 80, genbuster_id real 200, genbuster_ood ai 160, genbuster_wild ai 150.

| measure | training normalisation (360p, CRF 28) | harsh re-encode (240p, CRF 35) + app normalisation | tab-capture-like (VP9 WebM, 720p, 2.5 Mb/s) + app normalisation | real-world downscale to 480p + app normalisation | upscale to 1080p + app normalisation | as-is (no re-encode: the app without ffmpeg) |
|---|---|---|---|---|---|---|
| auc_id | 0.852 | 0.815 | 0.851 | 0.852 | 0.848 | 0.865 |
| auc_ood | 0.957 | 0.940 | 0.957 | 0.955 | 0.956 | 0.955 |
| auc_wild | 0.866 | 0.821 | 0.864 | 0.865 | 0.866 | 0.886 |
| fpr_ffpp_real | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| fpr_genbuster_real | 0.045 | 0.025 | 0.040 | 0.040 | 0.040 | 0.055 |
| median_score_ai | 1.498 | 0.832 | 1.444 | 1.444 | 1.464 | 1.723 |
| median_score_real | -1.478 | -1.622 | -1.442 | -1.444 | -1.478 | -1.398 |
| n_ai_id | 80 | 80 | 80 | 80 | 80 | 80 |
| n_ai_ood | 160 | 160 | 160 | 160 | 160 | 160 |
| n_ai_wild | 150 | 150 | 150 | 150 | 150 | 150 |
| n_ffpp_real | 135 | 135 | 135 | 135 | 135 | 135 |
| n_genbuster_real | 200 | 200 | 200 | 200 | 200 | 200 |
| recall_id | 0.425 | 0.212 | 0.388 | 0.412 | 0.388 | 0.475 |
| recall_ood | 0.775 | 0.519 | 0.775 | 0.775 | 0.769 | 0.800 |
| recall_wild | 0.453 | 0.260 | 0.440 | 0.447 | 0.453 | 0.527 |

AUC per tier is against the GenBuster reals in the subset; recall and FPR are at the fixed threshold. `n_*` columns are clip counts.
