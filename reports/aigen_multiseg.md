# AI-video detector: whole-clip scoring — 2026-11-06 12:37

Up to 3 evenly spaced 2 s segments per clip; segments per clip: 1: 1155, 2: 5856, 3: 1644. Model recipe and C (0.01) from the shipped bundle. Rule fixed in advance (script docstring): the best of avg / avg-train on the pool replaces first only if every gate holds.

## Pool (selection)

| variant | macro LOGO AUC | worst family | threshold (OOF reals, 5%) |
|---|---|---|---|
| first | 0.8414 | 0.543 | +1.1149 |
| avg | 0.8454 | 0.552 | +1.0372 |
| avg-train | 0.8449 | 0.546 | +0.9678 |

Best candidate on the pool: **avg**.

| gate | pass | value |
|---|---|---|
| pool macro LOGO AUC >= first's | yes | 0.8454 vs 0.8414 |
| GenBuster-real FPR <= 6% | yes | 0.050 (first 0.054) |
| id recall >= first's | yes | 0.534 vs 0.522 |
| ood recall >= first's | yes | 0.840 vs 0.831 |
| wild recall >= first's | **no** | 0.540 vs 0.547 |
| FF++ c23 test reals flagged <= 2 | yes | 0/135 |

**Rule says: keep first.**

## Segments per clip, by label (all clips)

| label | 1 seg | 2 seg | 3 seg |
|---|---|---|---|
| ai | 968 | 4438 | 757 |
| real | 187 | 1418 | 887 |

## External (gates adoption; never chooses between candidates)

| measure | first | avg | avg-train |
|---|---|---|---|
| auc_id | 0.880 | 0.886 | 0.883 |
| auc_ood | 0.961 | 0.964 | 0.963 |
| auc_wild | 0.880 | 0.886 | 0.875 |
| fpr_ffpp_real | 0.000 | 0.000 | 0.000 |
| fpr_genbuster_real | 0.054 | 0.050 | 0.060 |
| median_score_ai | 1.828 | 1.834 | 1.893 |
| median_score_real | -1.442 | -1.468 | -1.462 |
| n_ai_id | 1000 | 1000 | 1000 |
| n_ai_ood | 1000 | 1000 | 1000 |
| n_ai_wild | 150 | 150 | 150 |
| n_ffpp_real | 135 | 135 | 135 |
| n_genbuster_real | 1000 | 1000 | 1000 |
| recall_id | 0.522 | 0.534 | 0.556 |
| recall_ood | 0.831 | 0.840 | 0.864 |
| recall_wild | 0.547 | 0.540 | 0.527 |
