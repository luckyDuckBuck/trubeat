# AI-video detector: external clips that are near-copies of training clips — 2026-11-07 12:43

Bundle `aigen.joblib` (SHA-256 `5aec92d96e76…`, threshold +1.1149, unchanged). Feature table `aigen_features_v2.pkl`: 5370 pool (training) clips, 3285 external clips. A clip is a near-copy when its mean DINOv2 embedding (the 384 features the model scores) has cosine ≥ 0.99 with a pool clip.

**58 external clips are near-copies** (cosine median 1.0000, min 0.9913). The leakage check by group key could not see them: GenBuster-Bench and AIGVDBench name their clips differently. Of the other external clips, the most similar pool clip has cosine 0.9899 at most (median 0.5655).

| source | label | generator | near-copies |
|---|---|---|---|
| genbuster_id | real | real | 4 |
| genbuster_ood | ai | gen3 | 6 |
| genbuster_ood | ai | jimeng | 5 |
| genbuster_ood | ai | kling | 12 |
| genbuster_ood | ai | luma | 3 |
| genbuster_ood | ai | pika | 5 |
| genbuster_ood | ai | sora | 10 |
| genbuster_ood | ai | vidu | 2 |
| genbuster_ood | ai | wanx | 11 |

## External numbers, as reported and without the near-copies

| metric | as reported (all external clips) | without near-copies | AUC vs GenBuster reals |
|---|---|---|---|
| false alarms, GenBuster reals | 54/1000 = 0.054 [0.042, 0.070] | 53/996 = 0.053 [0.041, 0.069] | — |
| false alarms, FF++ c23 test reals | 0/135 = 0.000 [0.000, 0.028] | 0/135 = 0.000 [0.000, 0.028] | — |
| recall, GenBuster in-domain tier | 522/1000 = 0.522 [0.491, 0.553] | 522/1000 = 0.522 [0.491, 0.553] | 0.880 → 0.880 |
| recall, GenBuster out-of-domain tier (2024-era tools) | 831/1000 = 0.831 [0.807, 0.853] | 786/946 = 0.831 [0.806, 0.853] | 0.961 → 0.961 |
| recall, GenBuster wild tier (the newest) | 82/150 = 0.547 [0.467, 0.624] | 82/150 = 0.547 [0.467, 0.624] | 0.880 → 0.881 |

Rates are over the clips scored; intervals are Wilson 95%. The model, its threshold and every clip's score are unchanged; only the near-copies are left out of the denominators.
