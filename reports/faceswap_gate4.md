# Face-swap detector, gate 4 — 2026-11-07 07:32

Bundle: the Colab result (SHA-256 `ddf4242d…`), installed by `train_faceswap.py ship` as `src/trubeat/models/faceswap.joblib` (probe, threshold +1.5772).

Real, non-FaceForensics clips (GenBuster-Bench + AIGVDBench reals) in a fixed shuffled order: 706 scored, 300 with a face in at least 4 sampled frames.

Flagged as face-swap: **115/300 = 38.3%** [33.0%, 43.9%].

**Gate 4 (<= 3%, n >= 200): fail.**

| source | with a face | flagged |
|---|---|---|
| aigvdbench | 142 | 9 |
| genbuster_id | 158 | 106 |
