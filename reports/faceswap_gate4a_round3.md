# Face-swap detector, gate 4a — 2026-11-07 11:42

Bundle `faceswap.joblib` (SHA-256 `e0d39bfe13b2…`; probe, backbone vit_base_patch16_clip_224.openai, cls, finder landmarker, threshold +1.5772).

Clips: the frozen gate-4a list `faceswap_gate4a.csv` (706 GenBuster-Bench and AIGVDBench real clips, every file's SHA-256 checked), every clip processed: 706 processed, **300 judged** (a face in at least 4 of 16 sampled frames), 0 not judged because of an error or a too-short clip.

Flagged as face-swap: **115/300 judged = 38.3%** [33.0%, 43.9%] (Wilson 95%).

**Gate (<= 3% of judged, n judged >= 200, no source above 6%): fail.**

| source | processed | judged | flagged (95% CI) |
|---|---|---|---|
| aigvdbench | 391 | 142 | 9/142 = 6.3% [3.4%, 11.6%] |
| genbuster_id | 315 | 158 | 106/158 = 67.1% [59.4%, 73.9%] |
