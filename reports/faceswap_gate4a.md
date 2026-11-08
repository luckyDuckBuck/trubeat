# Face-swap detector, gate 4a — 2026-11-08 23:10

Bundle `faceswap_v2.joblib` (SHA-256 `fa375d3d7f4e…`; probe, backbone vit_base_patch16_clip_224.openai, cls+patch, finder twopass, threshold +3.1987).

Clips: the frozen gate-4a list `faceswap_gate4a.csv` (706 GenBuster-Bench and AIGVDBench real clips, every file's SHA-256 checked), every clip processed: 706 processed, **383 judged** (a face in at least 4 of 16 sampled frames), 0 not judged because of an error or a too-short clip.

Flagged as face-swap: **61/383 judged = 15.9%** [12.6%, 19.9%] (Wilson 95%).

**Gate (<= 3% of judged, n judged >= 200, no source above 6%): fail.**

| source | processed | judged | flagged (95% CI) |
|---|---|---|---|
| aigvdbench | 391 | 197 | 18/197 = 9.1% [5.9%, 14.0%] |
| genbuster_id | 315 | 186 | 43/186 = 23.1% [17.6%, 29.7%] |
