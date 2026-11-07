# Face finder v2: adoption check (docs/13 §6.1) — 2026-11-07 10:37

Data: 259 FaceForensics++ c23 test clips on this laptop (140 real, 119 fake), 16 sampled frames each (4144 frames; most common sizes: 640x480 (1216), 1280x720 (1136), 1920x1080 (576), 600x480 (336)). Round-3 finder = the Face Landmarker on the whole frame; two-pass = that same pass first, and only when it finds nothing, YuNet on a <= 1280 px copy followed by the Landmarker on a 2x crop of each detection. Runtime 9 min.

**Decision: adopt the two-pass finder** (rule 1 pass, 2 pass, 3 pass, 4 pass).

## Rule 1: small faces pasted into a scene (found and landmarked)

38 faces from distinct test reals, each pasted as a region 1.6x its landmark box into a frame of another clip resized to the canvas, with that frame's own faces filled flat and checked absent (both finders and YuNet at score 0.3 find nothing); success = the returned box centre within 0.35 x the face size of the true centre and its face size within 0.6-1.6 x. Wilson 95% intervals.

| canvas | face | round-3 finder | two-pass | pass (>= 95%) |
|---|---|---|---|---|
| 1280x720 | 30 px | 0/38 = 0% | 0/38 = 0% [0%, 9%] | reported (< min_face_px) |
| 1280x720 | 60 px | 0/38 = 0% | 38/38 = 100% [91%, 100%] | yes |
| 1280x720 | 100 px | 0/38 = 0% | 38/38 = 100% [91%, 100%] | yes |
| 1280x720 | 200 px | 38/38 = 100% | 38/38 = 100% [91%, 100%] | yes |
| 1280x720 | 400 px | 38/38 = 100% | 38/38 = 100% [91%, 100%] | yes |
| 1920x1080 | 30 px | 0/38 = 0% | 0/38 = 0% [0%, 9%] | reported (< min_face_px) |
| 1920x1080 | 60 px | 0/38 = 0% | 38/38 = 100% [91%, 100%] | yes |
| 1920x1080 | 100 px | 0/38 = 0% | 38/38 = 100% [91%, 100%] | yes |
| 1920x1080 | 200 px | 8/38 = 21% | 38/38 = 100% [91%, 100%] | yes |
| 1920x1080 | 400 px | 38/38 = 100% | 38/38 = 100% [91%, 100%] | yes |

## Rule 2: no regression on the test clips

| label | clips | round-3: face in >= 12/16 | two-pass: >= 12/16 | round-3: >= 4/16 | two-pass: >= 4/16 | pass |
|---|---|---|---|---|---|---|
| fake | 119 | 113 | 119 | 114 | 119 | yes |
| real | 140 | 131 | 140 | 134 | 140 | yes |

## Rule 3: same crop box where both find the face

Frames where both found a face: 3927; boxes within 5% of the side (max of centre and side differences): **3927/3927 = 100.0%** (rule: >= 95%) -> pass. Median difference 0.000 of the side.

Two-pass source of the returned face: landmarker 3927, yunet 217.

## Rule 4: cost

YuNet detection on the sampled frames: mean 43 ms per frame over all sizes; 71 ms on frames of at least 1280x720 (1712 frames). Rule: <= 150 ms per 1080p frame -> pass.

## Item 5 (reported, not a rule): the user's extension capture

The capture as recorded (the whole tab at Chrome's default size, 960x540), 16 sampled frames: round-3 finder 0/16, two-pass 16/16 (a face in >= 12 of 16: yes); median largest YuNet detection 66 px (faces under `min_face_px`, 40 px, are never used). The extension's own changes (full-resolution capture, crop to the player) are tested in docs/13 §6.4.

