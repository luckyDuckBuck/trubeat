# Extension v1.3.0: crop to the video (docs/13 §6.4) — 2026-11-07

**Rule (§6.4, fixed before the check).** A synthetic page recording goes
through the extension's backend path with the video's box. It passes if the
face-swap detector finds a face in ≥ 12 of 16 frames and the AI-video layer
scores the cropped region. Together with §6.1 item 5 on the user's real
capture, a pass means v1.3.0 ships with the crop on.

**Fixture.** FaceForensics++ c23 `000.mp4` (640×480, 25 fps, a news
presenter; real), scaled to 960×720 and composited into a 1920×1080 page at
(240, 200)–(1200, 920), with a dark top bar, a grey side panel and a
progress bar. It is encoded as VP9 WebM at 2.5 Mb/s in real-time mode (the
extension's MediaRecorder settings) and is 15 s long. It stays in a local
scratch folder and is not committed, because FaceForensics++ may not be
redistributed.

**Backend.** `trubeat serve` at this commit's code, with the shipped bundles
(face-swap detector round 3, evidence only, Face Landmarker finder).

| check | with the video's box | whole tab |
|---|---|---|
| extension code path (`tests/js/extension_print_result.mjs`), verdict | real (region sent to `/analyze_video_region`) | — |
| face-swap detector: frames with a face | **16 / 16** | 6 / 16 |
| AI-video layer: frames it scored | **480×360** (the 960×720 crop) | 640×360 (the 1920×1080 page) |
| AI-video score (threshold +1.115) | −2.96 | −2.87 |
| verdict | No sign of AI generation found | No sign of AI generation found |

**§6.1 item 5** (the user's own capture, two-pass finder): a face in 16 / 16
frames, against 0 / 16 with the round-3 finder (`reports/facefind.md`).

**Outcome: pass.** v1.3.0 analyses only the playing video's box when the
box stayed put during the recording, and falls back to the whole tab
otherwise. The panel says which was used.

Limits: one synthetic page and one real capture. The fixture's player
holds a single large face. Cross-site embedded players are probably
analysed as the whole tab, because Chrome hides their frames from an
extension with only `activeTab` (extension README). Nothing here was run in
a real Chrome; the README's QA checklist covers that before release.
