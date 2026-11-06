# 11 — Audit and improvement plan (2026-11-01)

**Status: APPROVED to execute (human, 2026-11-01):** *"start running whatever
does not need me; if any downloads are needed also do [them]"*. That covers
every task with no human gate in §6, plus public downloads (dependencies,
model weights, freely downloadable datasets). Anything behind an account,
sign-up form or licence click-through stays a human gate and is logged in
`reports/BLOCKERS.md`. This plan adds work that `docs/07` and `docs/10` do not
contain (a new detector, a changed verdict contract); changes to the frozen
interfaces in `docs/04` §4 still get a doc edit and a commit note.

---

## 1. Bottom line

The complaint: *"results are usually inconclusive even when I know whether the
video is AI or not; it's unable to predict properly."* The audit confirms it
and finds three root causes, in order of impact:

1. **The checks that decide "Inconclusive" are measuring the wrong thing.**
   The SNR check that blocks verdicts is a measure of slow drift and clip
   length, not of a pulse (§3, F1). The 10-second minimum rejects most AI
   generator output outright, since those clips usually run 4–15 s (F3).
   The face tracker misses small or wide-shot faces (F6).
2. **The logic is inverted for AI video.** "No pulse found" ends the
   analysis as Inconclusive. But a fully generated face has no real blood
   flow, so for that kind of video a missing pulse is evidence, not a reason
   to stop (F2). Your own upload shows this exactly: every signal pointed to
   fake, and the SNR check blocked the verdict by 0.46.
3. **Even when it decides, the detector is close to a coin flip, and it was
   never built for AI-generated video.** It was trained on a few hundred
   2019-era face-swap clips (FaceForensics++ c23) using 10 hand-made pulse
   features. On held-out test clips, 57% of its decisive verdicts are right;
   always answering "real" scores 54%. It wrongly flags 22% of real videos
   as fake (F4). It has never seen a Sora, Veo, Kling or Runway clip (F5).

**What follows from this:** relaxing the thresholds would turn Inconclusive
into coin flips, not into correct answers. Making the tool "predict properly"
on AI video needs a different primary detector, one that looks at the whole
frame and how it changes over time. The pulse analysis then becomes one
supporting signal: useful on close-up, well-lit, high-bitrate face video and
honestly silent elsewhere. §4 describes that design and §5 lays out the plan.

---

## 2. How this audit was done (reproducible)

| check | how | result |
|---|---|---|
| Live app | `trubeat serve`, driven through the Gradio API (`/run`) on FF++ clips and on the user's upload in `/tmp/gradio/` | works end to end; ~12 s for a 16 s clip |
| User's clip | `analyze_video()` on `istockphoto-2189556850-640_adpp_is.mp4` (768×432, 25 fps, 10.88 s) | Inconclusive; see the table below |
| Verdict replay | `scripts/audit_verdicts.py`: the app's own functions replayed on all 583 cached FF++ traces | matches the live app exactly on the clips cross-checked (000, Deepfakes 033_097) |
| SNR check | same script: the app's `_mean_snr` on white noise, random-walk drift and a weak 72 bpm pulse | §3, F1 |
| Tests | `pytest tests` | 141 passed, 2 skipped; 1 environment-dependent failure (F12) |
| Literature | papers and vendor pages (sources in §8) | §4 |

The user's clip, stage by stage:

| stage | value | outcome |
|---|---|---|
| length | 10.88 s | passes the 10 s minimum by 0.88 s |
| face tracked | 100% of frames | pass |
| per-region heart rate | forehead 48, cheek_l 115, cheek_r 111 bpm (std 30.6) | regions strongly disagree |
| detector score | −0.77 (cutoff −0.013) | well on the fake side |
| **SNR** | **9.54 (gate 10)** | **blocks the verdict → Inconclusive** |

Reproduce everything: `python scripts/audit_verdicts.py` (writes
`runs/audit_verdicts.csv`).

---

## 3. Findings

Severity: **Critical** = directly causes the reported failure; **High** =
makes verdicts wrong or unreliable; **Medium** = defect with limited reach;
**Low** = hygiene.

### 3.1 Why verdicts come back Inconclusive

**F1 — Critical — The SNR check does not detect a pulse.**
`app._mean_snr` (`src/trubeat/app.py:285`) is the in-band peak power divided
by the mean, over 0.7–14 Hz, computed on a signal that has had its mean
removed but has not been detrended or filtered. Measured with the app's own
function (median of 300 draws, pass mark 10):

| signal | 10.9 s | 30 s | 120 s |
|---|---|---|---|
| white noise, no pulse | 5.5 | 6.3 | 7.7 |
| random-walk drift, **no pulse** | **22.5** | **32.0** | **45.8** |
| weak 72 bpm pulse + noise | **7.0** | 16.6 | 64.4 |

Drift with no pulse passes easily. A genuine weak pulse fails on a short
clip. The value also grows with clip length (Spearman 0.29 against duration
on FF++ real clips). So a single cutoff at 10 sits between noise (5.5) and a
real short-clip pulse (7.0) with no margin, and it decides Inconclusive.

**F2 — Critical — "No pulse" ends the analysis, even when that is the
evidence.** `verdict_for` (`src/trubeat/app.py:311`) returns Inconclusive
whenever SNR < 10 (`:339`, `:347`), before the detector's score is even
considered. A fully generated face has no real blood flow, so for AI video a
missing pulse is the signal. *Caveat (do not over-correct):* FF++ shows that
real web video often has no recoverable pulse either (`reports/g2_detector.md`,
R-01: the regions disagree by ~8.6 bpm in both classes). "No pulse" can only
count as evidence when the capture conditions say a pulse *should* be
visible. That model is task B4.

**F3 — Critical — The 10-second minimum rejects most AI clips.**
`ingest.MIN_DURATION_S = 10.0` (`src/trubeat/ingest.py:12`) raises
`E_SHORT_VIDEO`, and the 10 s feature window means a shorter clip produces
no detector score at all. Current generators output about 8 s (Veo 3.1),
about 15 s (Kling 3.0) and 12–20 s (Sora 2, depending on plan), and short-form
reposts are often trimmed further. Your clip cleared the minimum by 0.88 s.

**F6 — High — Faces that are small in the frame are not found.**
`face.track_landmarks` runs MediaPipe Face Landmarker on the whole frame
(`src/trubeat/face.py:56`), after ingest has already downscaled every frame
to 480 px tall (`ingest.py:27`). Its internal detector is tuned for close-up,
selfie-distance faces. FF++ Deepfakes 033_097 is a TV presenter whose face is
clearly visible, yet it was tracked in 0 of 809 frames, and 16 FF++ clips
fail the same way. A vertical 1080×1920 phone or Shorts video is decoded at
270×480 before detection.

### 3.2 Why decisive verdicts are unreliable

**F4 — Critical — The detector is close to chance.** On the FF++ test split
(255 clips), replayed through the app:

| | Deepfakes | Face2Face | FaceSwap | NeuralTextures | real |
|---|---|---|---|---|---|
| correct (decisive verdicts) | 61% | 26% | 31% | 15% | 78% |
| detector AUC vs reals | 0.69 | 0.47 | 0.60 | 0.41 | — |

Pooled AUC is 0.544, and 57% of decisive verdicts are right, against 54% for
always answering "real". 28 of 127 real test clips (22%) are flagged fake.
This matches the G2 report (AUC 0.543, false-positive rate 0.237), which
already recorded G2 as **failed**. The app ships the failed model with a
caveat. The caveat is honest, but it means a decisive answer from the app
carries little information.

**F5 — Critical — Nothing in the system targets fully AI-generated video.**
The training data, features and caveat are all about manipulating a real
recorded face (face swap or reenactment). Text-to-video and image-to-video
output has no "real performer" underneath. Its tells are in the whole frame
and in how it moves over time: physics, texture statistics, the path frames
trace through a vision model's feature space. The current pipeline looks at
three skin patches only.

**F7 — High — Heart-rate reading locks onto drift, not the pulse.** The
bundled model uses `method: green`, which returns the raw green channel
(`signal.py:110`) with no band-pass filter. `hr.estimate_hr` only removes a
linear trend (`hr.py:57`). On FF++ real clips, **55% have at least one region
reading under 50 bpm** (just above the 42 bpm lower band edge, where slow
drift lands), and only 39% meet the app's own 5 bpm "regions agree" bar. The
coherence and correlation features are also computed on the unfiltered
signal, so they mostly measure lighting changes shared across the whole face,
not the pulse. That explains their weak separating power (agreement AUC 0.54).

**F8 — High — The cutoffs are hard edges, and confidence isn't calibrated.**
Every check is a single threshold: 9.5 vs 10 flips the result. The verdict
carries no probability, against `AGENTS.md` §6 ("every prediction carries
confidence"). The detector threshold was tuned on the small validation split
(82 manifest rows), and its false-positive rate already drifted from 0.20
(validation) to 0.24 (test).

### 3.3 Defects found while running

| ID | sev | defect | where |
|---|---|---|---|
| F9 | Medium | `trubeat predict --model src/trubeat/models/detector.joblib` crashes: `Feature names unseen at fit time: win_end_s, win_start_s`. The CLI passes every feature column to the model; the app uses the bundle's `feature_columns`. | `model.py:112` vs `app.py:244` |
| F10 | Medium | When no face is found, a meaningless detector score (+4.93, computed from zero-filled features) still appears in the evidence table. | `app.py:660` |
| F11 | Medium | The plot is titled "conditioned, 0.7-3 Hz band" but shows the raw green channel. The title and axis labels are near-black on the `#0D0D0D` background and unreadable. | `app.py:684–686` |
| F12 | Low | Plain `pytest` (the README's command) also collects `third_party/rppg-toolbox` tests, which need `torch`, and fails. When any server is up on :7860, the extension test switches to live mode and loads a multi-GB raw UBFC `.avi`, which got it OOM-killed on this 7 GB machine (kernel log 05:13:23). | `pyproject.toml` (no `testpaths`), `tests/js/extension_test.mjs:51–57` |
| F13 | Low | `trubeat train` still trains on the window-position columns (a known wart). Its models and the G2 bundle are not interchangeable, and nothing checks. | `cli.py:87` |
| F14 | Low (unmeasured) | The extension always captures 20 s. On an 8 s clip that capture includes loop seams, or a frozen end frame that tracks as a valid face with no signal. | `extension/background.js:25` |

---

## 4. Target design

```
                 ┌──────────────────────────────┐
 video ──► [0] provenance   C2PA manifest, generator metadata, visible watermark
       │         → definitive "AI-generated (declared)" when present
       ├──► [1] AI-video detector (NEW, primary)
       │         sampled frames → frozen vision backbone → temporal-geometry
       │         stats (curvature, step size, second-order differences)
       │         → calibrated P(AI-generated)
       ├──► [2] face-manipulation detector (existing rPPG, repaired)
       │         only when a face is large/clean enough; reports pulse
       │         consistency + "pulse expected but absent"
       └──► [3] capture-quality model (face px, bitrate, motion, light)
                 → weights how much [2] can say
                 ▼
          fusion → P(fake) with CI-backed calibration
                 → verdict: likely AI / likely real / not enough evidence
                   + which layer drove it + the evidence table
```

Design decisions:

- **Layer [1] is the primary detector for AI video.** Two 2025 methods show
  that frozen foundation-model features plus light statistics can detect
  generated video without training a big network: **D3**
  (training-free, second-order temporal differences; ICCV 2025) and
  **ReStraV** (frame paths through DINOv2 feature space are less "straight"
  in generated video, with a lightweight classifier; NeurIPS 2025). Both suit
  a CPU laptop: DINOv2-small on 16 sampled frames is a few seconds per clip.
  Treat their reported accuracies as the authors' in-distribution claims;
  our number is whatever leave-one-generator-out evaluation gives (§5, D5).
- **Larger trained detectors are a later option, not the start.** DeMamba
  (with the GenVideo benchmark) and UNITE (SigLIP features, full-frame, built
  for both face edits and text/image-to-video; CVPR 2025) are the reference
  points if the light approach plateaus. Check licences before use.
- **Provenance first, never alone.** C2PA manifests (OpenAI states it adds
  Content Credentials to Sora output) and Google's SynthID are high
  precision when present. C2PA metadata is lost on re-encode, so screen and
  tab captures have none. SynthID survives better, but it is checked
  through Google's own tools; I found no offline library to verify it
  locally. Absence proves nothing.
- **rPPG becomes a specialist.** It answers only when layer [3] says a pulse
  should be visible. Otherwise it says "not applicable", not "Inconclusive".
- **Three outcomes, with probabilities.** Inconclusive is reserved for "no
  layer has evidence", and the UI shows P(fake) with its calibration source.

---

## 5. Plan

Effort assumes one developer on this laptop (12 cores, 7 GB RAM, no CUDA).
Every task follows `AGENTS.md`: config-driven, tests first, one commit per
task. Each phase ends with a short report in `reports/`.

### Phase A — Fix the defects (1–2 days; no interface changes)

| task | what | done when |
|---|---|---|
| A1 | Commit `scripts/audit_verdicts.py` as the regression baseline for every later phase | its output is quoted in each phase report |
| A2 | F9: `predict_video` selects the bundle's `feature_columns`; `cli predict` accepts a G2 bundle | `trubeat predict --model src/trubeat/models/detector.joblib <clip>` prints JSON; test added |
| A3 | F10: do not compute the detector score when the face is not tracked (zero-filled input). The score stays when only SNR fails, since that is informative and B2 replaces the SNR check | no-face test asserts `detector_score is None` |
| A4 | F11: plot what is actually analysed (band-passed) and colour the title and labels | snapshot test on the label colours; title matches the processing |
| A5 | F12: `testpaths = ["tests"]`; the extension harness's live mode needs explicit opt-in (`TRUBEAT_LIVE=1`) and a small clip | plain `pytest` green with or without a server running |
| A6 | F13: `cli train` drops the position columns and saves `feature_columns` in the bundle; `predict` checks them | train → predict round-trip test |

### Phase B — Repair the pulse measurement (3–5 days)

| task | what | done when |
|---|---|---|
| B1 | F7: band-pass (0.7–3 Hz) every method's output before heart-rate and consistency features. Retrain G2 on the conditioned features (the bundle depends on them) | UBFC: per-region spread ≤ 2 bpm median; FF++ real share of regions < 50 bpm drops from 55% to < 15% |
| B2 | F1: replace `_mean_snr` with the standard rPPG SNR (de Haan & Jeanne 2013; reference implementation already vendored at `third_party/rppg-toolbox/evaluation/post_process.py:78`): power within ±6 bpm of the pulse frequency and of twice that frequency, against the rest of the pulse band, in dB. With no ground truth, the pulse frequency is the estimated peak, so the test must also cover "noise that happens to peak somewhere" | on synthetic tests: noise and drift pass ≤ 5%, a 72 bpm pulse at realistic amplitude passes ≥ 95%, and the result varies < 2 dB across 10/30/120 s |
| B3 | F6: detect the face first with a full-range detector (OpenCV `FaceDetectorYN` is in the installed cv2 4.9 and needs the small YuNet ONNX file from the OpenCV model zoo; or MediaPipe's full-range Face Detector), crop with a margin at native resolution, then run the landmarker on the crop. Memory: decode at native resolution only around the face box, not whole frames (the 480 px cap exists to avoid OOM) | 033_097 and the 16 NO_FACE FF++ clips track ≥ 70% of frames; a vertical 1080×1920 test clip tracks |
| B4 | F2 support: a capture-quality score (face size in pixels, bits per pixel, motion, exposure) and an "expected pulse visibility" model fitted on UBFC (visible) vs FF++ (mostly not) | reports P(pulse visible \| conditions) with a reliability curve |
| B5 | Re-run the G2 evaluation with B1–B3, same protocol, bootstrap CIs | report with per-family AUCs; honest whichever way it goes |

#### Phase B result so far (2026-11-01, overnight run): the pulse is not measurable on FF++ c23

B2 is implemented (`hr.pulse_snr_db`, tested). On synthetic data it meets its
bar: noise and drift sit at −3 to −8 dB, with a 95th percentile of ≤ −0.1 dB
at every length from 5 to 120 s. A clear pulse (amplitude equal to the noise
std, heart rate wandering) scores 1–5 dB. The old metric passed drift at
22–46 and grew with length. **0 dB** is therefore the pulse-visible threshold.

Measured on the 534 cached FF++ c23 clips that pass the face and length
checks, using filtered signals (`runs/b2_snr_ffpp.csv`):

| method | real-clip median SNR | real clips ≥ 0 dB | AUC of low SNR → fake (DF / F2F / FS / NT) |
|---|---|---|---|
| green (filtered) | −3.5 dB | **0%** | 0.57 / 0.47 / 0.56 / 0.47 |
| POS | −4.0 dB | **0%** | 0.50 / 0.50 / 0.42 / 0.45 |
| CHROM | −3.9 dB | 1% | 0.47 / 0.50 / 0.45 / 0.51 |

The user's iStock clip reads −6 to −8 dB per region. **Real web video looks
like noise to this pipeline**, which confirms R-01 with the standard metric.
The old SNR check passed these clips only because it counted drift, and the
G2 detector's Deepfakes signal (AUC 0.69) cannot be physiological, since
there is no measurable pulse to be inconsistent. Filtering lowered the
share of real clips with a region under 50 bpm from 55% to 39%; the rest is
noise peaking anywhere, which filtering cannot fix.

**Re-plan (keep only changes that improve the tool):**

- Not wiring B2 in as a hard gate: on web video it would make ~100% of clips
  Inconclusive, the exact failure being fixed. It becomes the "is the pulse
  layer applicable?" signal in the fused verdict (E1), reported as evidence.
- **Deferred: B1b, B3, B4, B5.** They improve a measurement that is below the
  noise floor on the target material, so they can't change verdicts there.
  They return if the user's labelled set (F-1) shows close-up,
  high-quality clips where the pulse is measurable.
- **D moves up** to straight after A. The improvement the user can see comes
  from the whole-frame AI-video detector.

### Phase C — Verdict contract (2–3 days; changes the `PulseAnalysis` schema → doc edit to `docs/04`)

| task | what | done when |
|---|---|---|
| C1 | Soft checks: each check scales down confidence instead of acting as a cliff; "Inconclusive" only when no layer has evidence | coverage vs accuracy curve published; Inconclusive rate on FF++ test reported at a fixed false-positive rate |
| C2 | Calibrated probability (Platt or isotonic on validation) and an operating point picked for false-positive rate ≤ 10% | reliability diagram in the report |
| C3 | F3: allow 5–10 s clips with a 5 s window for consistency and AI-video features (heart rate is only reported for ≥ 10 s) | 8 s clips give a scored result; the test asserts the HR field says "too short to report" |
| C4 | The headline names the layer and the probability, e.g. "Likely AI-generated (0.86): whole-frame detector; pulse analysis not applicable (face too small)" | UI test |

### Phase D — AI-generated video detector (the "script detection" ask; 2–3 weeks)

| task | what | done when |
|---|---|---|
| D1 | **Data** (human gate): a real vs generated set with the generator recorded per clip. Sources: the GenVideo (DeMamba) and GenVidBench subsets, clips from open-weight generators (e.g. Wan, HunyuanVideo; check the licence). These need a GPU, so they run on a free cloud notebook, not this laptop (§5.1). Plus **your own labelled clips** (§6) | `data/manifests/aigen.csv` with `generator`, `source`, `split_group`; the leakage validator passes |
| D2 | `src/trubeat/provenance.py`: C2PA manifest read (c2pa-python, offline), container metadata and encoder tags, optional visible-watermark template match | unit tests on signed and unsigned fixtures; zero network calls |
| D3 | `src/trubeat/aigen.py` + `trubeat aigen <video>`: sample 16–32 frames, frozen DINOv2-S (or CLIP ViT-B) embeddings, ReStraV-style curvature and step statistics plus D3-style second-order temporal differences → logistic regression. Embeddings cached by content hash, same as the trace cache | runs on CPU in < 15 s for a 10 s clip; property test: a static image animated with affine motion is scored differently from noise |
| D4 | Training: group-aware, threshold on validation only, config in `configs/p5_aigen.yaml` | model bundle with `feature_columns`, `threshold`, config hash |
| D5 | **Evaluation the honest way**: leave-one-generator-out AUC with bootstrap CIs, plus robustness to re-encoding (CRF 23/35), downscaling to 480p, and a screen recording through the extension | `reports/aigen_detector.md` with the full ROC; no single accuracy quoted |
| D6 | Dependency: `torch` CPU wheels + `transformers` or `timm` (numpy < 2 pin preserved) | human gate (§6) |

### Phase E — Fusion, app and extension (3–5 days)

| task | what | done when |
|---|---|---|
| E1 | Fusion: provenance overrides; otherwise logistic stacking of P(AI) from D and P(manipulated) from repaired rPPG, weighted by B4 | stacked model beats each layer alone on validation, with a CI |
| E2 | App: evidence panel per layer; the caveat rewritten to the measured envelope | e2e test on fixtures |
| E3 | F14: the extension reads the `<video>` element's duration and captures min(duration, 20 s), and detects loops and end frames | JS harness tests |

### Phase F — Evaluate on what you actually care about (ongoing)

| task | what | done when |
|---|---|---|
| F-1 | **Your labelled set**: `data/raw/user_eval/` + `data/manifests/user_eval.csv` (path, label, generator if known, source). Local only, never uploaded or committed | every phase report includes it as a separate row |
| F-2 | Cross-dataset: Celeb-DF v2 (form already prepared in `reports/HUMAN_GATES.md`) | per-dataset AUC with CI |
| F-3 | Track the Inconclusive rate, false-positive rate on real videos, and recall per generator, phase over phase | table in each report |

### 5.1 Where the work runs (decided 2026-11-01)

Machine: Ryzen 5 4600H (12 threads), 7.1 GiB usable RAM, **no swap**, no
CUDA GPU, 79 GB free disk.

- **Local (everything except the two items below):** Phases A, B, C, E, F,
  and D2–D5. The design is CPU-first on purpose: the shipped app and
  extension must run on a laptop like this one with zero network calls, so
  a detector that only runs in the cloud would break the privacy promise.
- **Free cloud GPU notebook (Kaggle or Colab), public data only:**
  generating clips with open-weight video generators (D1), and optionally
  bulk embedding extraction if a public dataset subset grows past what an
  overnight local run can handle.
- **Never in the cloud:** your own videos (`data/raw/user_eval/`), and
  FF++/Celeb-DF unless their licences permit it *and* the human approves
  in writing (`AGENTS.md` §3, §7). Paid cloud is not needed; any spend needs
  approval.
- **Machine prerequisite:** add swap (zram or a swapfile) before Phase B.
  With no swap, any memory spike is an instant OOM kill. That is what
  killed the extension test during this audit.
- **Working rules for 8 GB:** one heavy job at a time; decode only sampled
  frames, or crop around the face box, never whole native-resolution
  clips; cache embeddings and traces to disk; download dataset subsets
  sized to the free disk.

**Order and dependencies:** A → B (B1–B3 in parallel) → C. D can start
straight after A (independent code path) but needs D1/D6 gates. E needs C and
D4. F-1 starts **now**.

---

## 6. Human gates and decisions

| # | what you do | unblocks |
|---|---|---|
| 1 | Approve this plan in writing (amends `docs/07` and `docs/10`) | everything |
| 2 | **Collect 30+ videos whose label you know** (AI and real, mixed sources, including the ones that came back Inconclusive) into `data/raw/user_eval/`, with a CSV of labels | F-1, the only test of "does it work on my videos" |
| 3 | `uv pip install torch --index-url https://download.pytorch.org/whl/cpu` plus `timm` (or `transformers`) | D3–D5 |
| 4 | Download the GenVideo / GenVidBench subsets after reading their licences (dataset terms may require a form) | D1 |
| 5 | Optional: generate clips with Sora / Veo / Kling on your own accounts (I never create accounts) | D1 coverage of current generators |
| 5b | Enable swap (zram or a swapfile; needs sudo) | safe Phase B+ runs on 8 GB |
| 5c | Optional: a free Kaggle or Colab account for GPU notebooks (public data only) | open-weight generator clips for D1 |
| 6 | Celeb-DF v2 access form | F-2 |

---

## 7. What not to expect

- **No single number like "96%".** Generated-video detectors lose accuracy on
  generators they never saw, and new generators ship every few months. The
  number that matters is leave-one-generator-out recall at a fixed
  false-positive rate on real video, re-measured as generators change.
- **The pulse layer will stay narrow.** On typical web video (small faces,
  compression), the pulse is mostly not recoverable for real faces either
  (R-01). Its job is close-up face video, not the general case.
- **Provenance is a gift, not a guarantee.** It's present on some
  first-generation downloads and stripped by nearly every re-share.
- **"Inconclusive" will not vanish.** The aim is that it means "no layer has
  evidence", happens on a measured minority of clips, and says why.

---

## 8. Sources

- Zheng et al., *D3: Training-Free AI-Generated Video Detection Using
  Second-Order Features*, ICCV 2025 — <https://arxiv.org/abs/2508.00701>
- Internò et al., *AI-Generated Video Detection via Perceptual Straightening*
  (ReStraV), NeurIPS 2025 — <https://arxiv.org/abs/2507.00583>
- Chen et al., *DeMamba: AI-Generated Video Detection on Million-Scale GenVideo
  Benchmark* — <https://arxiv.org/abs/2405.19707>
- *GenVidBench: A 6-Million Benchmark for AI-Generated Video Detection* —
  <https://arxiv.org/abs/2501.11340>
- Kundu et al., *Towards a Universal Synthetic Video Detector* (UNITE),
  CVPR 2025 —
  <https://openaccess.thecvf.com/content/CVPR2025/html/Kundu_Towards_a_Universal_Synthetic_Video_Detector_From_Face_or_Background_CVPR_2025_paper.html>
- OpenAI, *Advancing content provenance* —
  <https://openai.com/index/advancing-content-provenance>
- Generator clip lengths (vendor-reported, change often) —
  <https://www.aimagicx.com/blog/veo-3-vs-kling-3-vs-sora-2-april-2026-comparison>
- de Haan & Jeanne, *Robust pulse rate from chrominance-based rPPG*, IEEE TBME
  2013 (the SNR definition in B2)
- Internal: `reports/g2_detector.md` (G2 failed: AUC 0.543, FPR 0.237),
  `reports/raw_probe_result.md` (R-01), `scripts/audit_verdicts.py`

---

## 9. Round 2 (overnight, 2026-11-01): built, measured, audited

Instruction (human, 2026-11-01): *"once you have implemented current plan
conduct audit if needed and construct a plan around that audit; make changes
if it improves the tool"*. This section is that audit and that plan. Every
number below names its dataset and split; reports are under `reports/`
(gitignored, regenerated by the scripts named).

### 9.1 What changed in the tool

| layer | before | now |
|---|---|---|
| verdict | one pulse-consistency detector; Inconclusive whenever the (broken) SNR gate, the 10 s minimum or the face tracker said so | fused, in priority order: Content Credentials > AI-video detector > strict face-swap check > "No sign of AI generation found" > Inconclusive only when no layer could run (`app.verdict_for`) |
| AI-generated video | nothing | `src/trubeat/aigen.py` + shipped `src/trubeat/models/aigen.joblib`: frozen DINOv2-S, mean embedding of one 2 s segment (16 frames at 8 fps) after re-encoding to 360p CRF 28, logistic regression |
| provenance | nothing | `src/trubeat/provenance.py`: C2PA read offline (no remote manifests, no OCSP) |
| pulse | gate on a drift-dominated SNR | standard pulse SNR in dB, evidence only; on FF++ c23 real clips it is below 0 dB in 100% of cases, so it decides nothing |
| face-swap | G2 at its default threshold (FPR 22–24%) | G2 only at a strict threshold (3/135 FF++ test reals flagged); a weak, labelled check |
| extension | fixed 20 s capture | capture length from the page's `<video>` (duration, position, loop, paused) |

Deviation from E1: no stacked fusion. The pulse layer carries no signal on
web video (§5 "Phase B result so far"), so a stacker would learn to ignore
it; a priority rule with each layer's evidence shown is simpler and honest.

### 9.2 Measured (AI-video detector)

Pool = AIGVDBench (2,521 AI clips from 26 generators in 17 families; 1,200
content-paired reals) + 157 FF++ c23 original train/val reals. External =
GenBuster-Bench (other curators) + 135 FF++ c23 original test reals: never
used for training, selection or the threshold. Source: `reports/aigen_detector.md`.

| measure | value (95% CI) |
|---|---|
| leave-one-family-out AUC on the pool, macro over 17 families | 0.809 (worst SEINE 0.54, LTX 0.55, Pyramid-Flow 0.64; best Jimeng 0.97, Open-Sora 0.96, Pika 0.95) |
| false alarms, GenBuster real (OpenVid-style footage) | 57/1000 = 5.7% [4.4, 7.3] |
| false alarms, FF++ c23 test real (talking heads) | 0/135 [0, 2.8] |
| GenBuster ood: Sora, Kling, Gen-3, Luma, Pika, Vidu, Jimeng, Wanx | AUC 0.954 [0.946, 0.963]; caught 78% [76, 81] |
| GenBuster wild: Veo 3.1, Sora 2, Kling 2.5/2.6, Seedance, Hailuo, PixVerse, Wan 2.5/2.6 | AUC 0.859 [0.828, 0.889]; caught 45% [38, 53] |
| GenBuster id: CogVideoX, EasyAnimate, HunyuanVideo, LTX (open-source) | AUC 0.872 [0.856, 0.887]; caught 50% [47, 53] |
| latency, live app on this laptop | 2.5 s for a 5 s clip (AI layer; pulse skipped under 10 s); 13 s for an 11–16 s clip (all layers) |

The threshold chosen on pool reals at 5% FPR held externally (5.7%), so the
operating point transfers to OpenVid-style footage. It is untested on phone
video, screen recordings and animation (no such reals in any set).

### 9.3 Independent code review (fixed in `c71cd37`)

An independent review of the night's code found nine defects, all fixed with
tests: `cap_rows` dropped columns under pandas 3 (killed the first training
run); the pool cap dropped every FF++ real; the last segment repeated a frame
(sampler now grid-indexed; training frames verified unchanged 72/72); train
and serve used different segment counts (bundle now pins it); the shown
"probability" could contradict the verdict (replaced by the real-clip
percentile); a pulse-layer exception discarded the other layers; the
extension ignored loop/paused state; "face tracked 0%" was shown when never
measured; the evidence table showed the default face-swap threshold rather
than the applied one.

### 9.4 Round-2 scripted audit (`scripts/audit_aigen.py` → `reports/aigen_audit.md`)

| # | finding | evidence | action |
|---|---|---|---|
| R2-1 | **The pool's AI and real clips differ in format.** Metadata alone (fps, size, aspect, duration, bitrate) separates them with leave-one-family-out AUC 0.993: AI median 768×480 at 0.93 Mb/s, real median 1920×1080 at 3.7 Mb/s | `reports/aigen_audit.md` §1 | The detector never sees metadata (re-encode to 360p CRF 28, centre crop, time-based sampling), but a residual "soft = AI" cue could survive downscaling. Tested directly by resolution counterfactuals (R2-6) |
| R2-2 | Near-duplicate content across pool and external (mean-embedding cosine ≥ 0.97): 54/1000 GenBuster ood AI clips, 8/1000 GenBuster reals | audit §2 | Removing them changes nothing (recall identical, FPR 5.7% → 5.5%): no leakage effect |
| R2-3 | The AI-video detector does not see face edits: it flags 0% of FF++ c23 test fakes in all four families (Deepfakes 0/29, Face2Face 0/31, FaceSwap 0/27, NeuralTextures 0/28) and 0/40 reals | audit §4 | Expected (whole-frame features; edits are local). The caveat keeps the face-swap limits; a real face-swap detector is plan item P5 |
| R2-4 | Only the first 2 s of a clip is scored (the bundle is evaluated on one segment); every clip carried a misleading `W_SHORT_CLIP` | live app run | Fixed (`7197848`): `W_PARTIAL_COVERAGE` states what was scored. Multi-segment scoring is P2 |
| R2-5 | 579 AIGVDBench clips are under 2 s and cannot be scored (AccVideo, AnimateDiff, 78% of HunyuanVideo, Open-Sora T2V, VideoCrafter) | training log | Out of scope by design: such clips come back Inconclusive with `E_SHORT_VIDEO` |
| R2-6 | The first robustness rows used only 20 reals per source (an FPR of 0/20 bounds nothing below 16%) and skipped the app's own re-encode, so they did not measure the app path | `scripts/train_aigen.py` | Fixed: `--robustness-only` re-scores 200 reals per source through transform + the app's normalisation, plus resolution counterfactuals (480p downscale, 1080p upscale). Result: §9.5 |
| R2-7 | `scripts/eval_user_set.py` would have crashed on the first clip (read a field the review fix removed) | code read | Fixed with a test that uses a real `PulseAnalysis` (`088e92f`) |
| R2-8 | Richer descriptors do not help: adding per-dimension frame std or mean frame-to-frame change lowers leave-one-family-out AUC (0.803–0.806 at best vs 0.809–0.813 for the mean alone) | `scripts/explore_aigen_features.py` → `reports/aigen_feature_search.md` | Kept the shipped model (rule fixed in advance: adopt only at +0.01) |
| R2-9 | The trajectory-geometry features (ReStraV/D3-style step, curvature, acceleration) are near chance on unseen families (macro 0.53–0.57) | `reports/aigen_detector.md` | The embedding carries the signal; geometry stays in the code as a candidate set, unselected |

Acceptance status against §5: D3 latency met (< 15 s for a 10 s clip); the
D3 property test (affine-animated still vs noise) is not written. D5: the
re-encode/downscale rows exist; a screen recording through the real
extension needs a browser session (human). E1 deviates as noted in §9.1.

### 9.5 Robustness through the app path

`python scripts/train_aigen.py --robustness-only` re-encodes a fixed subset
of external clips (up to 40 AI clips per generator; 200 GenBuster reals and
the 135 FF++ test reals), then sends each through the app's own 360p CRF 28
normalisation, with the threshold fixed. Shipped model (v2):
`reports/aigen_robustness_v2.md`; v1, on a smaller AI subset (20 per
generator): `reports/aigen_robustness.md`.

| condition (v2, fixed threshold +1.11) | GenBuster real FPR (n=200) | FF++ real (n=135) | recall id / ood / wild | AUC wild |
|---|---|---|---|---|
| app path as trained (360p, CRF 28) | 5.5% | 0 | 44% / 83% / 55% | 0.878 |
| real clips downscaled to 480p first | 4.5% | 0 | 42% / 82% / 51% | 0.877 |
| AI clips upscaled to 1080p first | 5.0% | 0 | 44% / 83% / 53% | 0.878 |
| tab-capture-like (VP9 WebM, 720p, 2.5 Mb/s) | 5.5% | 0 | 42% / 82% / 53% | 0.878 |
| harsh re-encode (240p, CRF 35) | 3.5% | 0 | 29% / 62% / 35% | 0.837 |
| as-is (no ffmpeg on the machine) | 5.5% | 0 | 53% / 86% / 63% | 0.899 |

What this says:

- **R2-1, the resolution shortcut, is not supported.** Shrinking real clips
  to 480p does not raise false alarms (5.5% → 4.5%), and enlarging AI clips
  to 1080p does not hide them (wild 55% → 53%, ood 83% → 83%). v1 behaved the
  same (FPR 4.5% → 4.0%; wild 45% → 45%). The 360p re-encode appears to erase
  the format gap the metadata audit found. A 200-clip FPR has a wide interval
  (5.5% is [3.1, 9.6]), so a change of 1–2 points is within noise.
- **The extension's capture path is safe** (simulated): VP9 at 720p and
  2.5 Mb/s moves recall by at most 3 points (id 44% → 42%) and false alarms not at all. A
  real browser capture is still P7.
- **Heavy compression is the weak point, and it fails towards "real".** At
  240p CRF 35 (a re-shared, re-compressed clip), recall falls by about a
  third on every tier (wild 55% → 35%), while false alarms fall too
  (5.5% → 3.5%). Badly compressed AI clips will be missed more often, not
  flagged wrongly more often.
- Without ffmpeg the app scores the raw file: recall is higher and false
  alarms are unchanged on this subset, but the scores drift up (median real
  −1.85 → −1.60), so the measured operating point does not strictly apply.
  The app flags that case with `W_NO_FFMPEG`.

### 9.6 Round-2 plan

| # | what | why | done when |
|---|---|---|---|
| P1 | **Your labelled set** (human gate 2): put clips in `data/raw/user_eval/ai/` and `data/raw/user_eval/real/`, run `python scripts/eval_user_set.py` | the only test of "does it work on my videos"; every number above is on curated sets | `reports/user_eval.md` with inconclusive rate, catches and false alarms, each with a CI |
| P2 | **Tested, not adopted (§9.8).** Multi-segment scoring: average up to 3 segment scores, threshold recalibrated on out-of-fold pool reals scored the same way | a long clip is judged on its first 2 s today | external recall at ≤ 6% GenBuster FPR is no worse on any tier; adopt only then |
| P3 | Newest generators (45% caught; 55% after the §9.7 retrain, shipped): training clips from Veo 3, Sora 2, Kling 2.x, Seedance, Hailuo. GenBuster wild (150 clips) is the only held-out set for them, so it cannot also be training data | the biggest recall gap, and these are what people share now | a second public set of current-generator clips found (licence checked) or generated on your accounts (gate 5) |
| P4 | More kinds of real video in the pool: phone footage, social re-uploads, low-light, screen recordings (licence-clean sources) | false alarms are measured only on OpenVid-style and FF++ footage; R2-1 shows how narrow the pool's reals are | FPR reported per real-video kind |
| P5 | A real face-swap detector (frame-level CNN on FF++ c23 faces, trained on a free GPU notebook, gate 5c), replacing G2 at its strict threshold | today's check catches ~24% of Deepfakes swaps and is blind to reenactment | FF++ test AUC per family with CIs, and a Celeb-DF v2 cross-check (gate 6) |
| P6 | A larger frozen backbone (DINOv2-B or CLIP ViT-L/14) for the AI layer, embeddings computed once on a notebook GPU | linear probes on the small backbone plateau at 0.81 leave-one-family-out | macro LOGO AUC +0.03 or better at < 10 s CPU inference for a 10 s clip |
| P7 | Record a real tab capture through the extension on a few known clips (human: needs a browser) | the VP9 row is a simulation | the extension's verdicts on those clips logged in `reports/user_eval.md` |

### 9.7 P3, first step: Rapidata clips of current generators (rule fixed before any result)

Source: Rapidata's text-to-video human-preference sets on Hugging Face
(Apache-2.0, ungated). The MP4s cover Veo 2 / 3 / 3.1, Seedance 1 Pro,
Moonvalley Marey, Luma Ray 2, Runway Gen-3 Alpha, Sora (v1), Pika 1 / 2.2,
HunyuanVideo, Wan 2.1 and Mochi 1 on one ~100-prompt list. The Sora 2,
Kling 2.1, Ray 2 and Runway sets ship only GIF previews and are not used: a
GIF palette would teach "GIF = AI". The clips are added to the pool only
(grouped by prompt); GenBuster and the FF++ test reals stay external.

Consequence for the numbers: GenBuster ood/wild clips from Sora, Luma,
Runway, Pika, Veo 3.1 and Seedance then measure *a seen family from another
curator*, not an unseen generator. Unseen-generator performance stays the
pool's leave-one-family-out AUC.

Adopt the retrained model ("v2") over the shipped one only if all hold:

1. false alarms at v2's own out-of-fold threshold: GenBuster reals ≤ 7.3%
   (the shipped model's upper CI) and FF++ c23 test reals ≤ 2/135;
2. GenBuster wild recall ≥ shipped + 5 points, and no GenBuster tier's
   recall drops by more than 3 points;
3. pool macro leave-one-family-out AUC over the 17 original families ≥
   shipped − 0.01.

Otherwise the shipped model stays, and the result is recorded here.

**Result (2026-11-01 14:20, `scripts/compare_aigen_models.py` →
`reports/aigen_v2_decision.md`): all rules pass; v2 adopted and shipped
(`0c15937`).** 1,492 Rapidata clips (14 generator labels) joined the pool;
v2's threshold from out-of-fold pool reals is +1.1149.

| rule | v2 | shipped (v1) | pass |
|---|---|---|---|
| GenBuster-real FPR ≤ 7.3% | 54/1000 = 5.4% [4.2, 7.0] | 5.7% | yes |
| FF++ c23 test reals flagged ≤ 2 | 0/135 | 0/135 | yes |
| wild recall ≥ v1 + 5 points | 54.7% [46.7, 62.4] | 45.3% | yes |
| no tier drops > 3 points | id 52.2%, ood 83.1% | id 49.9%, ood 78.3% | yes |
| LOGO macro, original 17 families ≥ v1 − 0.01 | 0.819 | 0.809 | yes |

Checks on the result, run after the decision and not used for it:

- **Is the wild gain only Veo 3.1 being in training?** No. Without the 10
  GenBuster Veo 3.1 clips, wild recall is 44.3% → 52.9% (n = 140). On the
  two wild families with no version in v2's pool (Hailuo 2.3, PixVerse V5;
  n = 30) it is 46.7% → 56.7% (too few clips for a firm claim).
- **Near-duplicates between Rapidata and GenBuster?** None: 0/150 wild clips
  have a pool clip at cosine ≥ 0.97 (`reports/aigen_audit_v2.md`). The other
  counts are unchanged from v1 (54 ood AI clips, 8 reals), so the Rapidata
  clips added no near-duplicates.
- **Unseen-family AUC** (pool leave-one-family-out, all 21 families of v2):
  macro 0.841; the open-source tail stays weak (LTX 0.54, SEINE 0.55,
  Pyramid-Flow 0.62, SVD 0.67, EasyAnimate 0.68). New families held out
  whole: Veo 0.94, Seedance 0.93, Marey 0.92, Mochi 0.96.
- **Live app** (`trubeat serve`, driven through its API): the GenBuster
  Veo 3.1 clip v1 missed (+1.03) is now flagged (+1.20 ≥ +1.11); a GenBuster
  real (−3.27) and FF++ 000 (−2.98) stay real; the iStock clip is still
  decided by the face-swap check (AI score −1.97, 28th percentile of reals).
- Labels in the per-generator table of the decision report: Kling 2.6,
  Sora 2, Seedance 1.5/2.0 and Wan 2.5/2.6 have an *earlier version* of
  their family in v2's pool; the report first called them unseen (fixed,
  `7bd839a`).

The report's generator count ("38 generators") counts each mode and
version label (`wan`, `wan2.1`, `wan2.1_t2v`); the caveat says 21
generator families instead.

### 9.8 P2 result: whole-clip scoring tested, not adopted

`scripts/eval_multiseg.py` → `reports/aigen_multiseg.md` (2026-11-06).
Each clip's 2 s segments, up to 3 spread evenly over the clip, were embedded
(segment 0 is the one the shipped model was trained on; a cosine check
confirmed it re-embeds identically). Three descriptors, all with the
shipped recipe:
`first` (the shipped model: segment 0 only), `avg` (trained on segment 0,
scored on the mean of all segments = the average of segment scores) and
`avg-train` (trained and scored on the mean). Rule fixed in the script
before the run: pick `avg` or `avg-train` by pool leave-one-family-out AUC;
adopt it only if that AUC ≥ `first`'s, GenBuster-real FPR ≤ 6% (§9.6 P2),
every GenBuster tier's recall ≥ `first`'s, and FF++ test reals flagged ≤ 2.

| | first (shipped) | avg | avg-train |
|---|---|---|---|
| pool LOGO macro AUC (21 families) | 0.841 | **0.845** | 0.845 |
| threshold (OOF pool reals, 5%) | +1.115 | +1.037 | +0.968 |
| GenBuster-real FPR | 5.4% | 5.0% | 6.0% |
| recall id / ood / wild | 52.2 / 83.1 / **54.7%** | 53.4 / 84.0 / 54.0% | 55.6 / 86.4 / 52.7% |
| AUC id / ood / wild | 0.880 / 0.961 / 0.880 | 0.886 / 0.964 / 0.886 | 0.883 / 0.963 / 0.875 |
| FF++ test reals flagged | 0/135 | 0/135 | 0/135 |

`avg` won on the pool and passed every gate but one: wild recall 81/150
against 82/150. The rule required no tier to get worse, so **the shipped
model keeps scoring the first 2 s** (`W_PARTIAL_COVERAGE` says so). The gap
is one clip and well inside the interval, so this is not evidence that
whole-clip scoring hurts; it is evidence that it does not clearly help.
Changing the rule after seeing the numbers is what the rule exists to
prevent. A retest needs a rule with an explicit tolerance and a fresh
held-out set.

Notes: segments per clip, all clips: 1 segment 1,155, 2 segments 5,856,
3 segments 1,644. Real clips run longer (36% of reals have 3 segments vs
12% of AI clips), so segment count tracks the label; `avg` describes every
clip as a mean, so a 3-segment clip is a less noisy mean, not a different
kind of input. 55 of 8,655 clips could not be decoded at 3 segments and
were dropped from all three variants alike.

## 10. Progress log

Commits on `main`, oldest first, from the approval of
this plan: A2 `12c54a6`, A3 `3d9add0`, A4 `cd6af9c`, A5 `ce6e71e`, A6
`f0c231a`, B2 `30281ce`, D1/D3 `17d90e9`, D2/D4/D5 `973b7a6`, D3/D6
`1d3884d`, C/E1 `3c3f612`, E2/E3 `03a354e`, F-1 tooling `3277afd`, LOGO by
family `aed3b9b`, review fixes `c71cd37`, round-2 audit script `36fa416`,
shipped detector `1c635c1`, coverage warnings `7197848`, robustness via the
app path `59544af`, descriptor search `376d58f`, Rapidata download and the
§9.7 check `18b11bd`, Rapidata tests `32e9ec5`, docs/04 warning table
`0f443d1`, decision-report family labels `7bd839a`, v2 shipped `0c15937`, robustness
and v2 docs `85af3d4`, website workflow note `ec05412`, missing-extra error
`54cd394`, P2 result (the commit that adds `scripts/eval_multiseg.py`).

P2 (multi-segment scoring) ran end to end on 2026-11-06 and was not
adopted (§9.8); `scripts/eval_multiseg.py` is committed with its result.
