# 13 — Round 4: a face-swap detector that survives processing, finds small faces, and a repaired capture path; plus the audit fixes (2026-11-07)

**Status: approved to execute (human, 2026-11-07: "approve the plan";
"dont start implementing yet … only refine plan rightnow": implementation
starts in a new session).** Refined the same day with what arrived: the
Kaggle API token, the DF40 and Celeb-DF v2 download links, and Pexels'
unavailability. Written for hand-off to the executing agent. Nothing in §6 (decision rules) may change after a result
exists; a revision is recorded the way docs/12 §6 records its title-card
revision (dated, before the re-check).

**Scope.** (1) Face-swap detector v2: processing-invariant training with
real faces from more processing histories, modern swap methods if their
data arrives, and the detector adopted only by rules fixed here. (2) A
face finder that finds small faces (measured today: the current one misses
any face under about 12% of the frame width). (3) The browser extension's
capture: full resolution, cropped to the video element, and six failure
modes fixed. (4) Every flaw found by a four-part audit of the codebase on
2026-11-07 (Appendix A: 88 findings, a few the same defect seen by two audits), fixed in the phase where it matters.

---

## 0. Hand-off brief for the executing agent

Read, in this order: `AGENTS.md`; this document in full; `docs/12` §6 and
§9 (the rules and result this round builds on); `docs/04` §4 and §8 (the
frozen interfaces and the warning table); `reports/faceswap_report.md` and
`reports/faceswap_gate4.md`; `docs/11` §9.2, §9.7 (how the AI-video
detector's numbers were made honest). Then Appendix A here.

**Where things stand** (`main` at `5714521`, 2026-11-07):

- The app runs four layers (Content Credentials, AI-video detector,
  face-swap detector, pulse) and fuses them by priority (`app.verdict_for`).
- The round-3 face-swap detector (`src/trubeat/models/faceswap.joblib`:
  CLIP ViT-B/16 CLS + logistic regression, threshold +1.577) ships as
  **evidence only**: it passed the in-dataset and cross-dataset gates and
  flagged 115/300 = 38.3% of real faces from other sources (docs/12 §9).
- Extension v1.2.0 shows the detector's row; the website is current.
- Datasets on the laptop: `data/raw/ffpp` (603 FF++ clips: 140 test
  reals + 119 test fakes among them), `data/raw/aigen` (GenBuster-Bench,
  AIGVDBench, Rapidata, 21 GB), UBFC. Colab result under
  `runs/faceswap/colab_20261107/` (gitignored).

**What is fixed by this document:** the datasets' roles (§4), the
selection and adoption rules (§6), the outcome mapping (§6), the face
finder's adoption test (§6.1), the extension's crop test (§6.4), the gate
definitions and denominators (§6.2). **What is free:** implementation
details, file layout, how the cloud runs are sharded, the order within a
phase, the wording of texts (within the honesty rules).

**Standing rules** (AGENTS.md plus the user's rules, all still in force):
never upload or share faces/videos to any service beyond the public
datasets the cloud notebook downloads itself; the user's own videos
(RepresentUs, `data/raw/user_eval/`, `/tmp/gradio/*`) never leave the
laptop and are never used to tune; no AI attribution in commits; conventional
small commits, one per finished item; push `main` directly after the round's
review; the website gets one Updates entry in the same push cycle (website
repo README, "When the tool changes"); every number shown names its dataset
and split; never fake a gate; prefer Kaggle over Colab for cloud runs (Colab
flagged the notebook's deepfake keywords, and its free session ended ~7 h in).

**Human gates in this round** (§9). Done on 2026-11-07: the plan approved;
the Kaggle API token saved at `~/.kaggle/access_token` (local, never
committed); the DF40 and Celeb-DF v2 links received after their forms and
recorded in `runs/faceswap/round4_links.md` (gitignored). Pexels is
unavailable (key issuance paused). Left: reviewing the result report; the
extension release command if the agent's permission check blocks
`gh release create`; Kaggle phone verification only if the first kernel push
with internet is refused.

---

## 1. Why

### 1.1 Round 3 left two problems and found a third

| what was measured (docs/12 §9, reports/) | consequence |
|---|---|
| **Processing looks like a swap.** The detector flagged 1/140 FF++ test reals as downloaded, 45/140 = 32.1% [25.0, 40.3] after a WhatsApp-like re-encode (640 px wide, CRF 30), and 106/158 = 67% of GenBuster-Bench reals (5 s, 1024×1024 HEVC stock clips) vs 9/142 = 6.3% of AIGVDBench reals (1080p H.264 YouTube cuts). | A forwarded clip, the case the round started from, is where the flag means least. Shipped as evidence only. |
| **Faces are not found unless they are large.** DeepFakeDetection: 161/300 fakes and 83/150 reals had a face in fewer than 4 of 16 frames. Gate 4: 300 of 706 clips decided. The user's extension capture of a YouTube tab: "a face was found in 0 of 16 sampled frames" for a presenter filling a quarter of the player. | Wide shots, embedded players and phone video are not judged at all. |
| **The extension records the whole tab at Chrome's default size.** The user's 1920×1080 screen was captured at 960×540; the presenter's face was ~30 px; the backend then downscales to 480 px tall for the pulse layer. | Neither face check can run on the extension's main use case. |

### 1.2 Measured today: the face finder's limit, and a fix that works

`faceswap.FaceFinder` and `face.track_landmarks` use the MediaPipe Face
Landmarker on the whole frame. Its built-in detector is BlazeFace
short-range, which sees the frame at 128×128. Measured on 2026-11-07 with an
FF++ test frame pasted at known sizes into a plain canvas
(`FaceFinder.largest_face`, default options):

| frame | face width (share of frame width) | Landmarker | YuNet (OpenCV `FaceDetectorYN`, `face_detection_yunet_2023mar.onnx`) |
|---|---|---|---|
| 1920×1080 | 20 px (1%) … 120 px (6%) | missed at every size | found at every size, score 0.88–0.94; 280–460 ms per frame on this CPU |
| 1920×1080 | 200 px (10%) | missed | — |
| 1920×1080 | 240 px (12%) | found | — |
| 1280×720 | 60 px (5%), 100 px (8%), 140 px (11%) | missed | found, score 0.94–0.95 |
| 1280×720 | 180 px (14%) | found | found |
| 1280×720 copy of the 1080p frame with a 40 px face | — | — | found (20 px after the resize), **82 ms** |
| 256 px crop (2× margin) around a 40 px or a 100 px YuNet box | — | **found** (landmarks on the crop) | — |

So the Landmarker needs a face of roughly 12–14% of the frame width, and a
two-pass finder (YuNet on a ≤1280-wide copy, then the Landmarker on a 2×
crop) finds and landmarks faces down to ~20 px. YuNet also returned a few
low-score (0.60–0.83) 12–47 px detections on the native FF++ frame, so a
score floor, the existing `min_face_px`, and Landmarker confirmation on the
crop are all needed. The model is 233 KB, MIT-licensed (opencv_zoo), and
`cv2.FaceDetectorYN` exists in the pinned OpenCV 4.9.0.80.

### 1.3 The audit

Four independent read-only audits (core Python; training scripts and
notebook; extension and web UI; tests, CI, packaging, docs and website) plus
the author's own read produced 88 findings (Appendix A). The ones that
change what the tool does or claims:

- **Round-3 false-alarm rates for gates 2–3 count clips the detector never
  judged as "not flagged"** (`train_faceswap.held_out_tables`), while gate 4
  counts only judged clips. On judged clips, gate 3 is 3/67 = 4.5%
  [1.5, 12.4], not 3/150 = 2.0%; FF++ test reals are 1/134, not 1/140. The
  decision was unchanged (gate 4 failed anyway), but docs/12 §9, the README
  and the website quote the easier number. Fixed and corrected in P1.
- **The extension's recordings bypass the pulse layer's memory guard and
  downscale** (`ingest.open_video`, the no-frame-count path): a 60 s capture
  can need ~10 GB on a 7 GB laptop. And 1080p clips longer than ~68 s fail
  the pulse layer as "unreadable" because the 2.5 GB budget is checked on
  the whole clip before the 120 s cut.
- **`trubeat serve` binds 0.0.0.0** and keeps every upload in `/tmp/gradio`
  forever; the extension and README say 127.0.0.1 and "nothing is stored".
- **Extension failure modes:** a backend exception leaves "Analysing…"
  forever; a capture error locks the extension ("Already running") until the
  worker idles; "track ended" is reported to tab `null`; the DRM/black-frame
  guard is dead code (`drawImage(track)` throws); the content-script
  fallback injection never runs; permissions are broader than needed.
- **A real video can be made "AI-generated: declared by its Content
  Credentials"** by attaching an unsigned manifest: the C2PA layer decides
  even when the signature state is not Valid/Trusted.
- **CI never runs the embedders, the C2PA reader or the shipped
  `aigen.joblib`**; `pip install -e .` installs core deps only, so several
  tests pass vacuously.
- The AI-video detector's training never ran the identity-leakage validator
  (AGENTS.md §6); the LOMO selection margin in round 3 (0.838 vs 0.833) is
  inside fold noise; `config_hash` omits parameters that change the result;
  the gate-4 clip list cannot be rebuilt from anything in the repo.

---

## 2. The machine, and where each step runs

| resource | this laptop | consequence |
|---|---|---|
| CPU / RAM | Ryzen 5 4600H (12 threads); 7.1 GB RAM + 11 GB swap | one heavy job at a time; decode only the frames needed |
| GPU | GTX 1650 4 GB; PyTorch is the CPU build | fallback for embedding only (CUDA torch is a ~3 GB install) |
| disk | 45 GB free | the cloud still does the data work; the laptop keeps its 21 GB of gate sets |
| cloud | **Kaggle Notebooks** (user has Google AI Pro, but Colab flagged the notebook and cut the session) | 30 GPU-hours/week (floating; resets Saturday UTC), 12 h per session (CPU and GPU), 4 vCPU, ~20 GB persistent output under `/kaggle/working`, internet only after phone verification. Verify these at the first cell (`nvidia-smi`, `df -h`, `nproc`); they change. GPU availability can be degraded; CPU-only kernels have no weekly quota. |

**How the cloud is driven.** Not by the human clicking "Run all": the
agent drives Kaggle from the laptop with the official CLI
(`uv tool install kaggle --with six`; the user's API token, the new `KGAT_`
format, is at `~/.kaggle/access_token`, which the CLI reads; the Kaggle
username comes from `kaggle kernels list --mine` or the account page). A notebook is a
folder with `kernel-metadata.json` (`id` = `<user>/<slug>`, `code_file`,
`kernel_type: notebook`, `language: python`, `is_private: true`,
`enable_gpu`, `enable_internet`, `machine_shape: NvidiaTeslaT4`,
`dataset_sources`, `kernel_sources` = `["<user>/<previous-slug>"]` to mount
a previous run's output under `/kaggle/input/<previous-slug>/`). Commands:
`kaggle kernels push -p <dir> [--timeout <s>]`, `kaggle kernels status
<id>`, `kaggle kernels output <id> -p <dir> [--file-pattern <regex>]`. Each
stage is one kernel version whose outputs (feature tars, crops, reports)
land in `/kaggle/working` and are chained into the next stage. The human
never downloads anything by hand; the agent pulls the small results
(bundles, reports, manifests; never videos or crops).

**Split of work**

| step | where | who |
|---|---|---|
| code, tests, face finder, extension, app, docs | laptop | agent |
| frozen gate-4 list, laptop re-checks, RepresentUs and `user_eval` scoring, ship | laptop | agent |
| download FF++ (c23 + c40), DeepFakeDetection, DF40 and Celeb-DF v2 if granted; faces, variants, crops, embeddings, selection, held-out tables; stage 2 if triggered | Kaggle, API-driven | agent (token: human) |
| result review; the extension release command if blocked; phone verification only if asked (token, forms and approval: done 2026-11-07) | — | human |

**Fallback** if Kaggle GPUs are unavailable for days: CPU-only Kaggle
kernels for faces and variants (no quota), and either a Kaggle GPU kernel
later or the laptop's GTX 1650 for the embeddings only (crops are small:
the agent would pull them as a Kaggle Dataset, ~10 GB, with the human's OK).

**Privacy.** Only public research datasets are processed in the cloud, downloaded
there from their own servers. Nothing from the laptop is uploaded except
code. Inference stays local.

---

## 3. Design

### 3.1 Face finding v2 (`src/trubeat/facefind.py`, new; used by `faceswap.py` and `face.py`)

1. **Detect** with OpenCV YuNet (`cv2.FaceDetectorYN`, model
   `face_detection_yunet_2023mar.onnx`, shipped in `src/trubeat/models/`
   with its MIT licence noted in `NOTICE`) on a copy of the frame resized to
   at most `detect_width` (1280) wide; score floor `detect_score` (0.6), NMS
   0.3. Boxes are mapped back to native pixels.
2. **Select** the largest box whose native side is ≥ `min_face_px` (40).
   (Multi-person clips: one face per clip, the largest, as in round 3; the
   count of faces found per frame is reported as `n_faces_seen`.)
3. **Confirm and landmark** with the MediaPipe Face Landmarker (IMAGE mode)
   on a square crop of side `landmark_margin` (2.0) × the YuNet box, at
   least 256 px (upscaled if smaller). No landmarks → the frame has no face.
4. **Crop for the detector** exactly as round 3 did: square box =
   landmark bounding box × `crop_scale` (1.3), `cut_square` to `crop_size`.
   The crop geometry is unchanged, so round-3 and round-4 scores differ only
   by which faces are found. The landmarks also feed the pulse path's ROIs
   (`face.track_landmarks` gains the same two-pass detection; VIDEO mode is
   replaced by per-frame IMAGE mode on the crop, with the box tracked frame
   to frame to avoid re-detecting on every frame).

All numbers live in `configs/p6_faceswap.yaml` (and the bundle's stored
config), including the Landmarker confidences that are hard-coded today
(A-09). Adoption rule: §6.1.

### 3.2 Detector v2: data

**Principle.** The round-3 detector learned "processing = fake" because
every FF++ fake carries a second generation of compression and resampling
that no FF++ real carries. Round 4 makes processing carry no label
information: every training video, real and fake, exists in several
processing variants, and the selection rule penalises sensitivity to a
processing source it has not seen (§6.2).

**Processing variants** (ffmpeg; applied to both classes; the face boxes are
found once on the original and reused for the variants, scaled to the
variant's frame size, so variants cost decode + crop, not detection):

| id | transform | why |
|---|---|---|
| v0 | as downloaded (c23) | baseline |
| c40 | FF++'s own c40 encode of the same videos (downloaded, not synthesised) | heavier compression, same content; same split and group as the c23 twin |
| v1 | scale to 640 px wide (never up), libx264 CRF 30, `veryfast` | the WhatsApp-like re-share (round-3 protocol, kept for comparability; the `min(640, iw)` fix is B-16) |
| v2 | scale to 1024 px wide, libx265 CRF 28 (libx264 CRF 26 if libx265 is absent) | the stock-clip look that caused 67% false alarms |
| v3 | VP9 WebM, 720 px short side, 2.5 Mb/s | the extension's tab capture (the `tabcapture_normalize` recipe already in `configs/p5_aigen.yaml`) |
| v4 | down to 360 px short side then back up to the source size, libx264 CRF 23 | upscaled re-uploads |

**Crop-level augmentation** at embedding time (cheap, GPU): per crop, K = 2
deterministic augmented copies drawn from JPEG q 40–90, downscale-upscale by
0.3–0.9, Gaussian blur σ ≤ 1.2, brightness/contrast ±10%. Stored as a third
feature set per video so the selection can compare "none / video variants /
video variants + crop augmentation" (§6.2).

**Sets and roles**

| set | role | contents / size | access | licence |
|---|---|---|---|---|
| FaceForensics++ c23 and c40: original, Deepfakes, Face2Face, FaceSwap, NeuralTextures, FaceShifter | train / val / test by the official identity-disjoint split; c40 twins share split and group | 6 × 1000 videos per quality (~11 GB c23, ~3 GB c40 estimate) | existing access (terms accepted); `train_faceswap.py download` with `--compression c40` added | FaceForensics ToS: non-commercial research and educational use only; no redistribution of the data; employer bound (kaldir.vc.in.tum.de/faceforensics_tos.pdf) |
| DeepFakeDetection (Google/Jigsaw actors) | external (cross-dataset) only, as in round 3 | 150 real + 300 fake, c23 (+ the same files in c40) | same download | same ToS |
| **DF40**, face-swapping subset on FF++ identities: FSGAN, FaceSwap, SimSwap, InSwapper, BlendFace, UniFace, MobileSwap, e4s, FaceDancer, DeepFaceLab | **training fakes (modern methods)** by the FF++ split of their identities, and the FF++-test identities as held-out methods (gate 1b) | raw videos via the project's Google Drive; cap 300 train+val + 140 test videos per method (~4,400) | form done; the Drive folder of the original FS/FR videos is in `runs/faceswap/round4_links.md`; the agent lists it first, verifies that file names map to FF++ ids (the DF40 JSON path files help), and fails loudly if they do not; the pre-processed frame sets are not used (different crop convention). The reenactment (FR) methods are not trained on: up to 4 of them × 140 FF++-test videos are scored and reported like Face2Face | CC BY-NC 4.0 → the shipped model is non-commercial, attribution in `NOTICE` |
| **Celeb-DF v2** | external: cross-dataset gate 3b (fakes vs reals of its official test list, 518 videos) and an outside-real set (590 Celeb-real + 300 YouTube-real) | ~14 GB | form done; Drive link in `runs/faceswap/round4_links.md`; downloaded inside the Kaggle kernel (`gdown`); if Drive's shared-file quota blocks it, Celeb-DF is processed on the laptop instead (~14 GB, ~3 h CPU; held-out only, so one backbone) | terms accepted in the form; test-only use here |
| GenBuster-Bench + AIGVDBench reals, the 706 clips processed in round 3 | **gate 4a, frozen**: the identical clip list, never used for training | on the laptop | — | public |
| OpenVid-1M clips with a face (optional) | extra training reals with stock-style processing (the GenBuster-real look); a held-out subset reported as "stock reals FPR" | ≤ 3,000 clips chosen from the dataset's CSV (captions naming a person speaking, an interview or a portrait; 4–20 s), fetched one zip member at a time by HTTP range requests (`remotezip`) from the > 50 GB parts on Hugging Face, so no part is downloaded whole; any clip the AI-video detector flags is dropped; grouped by source video id | public, no form; the agent proves the range fetch on 20 clips first, else the round runs without extra reals | CC-BY-4.0; its card says research and non-commercial purposes; attribution in `NOTICE` |
| UADFV (optional) | report-only external set (49 real + 49 FakeApp swaps, 2018, low resolution) | ~100 videos | link in `round4_links.md` | research use; cited |
| RepresentUs clip; `data/raw/user_eval/` | reported case by case, never tuned on | laptop only | — | user's own |

Real sources deliberately **not** used for training: GenBuster/AIGVDBench
reals (they are the gate; and the AI-video detector was trained on
AIGVDBench reals, so training the FSD on them would leave no real set on the
laptop unseen by both layers, B-"worth knowing"), Celeb-DF reals (test),
DFD reals (external). CelebV-HQ needs YouTube downloads from a cloud IP
(fragile); CelebAMask-HQ is images; Pexels' API key issuance is paused
(2026-11-07). OpenVid-1M is the likely origin of GenBuster's reals, so the
dedupe below is mandatory whenever it is used.

**Dedupe.** Before training, every OpenVid or DF40 clip whose mean CLIP
embedding has cosine ≥ 0.97 with any gate-4a, Celeb-DF or FF++-test clip is
removed and counted in the report (the `audit_aigen.py` near-duplicate
method). Content hashes are also compared.

**Manifest schema** (one CSV, built per source then concatenated; validated
by `check_manifest` extended to cross-dataset groups):
`video_id, dataset ∈ {ffpp, dfd, df40, cdf2, openvid, uadfv}, family (original | method name),
label ∈ {real, fake}, path, split ∈ {train, val, test, external}, group,
quality ∈ {c23, c40, raw}, variant ∈ {v0, c40, v1, v2, v3, v4}, source_id
(FF++ id pair, DF40 method+pair, Celeb-DF id, OpenVid source video id)`. Group keys:
FF++ and DF40-on-FF++ = `ffpp:<sorted pair>`; DFD = `dfd:<both actors>`
(B-17); Celeb-DF = `cdf2:<id>`; OpenVid = `openvid:<source video id>`; UADFV = `uadfv:<id>`. A variant row
inherits split, group and label from its v0 row; `lomo_auc` and
`oof_scores` group by `group`, so variants of one video never straddle a
fold. Extra reals carry `family == "original"` (B-"worth knowing").

### 3.3 Detector v2: model candidates and selection

- **Backbones:** `vit_base_patch16_clip_224.openai` (round 3; 598 MB
  weights) and `vit_large_patch14_clip_224.openai` (new candidate; weights
  ~1.2 GB, CPU inference ~4× slower: measure and report). DINOv2 candidates
  are dropped (round 3: 0.70–0.76 vs 0.84 LOMO).
- **Features:** CLS, CLS + mean patch. **C:** 0.01, 0.1, 1.
- **Training-variant sets:** `base` (v0 + c40), `video` (+ v1–v4),
  `video+crop` (+ crop augmentation). The same held-out sets are always
  evaluated on all variants.
- **Selection** (train + val only; §6.2 fixes the rule): repeated
  GroupKFold (3 repeats × 5 folds, seeds 7, 11, 13) for the
  leave-one-manipulation-out AUC and for a new **leave-one-processing-source-out
  false-alarm rate**; a tolerance of 0.01 on LOMO AUC decides ties in favour
  of the cheaper model (B/16 over L/14, CLS over CLS+patch).
- **Threshold:** the `(1 − target_fpr)` quantile (3%) of out-of-fold real
  scores, each processing source weighted equally.
- **Bundle:** adds `sklearn_version`, `torch/timm/mediapipe versions`,
  `git_commit`, the backbone weight file's SHA-256, the finder config, the
  per-source gate-4 counts, and a `config_hash` that covers every parameter
  that changes the result (B-08).

**Stage 2** (only if stage 1 fails gate 1 or gate 4a; same gates): the
existing `finetune` (EfficientNet-B4) on the saved v0 crops with the DFDC-
style augmentation (JPEG, downscale, blur, colour, flips) and the video
variants' crops; the threshold from validation reals, with the caveat in
B-"worth knowing" (epoch and threshold on the same split) stated in the
report. Stage 2 crops are scored through the same JPEG q95 round trip at
inference (B-15).

**Not in this round:** SBI-style training on reals only, Effort-style
adapters on CLIP, multi-face scoring. Listed as candidates for a later
plan if v2 fails gate 4a.

### 3.4 The extension's capture (v1.3.0)

1. **Full resolution:** `getUserMedia` constraints add `minWidth/minHeight
   = maxWidth/maxHeight = the tab's viewport in device pixels` (capped at
   1920×1080) and `maxFrameRate: 30`; MediaRecorder gets a
   `videoBitsPerSecond` (6 Mb/s at 1080p). The README's "max width 1280"
   claim is replaced by the measured capture size shown in the panel.
2. **Crop to the video element, server-side:** `probeMainVideo` runs with
   `allFrames: true`, picks the largest **playing** `<video>`
   (`!paused && readyState ≥ 2`), and returns its `getBoundingClientRect()`
   plus `innerWidth/innerHeight` and `devicePixelRatio`; the offscreen
   document scales the rect by `track.getSettings().width / innerWidth` and
   sends `{x, y, w, h, capture_w, capture_h}` with the upload. The backend's
   `analyze_video` gains `region=(x0, y0, x1, y1)` in source pixels; every
   decode path crops in memory before anything else (`ingest.open_video`
   before its downscale; `faceswap.sample_frames`; `aigen` via `-vf crop`
   in `normalize_for_inference`, and in `_decode_reservoir` when ffmpeg is
   absent). No browser re-encode. If the rect changes by more than 10%
   during the capture (fullscreen toggled, scrolled away), the capture is
   discarded with a message.
3. **Failure modes fixed:** C-01…C-13 (Appendix A): `event: error` parsed and
   shown, inactivity timeout on the SSE read, re-POST instead of re-GET on a
   drop, `current.active` cleared on every error path and keyed by tab,
   `currentTabId` set, upload refused after a track end or under
   `MIN_CAPTURE_SECONDS`, black-frame sampling through a `<video>` element,
   the content-script fallback without `insertCSS({files: []})`,
   `content_scripts`/`web_accessible_resources` removed (inject on click
   under `activeTab`), overlay host re-parented into
   `document.fullscreenElement`, `pulse_measurable` and `warnings` in the
   JSON and rendered in the panel, the probe note kept until the state
   changes, a start-up margin on the planned length.
4. **Contract:** `app.api_payload(a) -> dict` factored out and unit-tested;
   the JS harness reads the key list from a JSON fixture exported by that
   test; the README field list matches.

### 3.5 The app and the API

- Bind `127.0.0.1` by default; `--host` opt-in prints the same warning as
  `--share`. `gr.Blocks(delete_cache=(60, 300), analytics_enabled=False)`;
  `launch(max_file_size="2gb", show_error=True)`; uploads unlinked after
  analysis; the privacy text says "the backend keeps the clip only while
  analysing it". Both click events share `concurrency_id="analysis"`.
- `ingest.open_video(path, frame_height, max_seconds, region)`: the budget
  is checked against `min(n_meta, max_seconds·fps)`, decoding stops there,
  the no-frame-count path downscales and counts bytes like the main path,
  the cap is on the short side (portrait video), and "too large" is
  distinguished from "unreadable". `aigen` bounds its no-ffmpeg decode to
  `max_seconds` and takes that constant from one place.
- One `ingest.probe(path) -> (fps, n_frames, duration_s)` with the
  timestamp cross-check, used by the three layers and the progress plan
  (A-16, A-08).
- **C2PA:** decisive only when `validation_state ∈ {Valid, Trusted}`;
  otherwise "Content Credentials claim AI generation but could not be
  verified" is evidence, and the verdict falls through (A-"worth knowing").
- **Texts from bundles:** `CAVEAT`, the detail strings and the thresholds
  are rendered from the installed bundles' recorded numbers at
  `build_demo` time (the FSD's gate-4 rate, the AI detector's `target_fpr`,
  the pulse check's strict threshold and its measured recall/FPR stored in
  the G2 bundle); an overriding bundle that lacks them is refused with a
  message (A-07).
- Error strings carry `path.name` only (C-14); the progress plan clamps the
  frame count and estimates when unknown (A-08); `torch.set_num_threads`
  once at start (A-13); bundle construction inside the try blocks and the
  FaceFinder closed on the uncached path (A-14).

### 3.6 Everything the verdict says, if v2 becomes decisive

Headline "Possible face-swap" from `decided_by == "face_swap"`; the pulse
check becomes evidence only (docs/12 §5 B); the detail quotes the bundle's
gate numbers; the caveat quotes the measured false-alarm rates per source;
the website Docs threshold table, the Home "what decides" copy, the
extension caveat and popup copy, README tables, docs/04 §2 and §8 all
change in the same push cycle. If v2 stays evidence only, the same texts
quote the new rates. The exact list is in P4.

---

## 4. Everything else this round needs

Beyond §3, the audit items grouped by when they are done. Appendix A has
every finding with file:line, evidence and fix; this table names the
groups.

| phase | group | items (Appendix A ids) |
|---|---|---|
| P1, before any training | measurement validity | B-01 denominators; B-02 extraction errors ≠ no face; B-03 leakage validator for the AI-video training scripts; B-05 notebook/kernel steps halt on failure; B-06 grouped bootstrap; B-08 config hash + versions in bundle; B-09 `evaluate` refuses missing features; B-10 frozen gate-4 manifest + per-source counts; B-11 candidate grid must be complete; B-13 no-face vs error in reports; B-14 atomic npz writes, ffmpeg timeout, download exit code; B-15 stage-2 JPEG parity; B-16 `whatsapp_like` never upscales; B-17 DFD groups; B-04 stage-2 trigger uses `gate_passed` |
| P1, before any training | pipeline correctness on the laptop | A-01 ingest fallback path; A-02 budget vs 120 s cut; A-06 aigen no-ffmpeg bound; A-16 one probe; A-17 short-side cap; A-08 progress plan; A-14 bundle construction guards; faceswap WebM duration (M-02) |
| P1 | face finder v2 (§3.1) and its adoption test (§6.1) | — |
| P1 | extension capture and failure fixes (§3.4), app/API fixes (§3.5) | C-01…C-21, A-03, A-04, A-05, A-18, C-14 |
| P1 | honesty of texts and docs | B-01's correction note in docs/12 §9 and the website; D-03 tracked reports; D-04 `NOTICE` + licence metadata; D-07 `deploy/hf` removed or rewritten; D-09/A-10 docs/04 §4/§8 drift; D-15 AGENTS.md stale statements; D-17 README pointers; D-18 HUMAN_GATES `[dev]`; C-16 wording drift between overlay/popup/app |
| P1 | CI and packaging | D-01/D-02 a CI leg with `.[aigen,serving]`, cached weights, Node and ffmpeg, `-rs` with a skip allowlist; D-05 `scikit-learn>=1.9.1,<2` + version check on load; D-06 CPU torch index in `[tool.uv]`, `uv.lock` committed or ignored; D-10 ruff on `scripts/`, pinned, `[tool.ruff]`; D-11 one OpenCV distribution; D-12 bundle score tests; D-21 Python 3.11 + 3.12 matrix; D-25 duplicate scipy pin; D-13 golden tests recorded or removed; D-14 can't-fail assertions |
| P4 | integration texts | §3.6 |
| backlog (not this round) | hygiene | A-09 hr.py dead harmonic block; A-11 CHROM/POS band; A-15 eval bootstrap reuse; B-12 title-card rule's second clause; D-20 website numbers test; D-23 extension release workflow; D-16 `serve.py` and unused extras; D-22 `docs/.directory`; C-19 options `isLocal`; aria-live on the overlay |

---

## 5. Data access and licences, in one place

- **FaceForensics++ and DeepFakeDetection:** already granted. Terms:
  non-commercial research and educational use only; associates must accept
  the same terms; the employer is bound. The shipped `faceswap.joblib` is
  trained on this data; `README` says so and `NOTICE` will.
- **DF40:** CC BY-NC 4.0; form done, links received (Drive folders for the
  original face-swap and reenactment videos, plus pre-cropped training and
  test frames, ~93 GB each, not used here). Only the raw
  videos of the ten face-swap methods on FF++ identities are used, capped
  as in §3.2.
- **Celeb-DF v2:** terms accepted in its form; download link received; test-only.
- **OpenVid-1M** (optional): CC-BY-4.0; its card says research and
  non-commercial purposes; attribution in `NOTICE`. The AI-video filter guards
  against generated clips labelled real; the dropped count is reported.
  **UADFV** (optional): research use; cited. **Pexels:** not used, key
  issuance paused on 2026-11-07.
- **YuNet model:** MIT (opencv_zoo). **MediaPipe Face Landmarker:** Apache
  2.0 (already shipped).
- **The repo** stays MIT for code; `NOTICE` lists the model files and their
  data licences; `pyproject` gets `license` and `license-files`; the
  website's "MIT" stickers gain "(models: see NOTICE)".

---

## 6. Decision rules (fixed now, before any result)

### 6.1 Face finder v2 adoption (laptop, before the cloud run)

Adopt the two-pass finder (§3.1) only if all hold, measured by
`scripts/check_facefind.py` (new) and recorded in `reports/facefind.md`:

1. **Synthetic small faces:** FF++ test frames pasted at 30, 60, 100, 200
   and 400 px into 1280×720 and 1920×1080 canvases: a face found and
   landmarked in ≥ 95% of cases per size (the current finder: 0% below
   ~12% of the frame width).
2. **No regression on large faces:** on the 140 FF++ test reals and 119
   test fakes on disk, the share of clips with a face in ≥ 12 of 16 sampled
   frames is ≥ the current finder's share (today 134/140 reals have ≥ 4).
3. **Same crops where both find the face:** on those clips the v1 and v2
   landmark boxes agree within 5% of the side in ≥ 95% of frames (so round-3
   and round-4 scores stay comparable).
4. **Cost:** ≤ 150 ms per 1080p frame on this CPU for detection (measured
   82 ms at 1280×720 today), so a 16-frame clip adds ≤ 3 s.
5. **The user's extension capture** (`/tmp/gradio/*/truebeat-capture.webm`
   from 2026-11-07, local only) with the server-side crop of §3.4: a face
   in ≥ 12 of 16 frames.

If 1–4 pass, the finder is adopted for both the face-swap and the pulse
paths and the cloud run uses it. If it fails, the round continues with the
old finder and the failure is recorded; the small-face item returns in the
next plan.

*Revision (2026-11-07, before the full check; after a 6-clip dry run of
`scripts/check_facefind.py`, recorded as docs/12 §6 did):*

- *Rule 1's 30 px size is reported, not required: `min_face_px` (40, in
  the same config) makes both finders ignore faces under 40 px by design,
  so the size could never pass. The other sizes and both canvases stay
  required at ≥ 95%.*
- *Implementation, not a rule: the dry run showed landmarks taken on a crop
  differing from whole-frame landmarks by more than 5% of the face size in
  a third of frames (rule 3). The finder now runs the round-3 pass first
  and uses YuNet + crop only when that pass finds nothing, so every
  round-3 crop is reproduced exactly. Synthetic scenes are now face-free
  (the background frame's own faces filled flat and checked absent): the
  first version pasted faces into frames that already held a larger face,
  which the finder correctly returned.*
- *Scope: the finder is adopted for the face-swap detector only. The pulse
  path keeps its tracker, because its decisive threshold (−0.605) was
  measured with that tracker; feeding it small faces it never saw would
  change what the threshold means.*

### 6.2 Detector v2 selection and threshold (train + val only)

- **Grid:** backbone {B/16, L/14} × feature {CLS, CLS+patch} × C {0.01,
  0.1, 1} × training set {base, video, video+crop}. The grid may not shrink
  silently (B-11): a missing cell fails the run.
- **LOMO AUC:** leave-one-manipulation-out over all fake families present
  in train+val (the five FF++ methods and every DF40 method), repeated
  GroupKFold 3 × 5; macro mean and sd reported.
- **LOPSO FPR:** leave-one-processing-source-out: for each source
  s ∈ {c23, c40, v1, v2, v3, v4, openvid (if present)}, train on the other
  sources (all classes), set the threshold at 3% on the other sources'
  out-of-fold reals, and record the share of source-s reals above it. The
  statistic is the maximum over sources.
- **Choice:** among candidates with max-LOPSO-FPR ≤ 6%, the highest LOMO
  macro AUC; a candidate within 0.01 LOMO AUC of the best but cheaper
  (B/16 before L/14; CLS before CLS+patch; `base` before `video` before
  `video+crop`) wins the tie. If no candidate meets the LOPSO constraint,
  the lowest max-LOPSO-FPR wins, and the report says the constraint was not
  met.
- **Threshold:** `(1 − 0.03)` quantile of the chosen candidate's
  out-of-fold real scores, each processing source weighted equally.
- **Reference scores** for the percentile shown in the app: the same
  out-of-fold reals.

### 6.3 Adoption gates (all on data never used above)

Rates are **over judged clips** (a face found in ≥ 4 of 16 frames) and
always printed as "k / n judged (of N total)". Intervals: Wilson for rates;
bootstrap by `group` for AUCs (B-06).

| # | gate | pass if | applies when |
|---|---|---|---|
| 1 | FF++ c23 test, AUC vs its reals | ≥ 0.85 on each of Deepfakes, FaceSwap, FaceShifter | always |
| 1b | DF40 face-swap methods on FF++-test identities, AUC vs FF++ test reals | mean over the available methods ≥ 0.80, and no method < 0.70 | if DF40 was obtained |
| 2 | FF++ c23 test reals flagged | ≤ 5 / 134 judged (3.7%) | always |
| 2b | the same reals after v1 (WhatsApp-like), after c40, and after v3 (tab capture) | each ≤ 5% of judged | always (the round-3 lesson) |
| 3 | DeepFakeDetection | AUC ≥ 0.70 and reals flagged ≤ 8% of judged | always |
| 3b | Celeb-DF v2 official test list | AUC ≥ 0.75 and its reals flagged ≤ 8% of judged | if Celeb-DF was obtained |
| 4a | the frozen round-3 gate-4 list (706 clips, re-processed with the adopted finder) | flagged ≤ 3% of judged, n ≥ 200, and no single source > 6% | always |
| 4b | Celeb-DF v2 reals outside its test list (Celeb-real + YouTube-real not in the 518) | flagged ≤ 5% of judged | if Celeb-DF was obtained |

**Outcome.** Every applicable gate passes → **decisive**: "Possible
face-swap" is decided by the detector and the pulse check becomes evidence.
Gates 1, 2 and 2b pass, any of 3–4b fails → **evidence only**, and the new
bundle replaces the round-3 bundle only if its gate-4a rate is lower with
non-overlapping 95% intervals (else the round-3 bundle stays and the result
is recorded). Gate 1, 2 or 2b fails → stage 2 is tried under the same gates;
if that fails too, nothing ships and the round-3 bundle stays.

**Reported in every case, never gated:** Face2Face and NeuralTextures AUC;
the DF40 reenactment methods scored (§3.2); DF40 face-swap methods built on
Celeb-DF identities against Celeb-DF v2 reals (cross-dataset, modern
methods); UADFV; every held-out AUC and rate under v1–v4 and c40; per-source rates with n;
the share of clips with no face per set, before and after the finder
change; the RepresentUs clip (score, threshold, faces found); the
`user_eval` set; the dedupe counts; the LOMO sd; latency of the whole
analysis on this laptop for a 15 s 1080p clip with a face; the gate numbers
of a stage-2 model if trained.

### 6.4 The extension's crop adoption

A synthetic fixture (an FF++ clip composited into a 1920×1080 page frame
with UI around it, recorded as VP9 WebM by ffmpeg) sent through the
extension's backend path with the rect: the face-swap detector finds a face
in ≥ 12 of 16 frames and the AI-video layer scores the cropped region
(verified by the frame size in `W_PARTIAL_COVERAGE`'s source). Plus §6.1
item 5 on the user's real capture. Both pass → ship v1.3.0; else the
capture-resolution fix ships alone and the crop stays behind a flag.

### 6.5 Rules about the rules

Numbers above are not tuned after seeing results. If a rule proves
unmeasurable (a set never arrives, a gate's n is too small), the gate is
recorded as not applicable and the caveat says what was not measured. Any
other change is a dated revision in this section before the re-check, as
docs/12 §6 did.

### 6.6 Revisions (2026-11-07, recorded before any cloud result)

Made while implementing. No cloud session had produced a feature when
these were written (the only sessions so far were cancelled while
downloading, item 10). The local smoke run (60 FF++ clips) only checked
that the pipeline runs end to end; its numbers mean nothing at that size
and decided nothing here.

1. **Selection in two stages (§6.2 Grid, Choice).** Every cell × repeated
   folds × LOPSO is ~3,400 logistic fits on up to ~48k rows, too slow for
   one 4-vCPU session. **Stage A** picks the representation (backbone ×
   feature × C) by LOMO macro AUC on the base set (v0 + c40), repeated
   group folds 3 × 5 (seeds 7, 11, 13); candidates within 0.01 of the best
   go on, at most 4. **Stage B** scores each kept representation with each
   training set (base, video, video+crop) by LOPSO (one fold assignment,
   seed 7) and by LOMO on that training set. The §6.2 choice rule then
   applies to the stage-B cells unchanged.
2. **LOMO is scored on the held-out fold's v0 rows** (fits use every row
   of the training set), so the training sets are compared on the same
   rows.
3. **LOPSO sources** are v0 (c23), c40 and v1–v4 (no OpenVid, item 9).
   Crop-augmented rows are training material, not a processing source:
   they never set a threshold and are never counted.
4. **Threshold:** the 97% quantile of the chosen cell's out-of-fold reals,
   each processing source weighted equally, augmented rows excluded.
5. **Sample weights:** every (label, processing source) cell carries the
   same total weight in every fit, so processing carries no label
   information (§3.2's principle, made explicit).
6. **Faces are found on each variant's own frames** (§3.2 said: once on
   v0, boxes reused). The app runs the finder on the processed video, so
   this measures what the app will do, including finding fewer faces in
   degraded video. Measured cost: 2–3 s per video for all variants.
7. **Window:** every variant is cut from the first 20 s and its frames
   are sampled there (`window_s`); the app samples within the first
   120 s. FF++ originals have a median length of 16.9 s (308 local
   originals; 34% are longer than 20 s), so most training clips are seen
   whole.
8. **Encoders:** v2 uses hevc_nvenc (constant QP 30) where a session has an
   NVIDIA encoder, else libx265 `ultrafast` CRF 28 (the CPU sessions of
   item 10), else libx264 CRF 26; v3 is libvpx-vp9 in real-time mode
   (cpu-used 8, 2.5 Mb/s), as a browser records. The encoder is stored per
   video and variant.
9. **No OpenVid-1M reals.** GenBuster-Bench's reals most likely come from
   OpenVid-1M; training on other clips of the same source videos would make
   gate 4a partly in-distribution in a way the 0.97 dedupe cannot rule out.
   The training reals stay FF++ originals, as in round 3.
10. **Everything runs on the laptop (§7, Appendix C).** Measured: one FF++
    video with all six variants costs ~13 s on the laptop (Appendix C
    assumed ~1.5 s per re-encode), ~16 h for one 4-vCPU Kaggle session,
    so the Kaggle code shards by a stable hash of the video id (`fetch
    --shard`; DFD actor groups are recomputed over the whole manifest).
    Kaggle then **cancelled all five sessions of this pipeline**
    (CANCEL_ACKNOWLEDGED, no failure message, 1–4 min after start): three
    GPU and one CPU-only run at the start of the FF++ download, and the
    DF40/Celeb-DF run while it was only listing zip contents on Google
    Drive. The user did not cancel, and the account's phone is verified. A
    small probe kernel without this code ran normally earlier the same day.
    So the cause is not the TUM server; most likely it is Kaggle's
    automated moderation of the code, as Colab flagged the round-3
    notebook. The code is not renamed or disguised to get past it. **Every
    step runs on the laptop** (`scripts/run_fsv2_local.py`), the user's
    choice, started when the user says the laptop is free:
    - FF++ c23 + c40 and 450 DFD clips go into `data/raw/ffpp` with the
      user's FF++ access (human gate 3).
    - DF40 and UADFV are streamed zip by zip through `runs/` (fetch,
      extract, delete), so the disk holds. Celeb-DF v2's test list and
      reals go into `data/raw/cdf2`.
    - Faces, variants and crops run on the CPU (Ryzen 5 4600H). B/16
      embeddings run on the GTX 1650.
    - Selection runs on the CPU in the app's environment.
11. **L/14 leaves the grid.** CLIP L/14 costs ~4.6× B/16 per crop: about
    9 h for FF++'s ~825k crops on the GTX 1650 alone, and the app would run
    ~4× slower with it. The grid is backbone {B/16} × feature {CLS,
    CLS+patch} × C {0.01, 0.1, 1} × training set {base, video, video+crop}.
12. **torch:** the app's environment keeps torch 2.15.0+cpu. Extraction
    runs torch 2.15.0's CUDA build in a separate environment under `runs/`
    holding the same versions of every other package. Each
    `extract_log.json` records the versions and devices. The gates on the
    laptop (4a, the RepresentUs clip, `user_eval`) recompute embeddings on
    the CPU, so a GPU/CPU difference cannot hide.
13. **Dedupe (§3.2).** In `select`, before any fit, each DF40 training clip
    is compared with every held-out clip in the cloud features (FF++
    test, Celeb-DF, DF40 test and on Celeb-DF, DFD, UADFV) on its mean v0
    face embedding (B/16 CLS); cosine ≥ 0.97 removes it. Calibration on face
    crops (60 FF++ smoke clips, 1,770 pairs): unrelated clips peak at 0.93
    (median 0.58), clips cut from one video reach 0.97–0.99, so the cut
    keeps its meaning. Against gate 4a the comparison runs on the laptop,
    because those clips never leave it: `train_faceswap_v2.py overlap`
    compares the DF40 training embeddings that `select` exports with the
    gate-4a clips; any hit is excluded and selection reruns (`select
    --exclude`) before gate 4a is measured with v2. Content hashes are not
    compared: every DF40 clip is a generated fake, so none can be
    byte-identical to a reference clip.
14. **Round-3 reference on the frozen list** (for the §6.3 interval rule),
    measured before any v2 result: `reports/faceswap_gate4a_round3.md`,
    115 / 300 judged = 38.3% [33.0%, 43.9%] (706 processed; AIGVDBench
    9 / 142, GenBuster 106 / 158). This reproduces round 3's gate-4 count
    exactly.

---

## 7. Order of work

| phase | what | where | done when |
|---|---|---|---|
| 0 | this plan committed and approved (2026-11-07); token and dataset links in hand | laptop | done |
| 1 | audit fixes that affect measurement and the pipeline (§4, P1 groups); face finder v2 + `check_facefind.py` + report; manifest schema, variants, crop-box reuse, LOMO/LOPSO selection, grouped bootstrap, frozen gate-4 manifest under `data/manifests/faceswap_gate4a.csv` (paths + SHA-256 of the 706 clips); `region` crop through all decode paths; extension v1.3.0 code; app/API fixes; CI leg; NOTICE; docs corrections; tests for each | laptop | full tests green locally and in the new CI leg; `reports/facefind.md` written; smoke run of the whole pipeline (manifest → extract with variants → select → evaluate) on the 603 local clips with the local 1-worker budget |
| 2 | Kaggle kernels (`notebooks/kaggle/`): K1 download (FF++ c23+c40, DFD, the DF40 face-swap videos and Celeb-DF v2 from the links in hand, OpenVid clips if the range-fetch check passed) + faces + variants + crops, checkpointed to `/kaggle/working` (sharded into two CPU kernels if faces take > 8 h); K2 embeddings for both backbones and the crop augmentation (GPU); K3 select + held-out tables + report; K4 stage 2 if triggered. Each kernel halts on the first failure, writes `extract_log.json`, and the agent pulls only reports, manifests and bundles | Kaggle via the CLI | `runs/faceswap/kaggle_<date>/` holds bundle(s), report, manifests; leakage check passes on the returned manifest; the grid is complete |
| 3 | gate 4a on the frozen list and 4b/3b if available, the RepresentUs clip, `user_eval`, latency; `ship`; §6.3 outcome recorded in a new §10 here | laptop | `reports/faceswap_gate4a.md` (+ JSON with per-source counts); the bundle's `mode` written by `ship` only |
| 4 | integration texts (§3.6): app caveat and details from bundle fields, README, docs/04, docs/11 pointer, docs/12 §9 correction note, extension texts, website Updates entry + Docs/Home/RunLocally, `NOTICE` | laptop | every number shown is the measured one; live app run on an FF++ clip, the RepresentUs clip and the user's capture |
| 5 | review: `/code-review` on the round diff + an independent read of the Kaggle kernels; fix; full tests, CI-style core install, the aigen CI leg, the JS harness | laptop | no open findings |
| 6 | push the tool; extension v1.3.0 release (zip from the pushed commit; the human runs `gh release create` if the agent is blocked); website Updates entry pushed after the release exists | GitHub, Vercel | CI green; site deployed; release asset serves 1.3.0 |

Budget (estimates, to be re-measured in P1's smoke run): faces at ~1.5
videos/s on 4 vCPU → ~16k videos ≈ 3 h; variants (ffmpeg ~1.5 s each, 4 in
parallel) on 6,450 FF++/DFD videos × 4 ≈ 2.7 h; embeddings on a T4: B/16
~20 min, L/14 ~1.5 h for ~720k crops; stage 2 ≈ 2 h. Disk: v0 crops at
320 px JPEG (~6 GB) + variant crops at 224 px (~5 GB) + features (< 1 GB)
within `/kaggle/working`'s 20 GB; DF40 videos are processed method by
method and deleted. GPU quota used: 3–8 of 30 h.

---

## 8. Risks and honest expectations

- **Gate 4a may fail again.** Variants teach invariance to the processing we
  can synthesise; GenBuster's 67% may also come from content (stock lighting,
  lenses, grading). If so, the result is an evidence-only detector with a
  lower rate and a clearer caveat, and the next plan needs stock-style
  reals in training (OpenVid by range fetch is the lever if this round did
  not get to use it).
- **Google Drive throttles large shared files** ("too many users have
  viewed or downloaded"). Celeb-DF then moves to the laptop; DF40 is fetched
  method by method with retries over days; a set that still cannot be
  fetched makes its gates not applicable, and the caveat says what was not
  measured. The pipeline is resumable, so a later kernel can add it.
- **The RepresentUs ad may still be missed.** A professional swap,
  re-compressed by WhatsApp, is the hardest case; it is reported, never
  tuned on.
- **Kaggle:** GPU availability is sometimes degraded; CPU kernels carry the
  face work; quotas and session limits must be read at the first cell.
- **A larger backbone costs users a 1.2 GB download** and ~4× the CPU time
  of the face layer. The tie rule prefers B/16 unless L/14 is clearly
  better.
- **Repeated folds make selection slower** (15 fits per candidate per
  family) but it is seconds on CPU; the round-3 margin (0.005) was noise.
- **Stage 2's threshold is optimistic by construction** (validation picks
  both the epoch and the threshold); its gates are read with that caveat.

---

## 9. Human gates (one action each)

| # | action | unblocks | how |
|---|---|---|---|
| 1 | Approve this plan in writing | everything | **done 2026-11-07** ("approve the plan") |
| 2 | Kaggle API token | phase 2 | **done**: saved at `~/.kaggle/access_token` (local only). Phone verification is needed for internet inside kernels; do it only if the agent reports that the first `kernels push` with `enable_internet` was refused |
| 3 | DF40 download form | gate 1b, modern methods in training | **done**: links recorded in `runs/faceswap/round4_links.md` |
| 4 | Celeb-DF v2 form | gates 3b and 4b | **done**: Drive link recorded in the same file |
| 5 | Pexels API key | — | not available: key issuance is paused (2026-11-07); OpenVid-1M replaces it in §3.2, no action |
| 6 | Review the result report (§10, written in phase 3) | phases 4–6 | read `reports/faceswap_v2_report.md`; approve the outcome the rules produced |
| 7 | If the agent's release command is blocked: run the `gh release create extension-v1.3.0 …` line it prints | phase 6 | one command in the terminal |

---

## 10. Result (2026-11-08)

Everything ran on the laptop (§6.6 item 10), from 2026-11-07 13:05 to
2026-11-08 23:14, including a 14.5 h wait while Google Drive refused the
DF40 downloads ("Quota exceeded", 06:12 to 20:36). Selection at commit
`676d4c3`, config hash `aaaae705`. Reports: `reports/faceswap_v2_report.md`
(selection and every held-out table), `reports/faceswap_gate4a.md` (gate 4a),
`reports/faceswap_v2_overlap.json`, `reports/faceswap_v2_decision.json`.

**Data.** 12,947 videos in 23 manifest parts:

- FaceForensics++ c23 and c40: 1,000 originals and 5,000 fakes over
  Deepfakes, Face2Face, FaceShifter, FaceSwap and NeuralTextures, on the
  official identity splits.
- DFD: 150 real, 300 fake (external).
- DF40: 2,400 training fakes (300 per method from 8 face-swap methods, on
  FF++ train/val identities). Its test zips hold 11 methods on FF++ test
  identities and 8 on Celeb-DF identities; they are report-only, except
  gate 1b.
- Celeb-DF v2: the 518-clip test list (178 real, 340 fake) and the 712
  reals outside it.
- UADFV: 49 real, 100 fake (report only).

Two FF++ videos lost one variant each (an ffmpeg timeout during a pause of
the run). Selection leaves those two variants out.

**Dedupe and overlap.** No near-copies on either side:

- vs the 5,383 held-out clips: 0 of the 2,400 DF40 training clips at cosine
  ≥ 0.97 (nearest: median 0.842, max 0.961).
- vs the gate-4a clips (440 with a face): 0 of 2,400 (nearest: median
  0.738, max 0.878).

**Selection (§6.2).**

- **Stage A** (LOMO macro AUC on the base set, 3 × 5 group folds): cls
  C=0.01 scored 0.890 (sd 0.003), cls+patch C=0.01 scored 0.887 (sd 0.003).
  Larger C scored lower.
- **Stage B:** no candidate met LOPSO ≤ 6%. The c40 source dominates
  (14.3% to 47.4% of its reals cross the threshold when it is left out).
  The rule therefore chose the lowest LOPSO: **cls+patch, C=0.01, training
  set `video`** (all six variants), LOPSO max 14.3%, LOMO 0.861.
- **Threshold:** +3.1987, set at 3% on out-of-fold reals of the training
  set, sources weighted equally. In-domain out-of-fold AUC on v0: 0.898.

**Gates (§6.3).** Rates are over judged clips, with Wilson 95% intervals.
AUC intervals resample identity groups.

| # | gate | measured | pass |
|---|---|---|---|
| 1 | FF++ c23 test AUC ≥ 0.85 (Deepfakes, FaceSwap, FaceShifter) | 0.973 [0.958, 0.986], 0.938 [0.916, 0.960], 0.915 [0.890, 0.938] | yes |
| 1b | DF40 face-swap methods on FF++ test identities: mean ≥ 0.80, none < 0.70 | mean 0.926, min 0.873 (facedancer), 7 methods | yes |
| 2 | FF++ c23 test reals flagged ≤ 5/134 | 1/140 = 0.7% [0.1, 3.9] | yes |
| 2b | the same reals after v1, c40 and v3, each ≤ 5% | 3/140 = 2.1% [0.7, 6.1]; 3/140 = 2.1%; 1/140 = 0.7% | yes |
| 3 | DFD: AUC ≥ 0.70, reals ≤ 8% | AUC 0.819 [0.785, 0.865]; 7/150 = 4.7% [2.3, 9.3] | yes |
| 3b | Celeb-DF v2 test list: AUC ≥ 0.75, reals ≤ 8% | AUC 0.778 [0.735, 0.818]; 20/178 = 11.2% [7.4, 16.7] | **no** |
| 4a | frozen 706-clip list: ≤ 3% of judged, n ≥ 200, no source > 6% | 61/383 = 15.9% [12.6, 19.9]; AIGVDBench 18/197 = 9.1%, GenBuster-Bench 43/186 = 23.1% | **no** |
| 4b | Celeb-DF v2 reals outside the test list ≤ 5% | 63/711 = 8.9% [7.0, 11.2] | **no** |

**Outcome: evidence only; v2 replaces the round-3 bundle.** Gates 1, 2
and 2b pass; 3b, 4a and 4b fail. On the same frozen list, v2 flags 15.9%
[12.6, 19.9] where round 3 flagged 115/300 = 38.3% [33.0, 43.9]. The
intervals do not overlap, so `ship` wrote v2 as
`src/trubeat/models/faceswap.joblib` (`mode: evidence`, SHA-256
`3c125d70…`).

The face-swap layer still never decides the verdict; the pulse check
decides as before (docs/12 §5 B). The caveat, and the extension's fallback
caveat, quote 16% from the bundle's claims. Stage 2 was not tried: it runs
only when gate 1, 2 or 2b fails.

**Reported in every case, never gated:**

- **Reenactment fakes, FF++ test (v0):**
  - AUC: Face2Face 0.911 [0.885, 0.937], NeuralTextures 0.833
    [0.806, 0.862].
  - Caught at the threshold, of 140 each: Deepfakes 73.6%, FaceSwap 56.4%,
    Face2Face 47.9%, FaceShifter 37.1%, NeuralTextures 16.4%.
- **Processing, FF++ test** (reals flagged, then Deepfakes AUC):

  | variant | reals flagged | Deepfakes AUC |
  |---|---|---|
  | c40 | 2.1% | 0.937 |
  | v1 (WhatsApp-like) | 2.1% | 0.933 |
  | v2 (stock-clip HEVC) | 1.4% | 0.968 |
  | v3 (the extension's VP9 capture) | 0.7% | 0.975 |
  | v4 (upscaled re-upload) | 0.0% | 0.965 |

  Round 3's WhatsApp-like false alarms were 45/133 judged = 33.8%
  [26.3, 42.2] (docs/12 §9, correction); v2's are 3/140 judged = 2.1%.
- **DFD under processing:** reals flagged 4.7% (v0) to 9.3% (c40); AUC
  0.58 (v1) to 0.82 (v0).
- **DF40, FF++ test identities:**
  - reenactment and lip-sync methods: AUC facevid2vid 0.895, fomm 0.955,
    pirender 0.903, wav2lip 0.681 (caught 2/140);
  - the 7 face-swap methods: AUC 0.873 to 0.972.
- **DF40 face-swap methods on Celeb-DF identities**, against Celeb-DF
  reals (cross-dataset): AUC 0.693 (simswap) to 0.983 (faceswap).
- **Celeb-DF v2 test fakes:** caught 131/340 = 38.5%.
- **UADFV:** AUC 0.870 [0.827, 0.917]; reals flagged 3/49.
- **Clips with a face** (judged, of N):
  - every FF++ test clip in every variant;
  - all but 1 of 712 Celeb-DF reals outside the test list;
  - all but 1 of the 1,200 DF40 fakes on Celeb-DF identities;
  - all but 1 of 100 UADFV fakes;
  - DFD: 297 to 300 of the 300 fakes per variant, and 149 of the 150
    reals in v1 and v2.

  On the frozen gate-4a list, the two-pass finder judges 383 of 706 clips;
  round 3's Landmarker judged 300 (`reports/facefind.md`).
- **The RepresentUs clip** (report only, never tuned on): v2 scores +2.71
  against threshold +3.20, face in 13 of 16 sampled frames. It is **not
  flagged**, but it scores above 96.5% of the model's real training faces.
  Round 3: −0.55 against +1.58 (13 of 16), not flagged.
- **`user_eval`:** not measured; no labelled clips on disk
  (`data/raw/user_eval` is absent).
- **Latency on this laptop** (Ryzen 5 4600H, 7 GB, CPU, all four layers): a
  15 s 1080p clip with a face takes 23.8 s cold (models load) and 20.4 s
  warm.
- **LOMO sd:** 0.003 for the chosen representation.

**What was not measured.**

- Gate 4a through the extension's capture path on live pages (§6.4 checked
  the crop on one capture).
- OpenVid reals (not used, §6.6 item 9).
- L/14 (dropped, §6.6 item 12).
- `user_eval`.

**Laptop numbers.**

- FF++ extraction: 6,450 videos × 6 variants in 14.1 h wall
  (0.13–0.15 videos/s, 3 workers, GTX 1650).
- DF40 test zips: ~2.2 videos/s (v0 only).
- Selection: 50 min at 2 parallel fits. Four fits ran the 7 GB laptop out
  of memory in a rehearsal.
- Gate 4a: 29 min at 2 workers.

---

## Appendix A — Audit findings (2026-11-07)

Four read-only audits (A: core Python; B: training scripts and notebook;
C: extension and web UI; D: tests, CI, packaging, docs, website) plus the
author's read (M). Severity: **H** = wrong result, crash, privacy or
honesty failure a user can hit; **M** = wrong under realistic conditions or
misleading; **L** = hygiene. Phase = where it is fixed (P1 = before any
training; P4 = with the integration texts; BL = backlog).

### A. Core Python (`src/trubeat`)

| id | sev | where | finding (evidence) | fix | phase |
|---|---|---|---|---|---|
| A-01 | H | `ingest.py:148-178` | The no-frame-count path (the extension's WebM: `CAP_PROP_FRAME_COUNT` = −2⁶³) skips the 480 px downscale and the `MAX_DECODED_BYTES` guard, keeps every native frame in a list, then stacks it (2× peak). 60 s of 1280×720 ≈ 10 GB → OOM on 7 GB. Returns the 480 px width/height that were never applied | `_prep` in the fallback loop, running byte budget, stop at `max_seconds`, real dims | P1 |
| A-02 | H | `ingest.py:130-138`, `app.py:487-492` | The 2.5 GB budget is checked on the whole clip before the 120 s cut: 1080p30 clips > ~68 s (60 fps > 34 s) raise `E_UNREADABLE`, the pulse layer never runs, and the headline can say "could not read the video" | `open_video(..., max_seconds)` checks `min(n_meta, max_frames)`, stops decoding there, keeps `W_TRUNCATED`; "too large" ≠ "unreadable" | P1 |
| A-03 | M | `app.py:1183` | `server_name="0.0.0.0"`: the biometric app is on every interface without auth; README, extension and AGENTS.md assume loopback. Gradio echoes any Origin for a LAN Host, so a LAN page can drive the API | default `127.0.0.1`, `--host` opt-in with a warning | P1 |
| A-04 | M | `app.py:922,947,1182` | Uploads and the suspicious-frame PNG are kept indefinitely (`delete_cache=None`), uploads unbounded (`max_file_size` unset); on tmpfs `/tmp` a big upload fills RAM | `delete_cache=(60, 300)`, unlink after analysis, `max_file_size="2gb"`, document `GRADIO_TEMP_DIR` | P1 |
| A-05 | M | `app.py:957-962, 1019-1028` | The UI click and the API click are separate events with their own concurrency limit of 1: two analyses can run at once, sharing `_FS_CACHE`/`_AIGEN_CACHE`, `torch.set_num_threads` and the IMAGE-mode landmarker; two 2.5 GB buffers → OOM | `concurrency_id="analysis"` on both, plus a module lock | P1 |
| A-06 | M | `aigen.py:117-168, 76-102, 577` | Without ffmpeg the AI layer decodes the whole upload (a 1 h clip ≈ 4.3 GB of kept frames) and spreads segments over the full length; the 120 s cap exists only as `-t 120` in the ffmpeg command | stop at `max_seconds` by timestamp; one shared constant | P1 |
| A-07 | M | `app.py:40-46,158,53-65,664,699` | The strict pulse threshold (−0.605) and the quoted numbers ("a quarter", "2%", "38%", "5%") are code constants while `models/*.joblib` overrides are allowed; nothing ties them to the bundle that scored the clip; the AI path hard-codes "5%" while the FSD path reads `target_fpr` | store thresholds and measured numbers in each bundle; render texts from bundle fields; refuse overriding bundles without them | P1 |
| A-08 | M | `app.py:124-132, 96-110` | `_frame_count` returns raw metadata: −2⁶³ for WebM → negative stage costs, bar pinned at 99% through decode and tracking (same bug as `5714521` fixed in `faceswap.py`) | clamp; estimate from timestamps when unknown; reuse one probe | P1 |
| A-09 | L | `hr.py:75-116` | The "harmonic awareness" block is dead: the strongest peak always wins (`0.6·sin 1.2 Hz + 1.0·sin 2.4 Hz → 144 bpm`); docstring promises the sub-harmonic | delete the block and the claim, or implement with a property test | BL |
| A-10 | L | `docs/04:253,256`, `model.py:116-120`, `app.py:543-591` | `W_LOW_CONFIDENCE` is never emitted; `E_NO_FACE` is raised only when no frame has a face (docs say < 70%); the app emits no `W_UNSTABLE_TRACK` below 70% | align docs/04 §8; emit `W_UNSTABLE_TRACK` in `_pulse_layer` | P1 |
| A-11 | L | `signal.py:76,88`, `configs/p2_detector.yaml` | CHROM and POS always band-pass at (0.7, 3.0); the config's `band: [0.7, 14.0]` is ignored for them | thread `band` through `extract_ppg` or state the limit in the YAML | BL |
| A-12 | L | `faceswap.py:226-227` | Landmarker confidences (0.5) hard-coded, outside the config and the hash | into `configs/p6_faceswap.yaml` and the bundle config | P1 |
| A-13 | L | `aigen.py:353`, `faceswap.py:300` | `torch.set_num_threads` is process-global and set inside each `_load` (6 then 4); the aigen bundle's 6 is honoured once | set once at start from one config value | P1 |
| A-14 | L | `app.py:274,302-305,498-499,307` | Bundle construction outside the try blocks (`Embedder(bundle["config"])`, `FaceEmbedder(...)`, `float(bundle.get("window_s"))`): a malformed overriding bundle escapes; the uncached CNN path never closes its FaceFinder | move inside the try; close the finder | P1 |
| A-15 | L | `eval.py:165-188` vs `detector.py:270-305` | Two bootstraps: `evaluate_detector` resamples clips, `detector.bootstrap_auc_ci` resamples groups; `trubeat eval` CIs ignore the real/fake pairing | reuse the grouped one | BL |
| A-16 | L | `app.py:124`, `ingest.py:107-184`, `aigen.py:137-199`, `faceswap.py:123-166` | Four fps/frame-count readers with different semantics and caps; the evidence row "clip length" shows 120.0 s for a 300 s upload when the pulse layer did not run | one `ingest.probe()` used everywhere; report the probed duration | P1 |
| A-17 | L | `ingest.py:126-129` vs `aigen.py:105-114` | The pulse layer caps frame height (portrait 1080×1920 → 270×480, face ~100 px) while the AI layer caps the short side | cap the short side (same memory bound) | P1 |
| A-18 | L | `app.py:1153` | `GRADIO_ANALYTICS_ENABLED=0` only in `main()`; `build_demo().launch()` from code starts Gradio's version-check thread | `gr.Blocks(analytics_enabled=False)` | P1 |
| A-19 | note | `app.py:646-657` | An **unsigned** C2PA manifest declaring `trainedAlgorithmicMedia` produces the confident headline "AI-generated: declared by its Content Credentials" (`W_C2PA_UNVERIFIED` never changes the verdict): any real video can be relabelled with a text edit | decisive only when `validation_state ∈ {Valid, Trusted}`; otherwise evidence | P1 |
| A-20 | note | `faceswap.py:232-248` vs `face.py:57` | The FSD scores the largest of up to 3 faces; the pulse path locks onto whichever face VIDEO mode found first: two-person clips can be judged on different people | §3.1 selects one face for both paths | P1 |

### B. Training and evaluation tooling (`scripts/`, notebook)

| id | sev | where | finding (evidence) | fix | phase |
|---|---|---|---|---|---|
| B-01 | H | `train_faceswap.py:403-446` | Gates 2–3 count unjudged reals as not flagged (`nan → −inf ≥ thr`); gate 4 counts judged clips only. Round 3 gate 3: 3/150 = 2.0% reported, 3/67 = 4.5% [1.5, 12.4] on judged; FF++ reals 1/134 not 1/140 | rates over judged clips, "k / n judged (of N)", same in `check_faceswap_reals.py`; correction note in docs/12 §9, README, website | P1 |
| B-02 | H | `train_faceswap.py:286-292,301-303,338-343` | Extraction errors are saved as empty npz and read back as "no face"; the fail-fast fires only when 100% of the first ≥ 50 fail (`errors == k >= 50`); `extract` exits 0 regardless | `load_features` returns `error`; `select`/`evaluate` refuse when errors exist; non-zero exit; `extract_log.json` in the result | P1 |
| B-03 | H | `train_aigen.py:352-367`, `eval_multiseg.py`, `compare_aigen_models.py` | The AI-video detector was trained, re-fitted and compared without the identity-leakage validator AGENTS.md §6 requires; roles are assigned by `source.startswith("genbuster")` | role-aware manifest + `validate_manifest` at the top of each script; a content-hash/cosine duplicate gate | P1 |
| B-04 | M | notebook cell 8 | Stage-2 trigger uses `ok is not False`, so an "n/a" family passes (the opposite of `gate_passed`); an empty stdout triggers a multi-hour fine-tune | call `gate_passed`; check return codes | P1 (kernels) |
| B-05 | M | notebook cells 4–7, 9 | `!` cells do not stop "Run all" on failure; a failed `select` cascades into fine-tune, a broken `extract --backbones ""`, and a zip without the bundle | `subprocess.run(check=True)` per step; assert outputs exist | P1 (kernels) |
| B-06 | M | `train_aigen.py:294-313` | Bootstrap and Wilson treat clips as independent; DFD has 28 actors, FF++ pairs contribute 2 reals + 2 fakes per family: the DFD CI 0.878 [0.831, 0.924] is too narrow | resample by `group`; print n_identities | P1 |
| B-07 | M | notebook cell 9, cells 0–1 | The WhatsApp re-extract has no checkpoint and runs last; "downloads resume from Drive" is false; FaceShifter and DFD re-encode rows were never produced | checkpoint every variant extract; run variants before `select`; correct the text (moot once §3.2 variants exist) | P1 |
| B-08 | M | `faceswap.py:73-77`, bundle | `config_hash` omits `target_fpr`, `min_seconds`, the chosen `C`, finder confidences; the bundle records no library versions, git commit or weight-file SHA; the notebook installs unpinned torch/timm | extend the hash; record versions and hashes; pin; a laptop self-check that re-embeds stored crops | P1 |
| B-09 | M | `train_faceswap.py:754-757` | `evaluate` with a wrong `--features-dir` prints "0/10 flagged, AUC nan" and gate 2 = pass; the section title is free text | fail when no feature file exists; store transform + config hash in each npz and print them | P1 |
| B-10 | M | `check_faceswap_reals.py:58-100` | The gate-4 set is a shuffled prefix of a gitignored pickle, stopped by a face count; only the pooled 38% reaches the bundle (sources: 67% and 6.3%) | `data/manifests/faceswap_gate4a.csv` (paths, sources, SHA-256); per-source k/n in JSON and bundle; the app states per-source rates | P1 |
| B-11 | L | `train_faceswap.py:491-493` | A candidate without features is skipped with a log line; the fixed grid can shrink silently | fail unless `--allow-missing`; assert the grid | P1 |
| B-12 | L | `check_graphics_guard.py:77-108`, `aigen.py:51-55` | Only the first clause of the title-card rule is enforced; the report header omits the motion criterion; parameters live in code | compute the second clause; print both; move to YAML and bundle | BL |
| B-13 | L | report writers | "N without enough faces" counts errors too; "706 scored" for 406 unscored clips; the gate-4 report was edited by hand after the run (the bundle line) | the script writes provenance; "processed / judged" wording | P1 |
| B-14 | L | `train_faceswap.py:300-308,242-251,202-206,141-149` | npz written in place (a killed run leaves a truncated file that crashes the resume); `whatsapp_like` has no timeout; `download` exits 0 with missing files | tmp + `os.replace`; unreadable npz = not extracted; `timeout=`; non-zero exit with the missing list | P1 |
| B-15 | L | `train_faceswap.py:209-218,615-618` vs `faceswap.py:572-580` | Stage-2 crops are JPEG q95 in training/evaluation but raw at inference | same round trip in `analyze` for `kind == "cnn"` | P1 |
| B-16 | L | `train_faceswap.py:202-206` | `scale=640:-2` upscales clips narrower than 640 px | `scale='min(640,iw)':-2` | P1 |
| B-17 | L | `train_faceswap.py:174-176` | DFD group = target actor only; harmless while external, wrong if DFD is ever trained on | group by both actors; a `check_manifest` test that can fail across datasets | P1 |
| B-18 | note | round 3 | The LOMO winner beat the runner-up by 0.005 with per-family AUCs from ~28 vs ~28 clips per fold | §6.2: repeated folds and a 0.01 tolerance | P1 |

### C. Extension and web UI (`extension/`, `app.build_demo`)

| id | sev | where | finding (evidence) | fix | phase |
|---|---|---|---|---|---|
| C-01 | H | `offscreen.js:467-472,369-385`; `app.py:1182` | A backend exception leaves "Analysing…" forever: Gradio sends `event: error` with `data: {"error": null}` (`show_error` is False), the code only checks `parsed.error`; the re-GET then hangs on heartbeats (no read timeout; the 6 min deadline is checked only between attempts) | parse `event:` lines, treat `error` as terminal; `show_error=True`; inactivity timeout via `AbortController` | P1 |
| C-02 | H | `offscreen.js:554` | The DRM/black-frame guard is dead: `drawImage(track)` throws (a `MediaStreamTrack` is not a `CanvasImageSource`) and is swallowed; 20 s of black is uploaded; README and QA checklist promise otherwise | draw from a `<video>` with `srcObject`, or `MediaStreamTrackProcessor`; test `sampleFrame` with a stubbed canvas | P1 |
| C-03 | H | `background.js:93-100` | `CAPTURE_ERROR` never clears `current.active`: after any capture-side failure every click says "Already running" until the worker idles, and each click resets the idle timer | clear on every error path; key state by tab; clear on `tabs.onRemoved` | P1 |
| C-04 | H | `offscreen.js:493-503,114-156` | `currentTabId` is never assigned: "track ended"/"recorder error" go to tab `null` and are dropped, while the partial recording is uploaded and judged | set it; a `failed` flag checked in `onstop`; refuse uploads under `MIN_CAPTURE_SECONDS` by elapsed time | P1 |
| C-05 | M | `offscreen.js:30-37,365-385`; README; tests 3d | "Gradio replays a completed event" is false in 6.30: after a disconnect the job is marked dead and a re-GET waits on heartbeats forever; test 3d(a) asserts the replay | re-POST the same uploaded path on a drop; one connection with a read watchdog; rewrite the test | P1 |
| C-06 | M | `offscreen.js:101-109,22,547`; README | No capture size is requested (whole tab at Chrome's default: 960×540 on the user's screen, up to 2880×1800 on DPR-2); "max width 1280" is only the dead sampling canvas; the element rect never leaves the page | §3.4 items 1–2 | P1 |
| C-07 | M | `background.js:354-357` | The content-script fallback never runs: `insertCSS({files: []})` rejects before `executeScript`; tabs opened before install get no overlay at all | drop `insertCSS`; test with a rejecting `tabs.sendMessage` | P1 |
| C-08 | M | `manifest.json:42-55` | `<all_urls>` content script and `web_accessible_resources` although injection on click already works: "read and change all your data on all websites" prompt and a fingerprinting vector | remove both; inject on click | P1 |
| C-09 | M | `app.py:977-1015`; `overlay.js:181-197` | The JSON omits `pulse_measurable` and the overlay never renders `qc.warnings`: an unqualified heart rate is shown when the backend says the pulse is noise; W_TRUNCATED/W_PARTIAL_COVERAGE/W_GRAPHICS_SKIPPED never reach the user | add both to the JSON; render them | P1 |
| C-10 | M | `background.js:148-166,206-216` | The probe runs in the top frame only (no iframes, no shadow DOM) and picks the largest `<video>` even if paused: embedded players get the configured 20 s across loop seams; a paused hero banner blocks a playing article video | `allFrames: true`; largest **playing** element | P1 |
| C-11 | M | `overlay.js:52-53,96` | In fullscreen (the README's own tip) the body-level fixed panel is not painted | re-parent into `document.fullscreenElement`; listen to `fullscreenchange` | P1 |
| C-12 | L | `app.py:124-131` | (= A-08) | | P1 |
| C-13 | L | `background.js:73-76,237-242`; `offscreen.js:160-174` | The plan note ("Short clip: recording 6 s") is overwritten after 500 ms; the stop timer ignores ~1 s of start-up latency, so a looping 8 s clip is still recorded into its restart | carry the note in PROGRESS; `floor(left − 1)` then clamp | P1 |
| C-14 | L | `app.py:470-482,738-746`; `ingest.py`, `faceswap.py`, `aigen.py` error strings | Absolute server paths in error texts reach the overlay and the evidence table (seen in the user's capture: `/tmp/gradio/…/truebeat-capture.webm`) | `path.name` only; full paths in the server log | P1 |
| C-15 | L | `app.py:901-902,918` vs `951-953` | The caveat loses its "Caveat: always applies:" heading after the first run | `gr.update()` or drop it from outputs | P1 |
| C-16 | L | `extension/README.md:36-44`, `overlay.js:16-38`, `popup.html:59-62`, `offscreen.html:9-13` | "Exactly three states, same wording as the app" is pinned to fallback strings the app never emits ("Likely AI-generated or manipulated"); popup copy is from 1.0 ("Detects face-swaps…"); the offscreen comment says it never touches the network | overlay labels and caveat from the backend payload; update copy; the test compares against the backend contract | P1 |
| C-17 | L | `options.js:19-35,48-54`; `manifest.json:14-17` | `isLocal` treats unparsable URLs as local (no permission requested) and `x.hf.space:7860` as a scheme; `[::1]` counted local but not in `host_permissions` | normalise like `background.js`; add `http://[::1]/*` | BL |
| C-18 | L | `serve.py` | Dead FastAPI stub whose only endpoint raises | delete, or wrap `analyze_video` | BL |
| C-19 | L | `background.js:300-302` | `current.active` resets when the upload finishes, so a second capture can start during analysis and the later result overwrites the panel unlabelled | keep active until `ANALYSIS_RESULT`/error | P1 |
| C-20 | note | tests | `sampleFrame`, `fail()`, the `active` lock, the fallback injection, size constraints, real SSE shapes (`event: error`, heartbeats), `probeMainVideo`, `build_demo`/`analyze_json` are untested; `test_no_remote_code_in_html_pages` skips every `<script src>` tag (`tag.startswith(">")`); the static fetch audit scans `background.js` only | tests for each in P1 | P1 |
| C-21 | note | overlay | no `role="status"`/`aria-live`: screen readers never hear the verdict | add | BL |

### D. Tests, CI, packaging, docs, website

| id | sev | where | finding (evidence) | fix | phase |
|---|---|---|---|---|---|
| D-01 | H | `tests/`, `models/aigen.joblib` | The shipped AI-video bundle is never loaded or scored by any test; the real DINOv2 embedder never runs in CI (`skipif(not _weights_cached())`) | a CI-safe load+score test (384-d random unit vectors, threshold, reference scores); an aigen CI leg | P1 |
| D-02 | H | `ci.yml:13-14` | CI installs core deps only: embedders, C2PA reader, CNN scorer, Gradio wiring never execute; `test_plain_video_has_no_c2pa` passes without the library; `-q` hides the skip list | a second job with `.[aigen,serving]` + CPU torch index + cached weights + Node + ffmpeg; `-rs` with an allowlist; `importorskip("c2pa")` | P1 |
| D-03 | M | `README.md:74`, `app.py:58`, docs/11–12 | The README and the always-visible caveat cite `reports/aigen_detector_v2.md`, which is not tracked; 13 more cited reports are untracked | force-add the markdown/JSON reports (no faces), or cite docs/11 §9 and docs/12 §9 | P1 |
| D-04 | M | `LICENSE`, `pyproject.toml`, website | The FF++ non-commercial restriction on `faceswap.joblib` is README prose only: no `NOTICE`, no `license` metadata; the site says "MIT. Free forever." three times; `deploy/hf/README.md` says `license: mit` | `NOTICE`, `license`/`license-files`, one sentence on the site | P1 |
| D-05 | M | `pyproject.toml:16` | `scikit-learn>=1.3`, bundles pickled with 1.9.1 | `>=1.9.1,<2`; `sklearn_version` in bundles, checked on load | P1 |
| D-06 | M | `uv.lock` (untracked), `pyproject` | The lock resolves `aigen` to CUDA torch (43 nvidia packages); no CPU index source; the lock is neither committed nor ignored | `[[tool.uv.index]]` pytorch-cpu + `[tool.uv.sources]`; re-lock; commit or ignore | P1 |
| D-07 | M | `deploy/hf/README.md`, `README.md:91-95` | Pre-docs/11 product text ("10–120 s", "AUC 0.73"), and three contradictory hosting statements | delete `deploy/hf` and the "Free hosting" paragraph, or rewrite | P1 |
| D-08 | M | `app.py:979-1013`, JS harness | The API dict has no Python test; the JS fixture is hand-written; README's field list omits `ai_*`, `fsd_*`, `decided_by` | `api_payload()` + test + shared JSON fixture (§3.4 item 4) | P1 |
| D-09 | M | `docs/04:253-256,169,172-173` | (= A-10) plus `predict_video`/`evaluate_hr`/`evaluate_detector` signatures lack `config` | doc edit | P1 |
| D-10 | M | `ci.yml:12` | `ruff check src tests` leaves `scripts/` unlinted (16 errors today); no `[tool.ruff]`; ruff unpinned; the website says `ruff check .` | lint scripts, config, pin, fix the site line | P1 |
| D-11 | M | `pyproject.toml:8-10` | `opencv-python-headless` and `opencv-contrib-python` both installed into one `cv2/` (80 shared files); the headless pin is moot (mediapipe forces contrib) | keep one distribution | P1 |
| D-12 | M | `models/faceswap.joblib` | Load-tested but never scored; the 768-d model and the CLIP path run in no test | `crop_scores` on random 768-d vectors; the aigen CI leg | P1 |
| D-13 | L | `tests/test_golden.py:9` | The two permanent skips are a Phase-1 placeholder never recorded | record the golden set or delete the file and the docs/08 §3 promise | P1 |
| D-14 | L | `tests/test_detector.py:646,150` | `assert ... or True`; `lo <= 1.0 and hi <= 1.0` (always true) | real assertions | P1 |
| D-15 | L | `AGENTS.md` §1, §2, §4, §7, §9 | Describes the pulse-first product, "Phase 0→4", the docs/10 backlog as live, `reports/roi_placement.png` (never existed), "reports/ gitignored" (7 force-added), Celeb-DF as a pending gate | one paragraph on what decides today; name docs/11–13; fix names | P1 |
| D-16 | L | `docs/04 §6-7`, `serve.py`, extras | Storage layout and deployment views are stale; `tracking` extra unused; `serving` carries fastapi/uvicorn for a stub | rewrite §6/§7; trim extras; delete the stub | BL |
| D-17 | L | `README.md:29,31-40,62,142,144` | "docs/09 is the task backlog"; "(00–09)"; configs list; Phase-0 checklist in the quick start; Content Credentials "needs —" though `c2pa-python` ships only in `aigen` | move `c2pa-python` to core; fix pointers | P1 |
| D-18 | L | `reports/HUMAN_GATES.md:9,19` | `pip install -e ".[dev]"` (no such extra); `data/raw/ubfc/` | fix | P1 |
| D-19 | L | website `Docs.jsx:157` | FAQ latency predates the face-swap layer | re-measure in phase 3; update | P4 |
| D-20 | L | website repo | No CI; every number hand-copied | a Node test against a JSON exported from the tool | BL |
| D-21 | L | `ci.yml:10` | Python 3.12 only; README recommends 3.11 | matrix 3.11 + 3.12 | P1 |
| D-22 | L | `docs/.directory`, the `.docx` | Tracked desktop stub; the brief is unindexed | `git rm --cached`; index the brief | BL |
| D-23 | L | `README.md:118-121` | "stable link, always the latest build" but the zip is cut by hand | `release.yml` on tag; assert manifest version = tag | BL |
| D-24 | L | `configs/p5_aigen.yaml:33-36` | Comment says inference does not re-encode; `aigen.analyze` does when ffmpeg exists; neither branch tested | fix the comment; tests for `W_NO_FFMPEG` and the re-encode | P1 |
| D-25 | L | `pyproject.toml:10,15` | scipy pinned twice | delete line 15 | P1 |

### M. Author's read

| id | sev | where | finding | fix | phase |
|---|---|---|---|---|---|
| M-01 | H | `faceswap.py:203-248`, `face.py:40-62` | The face finder needs a face ≥ ~12–14% of the frame width (§1.2): the cause of the DFD coverage, the gate-4 "no face" share and the extension result | §3.1 | P1 |
| M-02 | L | `faceswap.py:133-141,166` | On the no-frame-count path counting stops at `max_seconds`, so `duration_s` of a long WebM is capped at ~120 s | report the probed duration (A-16) | P1 |
| M-03 | L | `check_faceswap_reals.py`, `held_out_tables` | FF++ "1/140" vs gate-4 "of 300 judged": two denominators in one doc (= B-01) | | P1 |
| M-04 | note | extension capture | The user's capture was 960×540 of a 1920×1080 screen with the face ~30 px; the tip to go fullscreen hides the panel (C-11) | §3.4 | P1 |

---

## Appendix B — Interfaces this round adds or changes (docs/04 §4 edits go with the code)

```python
# facefind.py (new)
class TwoPassFinder:
    def __init__(self, config) -> None            # yunet model path, detect_width, detect_score, nms, landmark_margin, min_face_px, crop_scale
    def find(self, rgb: np.ndarray) -> Face | None  # largest confirmed face: box (native px), landmarks (478,2), yunet_score, n_faces_seen
# faceswap.py: FaceFinder delegates to TwoPassFinder; FaceCrop gains n_faces_seen; config gains the finder keys
# face.py: track_landmarks(frames, fps, progress=None, finder=None) -> (landmarks, valid) via TwoPassFinder, box tracked across frames
# ingest.py: probe(path) -> Probe(fps, n_frames, duration_s, width, height, trusted: bool)
#            open_video(path, frame_height=..., progress=None, max_seconds=None, region=None)   # additive
# faceswap.py: sample_frames(path, n, max_seconds, region=None)                                    # additive
# aigen.py:    analyze(path, bundle, ..., region=None); normalize_for_inference(path, norm, workdir, region=None)
# app.py:      analyze_video(..., region: tuple[int, int, int, int] | None = None)
#              api_payload(a: PulseAnalysis) -> dict   # the extension's JSON; unit-tested; fixture exported for the JS harness
#              verdict_for: C2PA decisive only if validation_state in {Valid, Trusted}
#              CAVEAT -> caveat_text(bundles) rendered at build_demo time
# scripts/train_faceswap.py: manifest-<dataset> builders; extract --variants v1,v2,v3,v4 --reuse-boxes; embed --augment K
#                            select with repeated folds, LOPSO, the §6.2 rule; evaluate on variants; ship records per-source gate-4 counts
# scripts/check_faceswap_reals.py: --manifest data/manifests/faceswap_gate4a.csv (frozen); per-source k/n in JSON
# scripts/check_facefind.py (new): §6.1 measurements -> reports/facefind.md
# notebooks/kaggle/: K1..K4 kernels + kernel-metadata.json; driven by scripts/kaggle_run.py (push, poll, pull)
```

## Appendix C — Cloud budget (estimates; re-measure in phase 1's smoke run)

| step | count | rate | time |
|---|---|---|---|
| download FF++ c23 + c40 (10 sets × 1000) + DFD 450 (+450 c40) | ~15 GB | mirror-bound (round 3: ~15 min for 16 GB) | ~0.5 h |
| faces (two-pass) on v0 + c40 + DFD + DF40 ~4,400 + ~560 DF40 reenactment test videos + Celeb-DF ~1,900 (+ OpenVid ≤ 3,000 if used, UADFV ~100) | ~15k–20k videos | ~1.5 videos/s on 4 vCPU | 2–3 h |
| variants v1–v4 on FF++ + DFD (train/val/test/external) | 6,450 × 4 re-encodes | ~1.5 s each, 4 in parallel | ~2.7 h |
| crops (16 per video; 320 px for v0, 224 px for variants) | ~700k | included above | ~11 GB |
| embeddings, B/16 and L/14, with K = 2 crop augmentations on train+val | ~720k + 2 × 330k crops | T4: ~600/s (B/16), ~150/s (L/14) | ~0.5 h + ~2 h |
| selection (repeated folds, LOPSO, grid of 36) | CPU | seconds per fit | < 0.5 h |
| stage 2 (if triggered) | ~70k crops × 6 epochs | T4 | ~2 h |

GPU quota used: 3–8 of 30 h per week. Sessions: K1 (CPU or GPU, ≤ 8 h),
K2 (GPU, ≤ 3 h), K3 (CPU, minutes), K4 (GPU, ≤ 3 h). Laptop: gate 4a on
706 clips with the two-pass finder ≈ 25 min (0.74 clips/s processed in
round 3, more clips judged now).
