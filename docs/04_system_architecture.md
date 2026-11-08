# 04 — System Architecture

## 1. Context

```
                    +-------------------+
                    |   Video file      |
                    | (mp4/mov/avi/mkv) |
                    +--------+----------+
                             |
                    +--------v----------+
                    |    TrueBeat      |
                    |    pipeline       |
                    +--------+----------+
                             |
            +----------------+----------------+
            |                                 |
   {label, confidence,               {metrics, plots,
    per-window votes,                  run report}
    hr_bpm, warnings}
```

Single-machine, offline-first. No component calls the network during inference.

## 2. Component architecture

```
+----------+   +-------------+   +--------------+   +-------------------+
|  Ingest  |-->| FaceTracker |-->| ROIExtractor |-->| SignalConditioner |
| ingest.py|   |  face.py    |   |    roi.py    |   |    signal.py      |
+----------+   +-------------+   +--------------+   +---------+---------+
                                                              |
                       +------------+   +---------------------+------------------+
                       | HREstimator|   | ConsistencyAnalyzer |  FeatureBuilder  |
                       |   hr.py    |   |   consistency.py    |   features.py    |
                       +-----+------+   +---------+-----------+   +-------+--------+
                             |                    |                       |
                             +----------+---------+-----------+-----------+
                                        |
                              +---------v---------+
                              |    Classifier     |
                              |     model.py      |
                              | LogReg/SVM (P2)   |
                              | CNN on PPG maps   |
                              |       (P3)        |
                              +---------+---------+
                                        |
                        +---------------+---------------+
                        |                               |
                 +------v------+                 +------v------+
                 |  Evaluator  |                 |  Serve (P4) |
                 |   eval.py   |                 |  app.py     |
                 +-------------+                 +-------------+
```

The Phase-4a web app (`app.py`) sits on top of the *same* pipeline the CLI
uses — it does not reimplement any stage. Its analysis half
(`analyze_video`) is a pure function returning a `PulseAnalysis`
(three-state verdict + per-region heart rate, inter-region agreement,
`face_valid_frac`, SNR, ROI waveforms), and the interface half
(`build_demo`) is Gradio wiring.

Since 2026-11-01 (docs/11 E1) `analyze_video` runs three independent layers
and fuses them: C2PA provenance (`provenance.py`), the whole-frame
AI-video detector (`aigen.py`, clips >= 2 s), and the pulse / face-swap
layer above (clips >= 10 s). Priority: provenance declaring AI generation >
AI-video detector over its threshold > strict face-swap score > "no sign
of AI generation found" when the AI-video detector ran and nothing fired;
otherwise inconclusive, with `layers` saying why per layer. The pulse
itself no longer decides: on web video it is not measurable for real faces
either (docs/11 B2). `PulseAnalysis` gained `ai_*`, `provenance`, `layers`,
`decided_by`, `pulse_snr_db` (dB) and lost the peak/mean `mean_snr`. Since 2026-11-06 (docs/12) a fourth layer, the face-swap detector
(`faceswap.py`, clips >= 2 s with a visible face), sits between the
AI-video detector and the pulse. Its bundle says whether it is decisive
or evidence only (docs/12 §6). When it is decisive and looked at a face,
the pulse check no longer decides; otherwise the pulse check keeps its
old role. `decided_by` is "provenance" | "ai_video" | "face_swap" (the
detector) | "pulse" (the pulse check) | "". The AI-video layer skips
title cards (`aigen.guarded_segments`, `W_GRAPHICS_SKIPPED`,
`E_GRAPHICS_ONLY`). The detector's numbers travel as `fsd_*` fields;
2026-11-07 added `fsd_outside_fpr`, the share of real non-FaceForensics
faces it flagged in gate 4 (from `bundle["gate4"]`), quoted whenever an
evidence-only flag is shown. Also 2026-11-07: `FaceSwapResult.top_frame` /
`top_box` (the whole sampled frame the most suspicious crop came from, and
the face box in it) and the matching `PulseAnalysis.fsd_top_frame` /
`fsd_top_box`; the app shows that frame with the face outlined, after the
analysis only. Images never travel in the JSON API. The installed bundle is evidence only
(docs/12 §9).

Since 2026-11-06 `analyze_video` takes an optional
`progress(fraction, stage)` callback: the web app's progress bar follows
the real stages (provenance, AI-video detector, decoding, face tracking
per frame, pulse), weighted by the clip's frame count. A browser extension can therefore reuse
`analyze_video` — over Gradio's built-in API or a purpose-built one —
without a second implementation.

## 3. Data flow (the pipeline, end to end)

1. **Ingest.** `open_video(path)` → frames `uint8 (T,H,W,3)`, plus `fps`
   (from container — never assumed), `n_frames`, duration. Gate: duration ≥ 10 s,
   fps ≥ 15, else `E_SHORT_VIDEO` / `E_LOW_FPS`. Videos taller than
   `frame_height` (default 480) are downscaled with the aspect ratio preserved
   and never upscaled — a 1920×1080 clip of ~900 frames is 5.8 GB decoded and
   OOM-kills the process on a small machine. `MAX_DECODED_BYTES` (2.5 GB)
   raises `E_UNREADABLE` rather than letting the kernel kill the process.
2. **FaceTracker.** MediaPipe Face Landmarker per frame → `landmarks (T,478,2)`,
   `valid (T,)` bool mask. Lost track → re-detect; if valid < 70% of frames,
   warn `W_UNSTABLE_TRACK`.
3. **ROIExtractor.** Landmark polygons → per-frame mean RGB per ROI →
   `traces (T,R,3)` float. P1–2: R = 3–6 (forehead, cheeks, …). P3: R = 32
   sub-regions tiled over the face (FakeCatcher layout).
4. **SignalConditioner.** Per ROI, per channel: `detrend` → SOS band-pass
   (`butter`, `sosfiltfilt`, zero-phase) → `extract_ppg(method)` with
   method ∈ {green, chrom, pos} → `ppg (T,R)`.
5. **HREstimator** (parallel branch). Window the forehead/mean-face PPG,
   `rfft` → strongest bin in 0.7–3.0 Hz → `bpm = f*60`, plus peak SNR.
6. **ConsistencyAnalyzer** (parallel branch). Sliding windows (default 10 s,
   50% overlap): pairwise magnitude-squared coherence between all ROI pairs
   (`scipy.signal.coherence`), cross-correlation at lag 0.
7. **FeatureBuilder.** Per window → feature vector: coherence mean/std/min,
   lag-0 correlation stats, spectral entropy, dominant-peak sharpness,
   per-ROI SNR. P3 also builds **PPG maps**: for each of the 32 regions, the
   time-domain signal and its power spectrum stacked into one 2-D map —
   the CNN input from the FakeCatcher paper.
8. **Classifier.** P2: sklearn `Pipeline(StandardScaler, LogisticRegression|SVC)`
   trained with `GroupKFold` grouped by subject/source video. P3: PyTorch CNN
   on PPG maps. Inference: each window votes; majority wins.
9. **Evaluator / Serve.** Batch scoring vs. manifests (eval), or single-video
   JSON + demo UI (serve).

## 4. Module interfaces

```python
# ingest.py
def open_video(path: str, frame_height: int | None = DEFAULT_FRAME_HEIGHT) -> VideoStream
    # frames, fps, n_frames, w, h, duration_s
    # frame_height caps the decoded frame height (aspect preserved, never
    # upscaled); None keeps native resolution.
    # 2026-11-06 (additive): optional progress(done, total) every 24 frames.
    # 2026-11-07 (docs/13 A-01, A-02, A-17): frame_height now caps the SHORT side (long side
    #   <= 16/9 of it), so vertical video keeps 480 px of width; additive max_seconds=None (stop
    #   decoding there, by timestamp) and region=None ((x0, y0, x1, y1) source pixels, cropped
    #   first); decoding stops at MAX_DECODED_BYTES instead of refusing; VideoStream gains
    #   truncated and full_duration_s. The no-frame-count path (MediaRecorder WebM) now
    #   downscales and counts bytes like the main path.
def probe(path, max_seconds=None) -> Probe   # 2026-11-07: fps, n_frames, duration_s, width, height,
    # from_metadata; frame timestamps when the metadata cannot be trusted (WebM: -2**63 frames)

# face.py
def track_landmarks(frames: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]
    # -> landmarks (T,478,2) float32, valid (T,) bool
    # 2026-11-06 (additive): optional progress(done, total) every 10 frames.

# roi.py
ROI_DEFS_P1 = {"forehead": [...], "cheek_l": [...], "cheek_r": [...]}  # landmark idx lists
def extract_roi_means(frames, landmarks, valid, roi_defs) -> np.ndarray  # (T,R,3)

# signal.py
def condition(trace: np.ndarray, fps: float, band=(0.7, 3.0)) -> np.ndarray
def extract_ppg(traces: np.ndarray, fps: float, method="pos") -> np.ndarray  # (T,R)

# hr.py
@dataclass
class HRResult: bpm: float; confidence: float; spectrum: np.ndarray; freqs: np.ndarray
def estimate_hr(ppg: np.ndarray, fps: float, band=(0.7, 3.0)) -> HRResult

# consistency.py
def coherence_features(ppg_win: np.ndarray, fps: float) -> dict  # pairwise mscohere stats
def lag0_correlation(ppg_win: np.ndarray) -> dict

# features.py
def window_features(ppg: np.ndarray, fps: float, win_s=10.0, overlap=0.5) -> pd.DataFrame
def build_ppg_maps(ppg: np.ndarray, fps: float) -> np.ndarray  # (R, F, T)  Phase 3

# model.py
def train_classifier(X, y, groups) -> Pipeline  # GroupKFold CV inside
def predict_video(path: str, model, config: dict) -> Prediction  # label, confidence, votes[], hr, warnings
    # config: band, method, frame_height, roi_defs, window_s, overlap (the run's YAML)

# eval.py
def evaluate_hr(manifest, config: dict) -> HRReport      # MAE, RMSE, Pearson r, per-subject table
def evaluate_detector(manifest, model, config: dict) -> DetectionReport
    # accuracy, AUC, precision, recall, FPR on reals, confusion, bootstrap CIs, n;
    # runs validate_manifest first
    # 2026-11-07 (docs/13 D-09): `config` was always a parameter of these three; the doc lacked it.

# hr.py (added 2026-11-01, docs/11 B2; additive)
def pulse_snr_db(ppg: np.ndarray, fps: float, bpm=None, band=(0.7, 3.0)) -> float  # dB

# aigen.py (added 2026-11-01, docs/11 Phase D; additive, whole-frame detector)
def sample_segments(path, config=None) -> tuple[np.ndarray, dict]  # (S, N, H, W, 3) uint8, meta
class Embedder: def embed(frames: np.ndarray) -> np.ndarray      # (N, D) unit-norm
def trajectory_features(emb: np.ndarray) -> dict                  # GEOMETRY_COLUMNS
def analyze(path, bundle, embedder=None, use_cache=False, normalize=True) -> AIGenResult
    # p_ai, score, label, warnings
    # 2026-11-06 (additive): optional progress(fraction, stage).

# provenance.py (added 2026-11-01, docs/11 D2; additive, offline C2PA read)
def read_provenance(path) -> ProvenanceResult  # has_c2pa, ai_declared, source_types, generator

# aigen.py (2026-11-07, docs/13 A-06 + §3.4; additive): analyze(..., region=None) crops the
#   region inside the ffmpeg normalisation (or while decoding without ffmpeg);
#   normalize_for_inference(path, norm, workdir, region=None, max_seconds=120) returns
#   (path, warnings, region_applied); the no-ffmpeg decode stops at config max_seconds (120 s).
#   sample_segments, guarded_segments and segment_embeddings take the same region=None (each
#   decoded frame is cropped; the embedding cache key includes the region).
# facefind.py (2026-11-07, docs/13 §3.1): TwoPassFinder(config).largest_face(rgb) -> (cx, cy, side) | None
#   (the round-3 pass first, YuNet + Landmarker-on-a-crop only when it finds nothing); last_seen,
#   last_source; resolve_yunet_path()
# app.py (2026-11-07, docs/13 §3.5): analyze_video(..., region=None); api_payload(a) -> dict (the
#   extension's JSON, keys app.API_PAYLOAD_KEYS, adds pulse_measurable and warnings);
#   parse_region(text); a second API endpoint analyze_video_region(video, region_json);
#   `trubeat serve --host` (default 127.0.0.1); uploads swept after an hour, the extension's
#   capture deleted once analysed; Content Credentials decide only when their signature
#   validates (Valid/Trusted), otherwise the claim is stated in the detail and the other layers
#   decide.
# bundles.py (2026-11-07, docs/13 D-05): load_checked(path) -> (obj, warnings): joblib.load plus
#   W_SKLEARN_VERSION when the bundle's saved scikit-learn major.minor (its `sklearn_version` key,
#   or the version sklearn reports while unpickling) differs from the installed one; the loaders
#   keep the warnings under bundle["_warnings"] and every analysis using the bundle reports them.
# claims.py (2026-11-07, docs/13 A-07): claims_for(bundle) -> dict | None (the bundle's own
#   `claims`, else models/claims.json by the bundle file's SHA-256); about(rate) -> words;
#   caveat(ai, pulse, fsd, fsd_mode) -> str. app: PulseAnalysis gains claims and caveat (the
#   API's caveat is the one for the models that analysed the clip); model_claims(...) and
#   current_caveat(); verdict texts and the pulse check's strict threshold come from the claims;
#   a bundle found by search that would override the packaged one without claims is refused.
#   2026-11-08 (review): claims_of(bundles) quotes the bundles the layers scored with (each
#   layer returns its bundle; one load per analysis); claims.well_formed() gates every claims
#   dict; a refused override is reported as W_OVERRIDE_REFUSED in the result's warnings.
# aigen.py (2026-11-06, docs/12 item A; additive): the title-card guard
def sample_segments(path, config=None, starts_s=None) -> tuple[np.ndarray, dict]  # explicit starts
def is_graphic(segment, config=None) -> bool          # mostly flat 8x8 blocks
def guarded_segments(path, config=None) -> tuple[np.ndarray, dict]  # skips title cards; E_GRAPHICS_ONLY
def segment_embeddings(path, embedder, config=None, use_cache=True, guard=False)  # guard: inference only,
    # cached under its own key (2026-11-06 review: analyze no longer bypasses the cache)

# faceswap.py (added 2026-11-06, docs/12; additive, face-swap detector)
def sample_frames(path, n, max_seconds, region=None) -> SampledFrames   # n frames, native resolution
    # 2026-11-07: region crops each kept frame (additive); the finder is chosen by config["finder"]:
    # "landmarker" (round 3; bundles without the key) or "twopass" (facefind.TwoPassFinder, docs/13 §3.1)
def face_crops(sampled, finder, config=None, progress=None) -> list[FaceCrop]
class FaceEmbedder: def embed(crops) -> np.ndarray         # stage 1: frozen backbone
class CNNScorer: def score(crops) -> np.ndarray            # stage 2: fine-tuned CNN logits
def analyze(path, bundle, embedder=None, finder=None, progress=None, region=None) -> FaceSwapResult
    # score, threshold, label (swap | no_swap | no_face | unknown), real_percentile,
    # n_faces, top_crop / top_time_s (the most suspicious face), warnings;
    # clips under min_seconds (2 s) -> unknown + E_SHORT_VIDEO (docs/12 §5 B)
def cnn_weights_path(bundle) -> Path | None  # stage-2 weights whose SHA-256 matches the bundle:
    # next to the bundle, then models/, data/models/, the package, ~/.cache/trubeat
def fetch_cnn_weights(bundle) -> Path        # `trubeat fetch-models`: download + SHA-256 check
# bundle["mode"] (decisive | evidence) is written only by `scripts/train_faceswap.py ship`,
# which applies docs/12 §6 to gates 1-3 (Colab) and gate 4 (scripts/check_faceswap_reals.py)
# and records the gate-4 result as bundle["gate4"] = {k, n, rate, ci, passed, bundle_sha256, ...}
```

## 5. Key design numbers

- **Frequency resolution:** a 10 s window at 30 fps = 300 samples → 0.1 Hz bins →
  **6 bpm steps**. A 30 s window → ~0.033 Hz → **2 bpm steps**. Use 10 s sliding
  windows for detection votes, 30 s windows when reporting a heart-rate number.
- **Bands:** 0.7–3.0 Hz = 42–180 bpm (HR work); 0.7–14 Hz for the detector —
  FakeCatcher deliberately keeps the above-pulse residual because it carries
  fake-vs-real signal.
- **Windowing:** 10 s windows, 50% overlap → a 60 s clip yields 11 votes.
- **Splits:** `GroupKFold` on `split_group` (= subject for UBFC, source-video for
  FF++ real/fake pairs). A manifest validator asserts no group crosses the split.

## 6. Storage layout

```
trubeat/
  src/trubeat/{ingest,face,roi,signal,hr,consistency,features,model,eval,serve}.py
  configs/{p1_hr.yaml, p2_detector.yaml, p3_cnn.yaml}
  data/
    raw/{ubfc_rppg, ffpp_c23, celebdf_v2}/   # DVC-tracked, never in git
    processed/traces/<dataset>/              # cached (T,R,3) traces, content-hashed
    manifests/*.csv
  runs/            # MLflow file store
  reports/         # eval reports (md + png)
  tests/
```

## 7. Deployment views

- **Phases 1–3 (dev):** laptop, `uv run trubeat …`, MLflow local, DVC local.
- **Phase 4 (service):** Docker image (`python:3.11-slim` + `libgl1` for OpenCV,
  model weights baked in) → `uvicorn serve:app`; Gradio demo on the same host.
  Still offline-capable: no external calls in the request path.

## 8. Error taxonomy

| Code | Meaning | Behavior |
|------|---------|----------|
| `E_UNREADABLE` | file won't decode | abort with message |
| `E_LOW_FPS` | fps missing or < 15 | abort — bpm math is invalid |
| `E_SHORT_VIDEO` | < 10 s usable | abort — frequency resolution meaningless |
| `E_NO_FACE` | no frame has a face (`model.predict_video`); no ROI yields a heart rate: face never tracked or all-NaN ROI traces (`eval.evaluate_hr`) | abort |
| `W_UNSTABLE_TRACK` | intermittent tracking | continue, flag in output |
| `E_MODEL_MISSING` | AI-video backbone weights not in the local cache | that layer is skipped with the message; `trubeat fetch-models` downloads them (the only network step, user-initiated) |
| `W_LOW_CONFIDENCE` | reserved; not emitted by any module (2026-11-07 audit) | — |

Warnings added with the fused verdict (docs/11). Each is shown in the app and extension
evidence and never changes a verdict by itself:

| Code | Meaning |
|------|---------|
| `W_PARTIAL_COVERAGE` | the AI-video layer scored only part of the clip (the shipped bundle: its first 2 s segment) |
| `W_SHORT_CLIP` | fewer segments fit than the AI-video bundle was evaluated with |
| `W_NO_AIGEN_MODEL` / `W_AIGEN_FAILED` | no AI-video bundle installed / that layer raised; the other layers still run |
| `W_NO_FFMPEG` / `W_NORMALIZE_FAILED` | the clip was scored as-is instead of in the training format |
| `W_C2PA_UNVERIFIED` | Content Credentials present but their signature state is not Valid/Trusted |
| `W_NO_C2PA_LIB` / `W_C2PA_READ` | provenance reader missing / failed; treated as "no credentials" |
| `W_PULSE_SKIPPED` / `W_PULSE_FAILED` / `W_FACESWAP_FAILED` | pulse or face-swap layer did not run (e.g. `E_SHORT_VIDEO` under 10 s) or raised |
| `W_UNSTABLE_TRACK` | the pulse layer tracked the face in fewer than 70% of frames; its face-swap check does not run (emitted by the app since 2026-11-07) |
| `W_TRUNCATED` | decoding stopped before the end of the clip, and the result covers only its first part: in the app the pulse layer (120 s, or less when the decoded-frame budget ran out); in `predict`, `hr`, eval and `cache_traces` the decoded-frame budget (`ingest.truncation_warning`, since 2026-11-08); says how much |
| `W_OVERRIDE_REFUSED` | a bundle in a repo override folder (`models/`, `data/models/`) records no measured numbers (`claims`, docs/13 A-07), so the packaged model scored instead; pass it explicitly to use it (emitted since 2026-11-08) |
| `W_SKLEARN_VERSION` | a model bundle was saved with another scikit-learn major.minor than the installed one; its scores may differ from the evaluated ones (emitted since 2026-11-07, docs/13 D-05) |
| `W_GRAPHICS_SKIPPED` | the AI-video layer skipped title-card / graphic segments and scored a later one (docs/12 item A) |
| `E_GRAPHICS_ONLY` | every candidate segment in the first 30 s was a graphic; the AI-video layer did not score |
| `W_FEW_FACES` | the face-swap detector found a face in fewer than 4 of its 16 sampled frames; it does not decide |
| `W_FACESWAP_DETECTOR_FAILED` | the face-swap detector raised; the other layers still run |

## 9. Design principles

1. **Config-driven, not hard-coded.** Bands, windows, ROI sets, methods all live
   in YAML; every report records the config hash.
2. **One stage, one module, one test file.** Signal math gets property tests
   (e.g., a synthetic 72 bpm sine must read 72 ± resolution).
3. **Cache the expensive step.** RGB trace extraction dominates runtime; cache
   by content hash and never recompute silently.
4. **Honest outputs.** Every prediction carries confidence and warnings; every
   evaluation carries its split protocol. No bare accuracy numbers.
