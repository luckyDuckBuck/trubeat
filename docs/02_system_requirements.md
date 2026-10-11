# 02 — System Requirements Specification (SRS)

## 1. Functional requirements

### Ingest
- **FR-01** Accept video files: mp4, mov, avi, mkv (H.264/H.265). Reject unreadable
  or zero-length files with a clear error, never a silent empty result.
- **FR-02** Read the true frame rate from container metadata (`CAP_PROP_FPS` /
  stream info). If fps is missing or < 15, abort with `E_LOW_FPS`. Never assume 30.

### Face & regions
- **FR-03** Detect the face and produce 478 facial landmarks per frame (MediaPipe
  Face Landmarker). Output a per-frame validity mask; re-detect when tracking is lost.
- **FR-04** Define named regions of interest (ROIs) from landmark polygons.
  Phase 1–2: forehead, left cheek, right cheek (3–6 patches). Phase 3: 32 sub-regions
  tiled across the face (FakeCatcher layout).
- **FR-05** Extract per-frame mean R, G, B per ROI → trace tensor of shape
  `(T frames, R regions, 3 channels)`.

### Signal processing
- **FR-06** Condition each trace: detrend slow drift (`scipy.signal.detrend`),
  then zero-phase band-pass (`butter` SOS + `sosfiltfilt`). Default bands are
  configurable; shipped defaults: **0.7–3.0 Hz** for heart-rate work,
  **0.7–14 Hz** for fake detection (FakeCatcher keeps the residual "noise"
  above the pulse band because it helps spot fakes).
- **FR-07** Extract the pulse waveform per ROI using a selectable method:
  `green` (baseline), `chrom`, `pos` (primary unsupervised method).
- **FR-08** Estimate heart rate per window: FFT (`numpy.fft.rfft`), strongest bin
  inside the pulse band × 60 → bpm, plus peak SNR as a confidence score.

### Detection
- **FR-09** Compute pairwise magnitude-squared coherence (`scipy.signal.coherence`)
  between ROI pulse signals per analysis window (default 10 s, 50% overlap).
- **FR-10** Build a window-level feature vector: coherence mean/std/min,
  cross-correlation at lag 0, spectral entropy, peak sharpness, per-ROI SNR.
  Phase 3 adds PPG maps: per-region time signal + power spectrum stacked into a
  2-D map (32 regions × time/frequency), the CNN input used by FakeCatcher.
- **FR-11** Train a classifier on labeled windows: logistic regression or SVM
  (Phase 2), CNN on PPG maps (Phase 3). Training must support group-aware
  cross-validation (see NFR-08).
- **FR-12** Predict: every window of a new video votes; majority vote wins.
  Output JSON: `{label, confidence, n_windows, votes[], hr_bpm, warnings[]}`.

### Evaluation & reporting
- **FR-13** Batch-score a dataset manifest: HR metrics (MAE, RMSE, Pearson r,
  SNR) against ground truth; detector metrics (accuracy, AUC, precision,
  recall, confusion matrix).
- **FR-14** Export a run report (markdown + plots): dataset, split protocol,
  method, all metrics with confidence intervals, and the exact config used.
- **FR-15** (Phase 4) Serve predictions over REST: `POST /predict` (video upload)
  → FR-12 JSON. Serve a Gradio demo page for manual uploads.

## 2. Non-functional requirements

| ID | Requirement |
|----|-------------|
| NFR-01 | **Accuracy gates:** G1 HR MAE ≤ 5 bpm (UBFC-rPPG, still clips); G2 detector AUC ≥ 0.70 person-disjoint (0.50+ minimum); G3 cross-dataset report with documented protocol. |
| NFR-02 | **Performance:** a 10 s 720p clip goes from file to feature vector in ≤ 60 s on a 4-core laptop CPU (Phases 1–2). Dataset batches may run overnight. |
| NFR-03 | **Robustness:** no-face frames, short videos, and corrupt files produce explicit error codes (`E_NO_FACE`, `E_SHORT_VIDEO`, …), never silent NaNs. |
| NFR-04 | **Privacy:** local-only by default. The inference path makes zero network calls. Dataset face images are never committed to version control. |
| NFR-05 | **Reproducibility:** seeded RNGs, locked dependencies (`uv.lock`), DVC-tracked data, every run logged to MLflow with its config hash. |
| NFR-06 | **Usability:** `pulsefake --help` documents every command; `pulsefake eval --dataset ubfc` reproduces the headline numbers in one command. |
| NFR-07 | **Maintainability:** typed Python, one module per pipeline stage, pytest coverage on all signal-math functions. |
| NFR-08 | **No identity leakage:** no subject (or FF++ real/fake pair from one source video) may appear on both sides of a train/test split. Enforced by group-aware splitting and a CI check on manifests. |

## 3. Interfaces

- **CLI:** `pulsefake hr <video>`, `pulsefake train --manifest …`, `pulsefake predict <video>`, `pulsefake eval --dataset …`.
- **Python API:** `pulsefake.pipeline.{extract_traces, estimate_hr, predict_video}`.
- **REST (P4):** `POST /predict` multipart video → FR-12 JSON; `GET /health`.
- **Manifest CSV schema:** `video_id, subject_id, source_video_id, label, split_group, dataset, quality, fps, duration_s, path`.
- **Trace cache:** NumPy `.npy` / Parquet keyed by content hash of (video bytes + ROI definition + method).

## 4. Requirement → phase traceability

| Phase | Covers |
|-------|--------|
| 0 | NFR-05, NFR-07 (skeleton, env, tests) |
| 1 | FR-01–FR-08, FR-13 (HR side), NFR-01 (G1) |
| 2 | FR-09–FR-12, FR-13 (detector side), FR-14, NFR-08, NFR-01 (G2) |
| 3 | FR-10 (PPG maps), FR-11 (CNN), NFR-01 (G3) |
| 4 | FR-15, NFR-02, NFR-04, NFR-06, NFR-01 (G4) |

## 5. Open decisions (resolve before Phase 3)

- OD-01: CNN architecture for PPG maps (small custom CNN vs. fine-tuned backbone).
- OD-02: Operating threshold for the real/fake decision (tune on validation; report the full ROC, not a single accuracy).
- OD-03: Whether to add a second modality (e.g., blending-boundary cues) — explicitly deferred; this project is pulse-only by design.
