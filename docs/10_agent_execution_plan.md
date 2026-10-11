# 10 — Agent Execution Plan (autonomous build)

## Objective

Build TrueBeat end-to-end with the agent working autonomously. The human acts
only at the marked gates. Initial direction: **execute every task that needs
no human, in backlog order, before anything else.**

## Context loading (start of every session)

1. Read `AGENTS.md`.
2. Read `docs/00_index.md`, then the docs for the current phase.
3. Read this file, take the topmost unfinished item, do it, commit, repeat.

## Track A — No human needed (do all of this first, in order)

Each item lists its done-criteria. Commit per item (`feat:`/`test:`/`chore:`).

- **A1 — Repo hygiene.** Confirm the `pulsefake → trubeat` rename is complete
  (no stale references), `requires-python >=3.11,<3.13` in `pyproject.toml`,
  `pytest` green. *Done: 2 passed, 2 skipped; `grep -ri pulsefake` clean.*
- **A2 — `ingest.py`.** OpenCV decode, true-fps read, `E_UNREADABLE` /
  `E_LOW_FPS` / `E_SHORT_VIDEO` gates. Tests: write a synthetic video generator
  (`tests/synth.py` — solid-color frames with an embedded sine pulse at a known
  bpm; plus corrupt/short/low-fps fixtures) and assert every error code fires.
  *Done: error taxonomy fully covered by tests.*
- **A3 — `face.py`.** MediaPipe Face Landmarker wrapper returning
  `(T,478,2)` + validity mask, re-detect on lost track. Tests: no-face video →
  `valid` all-False; synthetic landmark arrays → shape/dtype contract.
  *Done: contract tests green.*
- **A4 — `roi.py`.** Landmark-polygon masks → per-frame mean RGB `(T,R,3)`.
  Fill `ROI_DEFS_P1` (forehead, cheeks) and render `reports/roi_placement.png`
  (polygons drawn on a sample frame) for the human's later glance — does not
  block. Tests: synthetic landmarks with known colors → exact means.
  *Done: means exact on synthetic input; placement plot saved.*
- **A5 — `signal.py`.** Detrend + SOS band-pass + `green`/`chrom`/`pos`
  extraction. Property tests: 72 bpm sine → 72 ± frequency resolution;
  CHROM/POS cancel a single-channel illumination ramp (their design guarantee).
  *Done: property tests green.*
- **A6 — `hr.py`.** FFT peak-pick → bpm + peak-SNR confidence. Tests: synthetic
  pulse at 60/72/120 bpm reads back within one frequency bin.
  *Done: all synthetic bpm cases pass.*
- **A7 — `consistency.py` + `features.py`.** Pairwise `mscohere`, lag-0
  correlation, spectral features → per-window feature rows. Tests: coherent
  synthetic ROI sines → high coherence; independent sines → low; feature
  DataFrame schema stable. *Done: separation clear on synthetic data.*
- **A8 — `model.py`.** sklearn `Pipeline(StandardScaler, LogisticRegression |
  SVC)`, `GroupKFold` on `split_group`; `predict_video` → majority vote +
  `Prediction` JSON. Tests: synthetic separable features → AUC 1.0; validator
  fixture. *Done: trains, predicts, votes.*
- **A9 — `eval.py`.** `validate_manifest` (fails on leakage — test with a
  purpose-built bad manifest), HR metrics (MAE/RMSE/Pearson/Bland-Altman),
  detector metrics (acc/AUC/PR/confusion + bootstrap CIs), markdown report
  writer per `docs/08` §5. *Done: validator catches leakage; report renders.*
- **A10 — `cli.py` wiring.** `trubeat hr|train|predict|eval` end-to-end on
  synthetic manifests. *Done: every subcommand runs offline.*
- **A11 — rPPG-Toolbox sidecar.** Clone upstream to `third_party/rppg-toolbox`
  (reference only — never a dependency; check its license before vendoring
  anything). *Done: cloned, license noted.*
- **A12 — Synthetic end-to-end.** Build two synthetic video suites:
  (a) "real": coherent multi-ROI pulse at known bpm; (b) "fake": incoherent
  ROI signals. Run the full pipeline: assert bpm within one bin on (a),
  and the classifier separates (a)/(b). This validates the entire system with
  zero downloaded data. *Done: suite green — the pipeline provably works
  before it ever sees a real face.*
- **A13 — CI.** `.github/workflows/ci.yml`: `pytest` + `ruff` on push/PR.
  *Done: CI configured.*
- **A14 — Human-gate prep.** Write `reports/HUMAN_GATES.md`: the exact
  one-command env setup, the dataset form links, the UBFC download steps,
  the FF++ c23 file list + download script. Each gate must be a single human
  action. *Done: checklist ready.*

**Definition of done for Track A:** `pytest` green, synthetic end-to-end green,
CI configured, `HUMAN_GATES.md` written, everything committed on the branch.

## Track B — Human gates (agent prepares, human acts, agent continues)

- **B1 — Environment.** Human: create env, `pip install -e .`, run `pytest`,
  paste result. (One command, from `HUMAN_GATES.md`.)
- **B2 — Dataset forms.** Human: submit FF++ + Celeb-DF access forms.
- **B3 — UBFC-rPPG.** Human: download into `data/raw/ubfc_rppg/`.
  Agent then: run `trubeat eval` → check **G1** (MAE ≤ 5 bpm), write report.
- **B4 — ROI glance.** Human: open `reports/roi_placement.png`, confirm or
  correct the polygons (one message).
- **B5 — FF++ c23.** Human: download the file list from `HUMAN_GATES.md`.
  Agent then: build manifests, cache traces, train, check **G2**.
- **B6 — Gate reviews.** Human reviews each phase-gate report; agent proceeds
  on approval.

## Operating rules

- Work top-down; one backlog item at a time; commit when its done-criteria hold.
- Blocked? Write `reports/BLOCKERS.md` (what / tried / needed), commit,
  continue with the next independent item. Never fake a gate.
- Phase gates (G1–G4, `docs/07`) are hard stops: no next phase until the gate
  is met or the human waives it in writing.
- Cloud note: bulk FF++ trace extraction and Phase-3 CNN training may move to
  cloud (cheap CPU batch / free Colab GPU) — same code, same Docker image,
  bigger manifest. Decided at B5, not before.
