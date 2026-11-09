
```
████████╗██████╗ ██╗   ██╗███████╗██████╗ ███████╗ █████╗ ████████╗
╚══██╔══╝██╔══██╗██║   ██║██╔════╝██╔══██╗██╔════╝██╔══██╗╚══██╔══╝
   ██║   ██████╔╝██║   ██║█████╗  ██████╔╝█████╗  ███████║   ██║   
   ██║   ██╔══██╗██║   ██║██╔══╝  ██╔══██╗██╔══╝  ██╔══██║   ██║   
   ██║   ██║  ██║╚██████╔╝███████╗██████╔╝███████╗██║  ██║   ██║   
   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═════╝ ╚══════╝╚═╝  ╚═╝   ╚═╝  
  
```

A from-scratch rebuild of **FakeCatcher** (Ciftci, Demir & Yin, 2020): read the
human pulse from ordinary video via remote photoplethysmography (rPPG), check
that facial blood-flow signals are consistent across face regions, and flag
videos that fail.

> A detectable pulse is **not** proof a video is real — modern deepfakes inherit
> the driver's pulse. TrueBeat targets *local spatiotemporal consistency* of
> blood flow, the signal the 2025 literature says survives. See `docs/`.

Today a verdict comes from Content Credentials and a whole-frame AI-video
detector first; a weak pulse-consistency check can still flag a possible
face-swap, while a face-swap detector, the heart rate and the pulse SNR are
shown as evidence only. Details under *Run the app*.

## Docs (read first)

[`docs/`](docs/00_index.md) holds the pre-implementation pack (00–10:
charter, requirements, tech stack, architecture, data strategy, risks,
implementation plan, acceptance tests, data access, the first agent
backlog) and one plan per round since, each with its measured results:
[docs/11](docs/11_audit_and_improvement_plan.md) (the AI-video detector and
the fused verdict), [docs/12](docs/12_faceswap_detector_plan.md) (round 3:
the face-swap detector, evidence only) and
[docs/13](docs/13_faceswap_v2_and_audit_plan.md) (round 4, in progress).

**Agents start here:** [`AGENTS.md`](AGENTS.md) is the operating manual
(OpenCode reads it natively); the current plan is docs/13.

## Quick start

```bash
uv venv --python 3.11 && source .venv/bin/activate   # 3.12 also fine
uv pip install -e . pytest   # the package and the test runner
pytest                       # no dataset needed; tests that need model weights or Node skip
```

With uv's project workflow, `uv sync` then `uv run pytest` does the same;
`uv sync` also installs the `dev` group (pytest and the exact ruff CI lints
with).

That is the core pipeline. The web app and the AI-video and face-swap
detectors need the extras below. Datasets are needed only to retrain or
re-measure the detectors (docs/05; access forms are human gates,
AGENTS.md §9).

## Run the app (Phase 4a)

A one-click web interface over the *same* pipeline the CLI runs.

```bash
uv pip install -e ".[serving]"      # gradio (optional extra)
# AI-video and face-swap detectors (CPU only; install torch from the CPU
# index first, or pip pulls ~3 GB of CUDA wheels):
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install -e ".[aigen]"
trubeat fetch-models                # one-time weight download (~85 MB DINOv2-S + ~570 MB CLIP
                                    # for the face-swap detector): the ONLY network step
trubeat serve                       # http://127.0.0.1:7860
```

Upload a video (**2–120 s**). Each clip goes through independent checks,
and the verdict says which one decided (docs/11 §4, docs/12, docs/13):

| check | what it looks at | needs |
|-------|------------------|-------|
| **Content Credentials** | a C2PA manifest declaring AI generation (offline read; re-encoding usually strips it). It decides only when the manifest's signature validates; an unverified claim is shown and the other checks decide | — (the reader, `c2pa-python`, installs with the core package) |
| **AI-video detector** | the whole frame over time: frozen DINOv2 features, how the clip moves through them (`src/trubeat/aigen.py`) | ≥ 2 s |
| **face-swap detector** (evidence only) | face crops from 16 frames, frozen CLIP ViT-B/16 features, a linear classifier trained on FaceForensics++ and DF40 (`src/trubeat/faceswap.py`). Its score is shown but **never decides**: it flagged 16% of real faces from other sources (docs/13 §10) | ≥ 2 s, visible face |
| **pulse face-swap check** | rPPG consistency across face regions, at a strict threshold (weak: on FaceForensics++ c23 test clips it catches about a quarter of Deepfakes swaps and flags 3 of 135 reals; docs/11 §9.1, §9.6) | ≥ 10 s, visible face |

| verdict | meaning |
|---------|---------|
| **Likely AI-generated** / **AI-generated: declared by its Content Credentials** / **Possible face-swap** | a check fired; the detail says which, with its score and threshold |
| **No sign of AI generation found** | the AI-video detector ran and nothing fired. *Not* proof the video is real |
| **Inconclusive** | no check could decide; the detail lists why for each one |

Measured accuracy, including false alarms on real videos, is in
`reports/aigen_detector_v2.md` (written by `scripts/train_aigen.py`) and
summarised in docs/11 §9. **Score it on your own videos:** put clips you know
the answer for in `data/raw/user_eval/ai/` and `data/raw/user_eval/real/`,
then run `python scripts/eval_user_set.py`.

From the command line: `trubeat aigen clip.mp4` (provenance + AI-video
detector, JSON).

The pulse rate itself no longer decides anything. With the standard pulse
SNR, real faces in web-quality video read like noise (docs/11, Phase B
result), so heart rate and pulse SNR are shown as evidence only.

**Privacy.** The local app is **fully private** — videos are analysed on this
machine and never leave it. `trubeat serve` listens on 127.0.0.1 only;
`--host 0.0.0.0` lets other devices on your network upload to it (it prints
a warning). Uploads stay in Gradio's temporary folder for about an hour at
most, and the extension's captures are deleted about two minutes after their
analysis returns (time for the extension to ask again if the connection
drops). `--share` publishes a temporary Gradio link:
**uploads to a shared instance leave the machine**, so use it for demos only,
never for real faces.

**No hosted version.** Face video is biometric data, so TrueBeat runs where
the video is: there is no public instance of the app.

**Extension hook.** The analysis is a pure function, separate from the UI:

```python
from trubeat.app import analyze_video
analysis = analyze_video("clip.mp4")   # -> PulseAnalysis (verdict + evidence)
```

`analyze_video` is what the UI calls, and the browser extension below calls
the same function through Gradio's built-in API (`app.api_payload` builds
its JSON), so there is no second implementation.

## Browser extension (Phase 4b)

A Chrome extension (Manifest V3, vanilla JS, no build step) in
[`extension/`](extension/README.md): one click checks the video playing in
the current tab and overlays the verdict on the page.

```bash
# load it: chrome://extensions → Developer mode → Load unpacked → extension/
```

Prefer a download? Grab the packaged zip from the releases page — the
stable link, always the latest build:

<https://github.com/luckyDuckBuck/trubeat/releases/latest/download/trubeat-extension.zip>

It captures ~20 s of the tab, uploads it to **your** backend, and renders
the three-state verdict with the evidence numbers. It never computes a
verdict itself and never recalibrates a threshold — it calls the same
`analyze_video()` the web app uses, over the Gradio API. No telemetry, no
storage, no network traffic to anything but the backend URL you set.

## Website

The website (marketing pages, documentation, research summary, a
run-it-locally guide and an Updates log) lives in its own repo,
[tripathiayushman/trubeat-website](https://github.com/tripathiayushman/trubeat-website)
and is served at <https://trubeat.vercel.app>. Nothing about the tool
changed with the split: this repo is the product; the website is its
front door.

## Layout

```
src/trubeat/   # one module per pipeline stage (interfaces: docs/04 §4);
               # models/ holds the shipped models (their terms: NOTICE)
configs/       # every run is config-driven: p1_hr* (heart rate), p2_detector
               # (pulse check), p3_cnn (Phase 3, not pursued), p5_aigen
               # (AI-video detector), p6_faceswap / p7_faceswap_v2 (face-swap
               # detector, rounds 3 and 4)
scripts/       # training, evaluation and the report writers
notebooks/     # cloud training: Colab (round 3), Kaggle kernels (round 4)
extension/     # the Chrome extension
data/          # DVC-tracked; raw faces NEVER committed (see .gitignore)
docs/          # the pre-implementation pack (00–10) and one plan per round (11–13)
tests/         # unit, synthetic-signal property and integration tests;
               # tests/js for the extension
third_party/   # reference clones only (rPPG-Toolbox) — never dependencies
```

## Status

Phase 1 (heart rate) and Phase 2 (classical detector) are implemented and
**both gates failed honestly** — see
[`reports/g2_detector.md`](reports/g2_detector.md) for the central write-up
(the Phase-2 result: an rPPG-consistency detector that clears the threshold
on the Deepfakes family and is structurally blind to the reenactment
families, because pulse inheritance leaves the performer's pulse intact).

Phase 4a: the Gradio web app (`trubeat serve`) is in place over the same
pipeline. Phase 3 (CNN on PPG maps) is **not pursued** — the reason is
recorded in `docs/07`.

**2026-11-01 (docs/11): the app now leads with a whole-frame AI-video
detector**, because the pulse could not decide on web video (on FF++ c23
real clips the pulse SNR is below 0 dB in 100% of cases). On held-out
GenBuster-Bench clips, never used for training, selection or the
threshold, the shipped model (the docs/11 §9.7 retrain with 1,492 Rapidata
clips of current generators added) flags 5.4% [4.2, 7.0] of real clips and
0/135 FF++ real test clips. It catches 83% [81, 85] of 2024-era commercial
generators (Sora, Kling, Gen-3, Luma, Pika), 55% [47, 62] of the newest
(Veo 3.1, Sora 2, Kling 2.6) and 52% [49, 55] of open-source ones. Clips
under 2 s are not checked. A hard re-encode (240p, CRF 35) lowers the
newest-generator catch rate from 55% to 35% without adding false alarms;
tab capture and resolution changes barely move it (docs/11 §9.5). It does
not detect face-swaps. The round-2 audit and the next steps are in
`docs/11` §9.

**2026-11-07 (docs/12): a face-swap detector, shown as evidence only.** A
face-crop detector (frozen CLIP ViT-B/16 + logistic regression, chosen by
leave-one-manipulation-out AUC) was trained in Colab on FaceForensics++ c23
train + val. On the held-out FaceForensics++ test set its AUC is 0.97
(Deepfakes), 0.95 (FaceSwap) and 0.93 (FaceShifter) with 1 of 134 judged
real clips flagged; on DeepFakeDetection, AUC 0.88 with 3 of 67 judged reals
flagged (rates over clips where a face was found; docs/12 §9 correction). But on
real non-FaceForensics clips with a face (GenBuster-Bench + AIGVDBench) it
flagged **115/300 = 38%** [33, 44], against a 3% limit fixed before the run,
so by that rule it ships as evidence only and never decides. Re-compression
alone sets it off: after a WhatsApp-like re-encode (640 px, CRF 30) it
flags 45 of 133 judged real FaceForensics++ test clips, against 1 of 134
before. It does
not flag the RepresentUs face-swap ad that started this round. It was
trained on FaceForensics++, whose terms allow non-commercial research use
only (NOTICE).

**2026-11-08 (docs/13): face-swap detector v2, still evidence only.**
Round 4 retrained the face-swap detector on two sources:

- FaceForensics++ c23 and c40, with every clip also re-encoded four ways,
  so re-compression carries no label;
- DF40's fakes from eight newer face-swap methods.

It uses a two-pass face finder (YuNet, then the Face Landmarker on a crop),
which finds the small faces round 3 missed (`reports/facefind.md`).

On held-out data:

| set | AUC | real clips flagged |
|---|---|---|
| FaceForensics++ test | 0.97 (Deepfakes), 0.94 (FaceSwap), 0.92 (FaceShifter) | 1 of 140 judged |
| the same, after a WhatsApp-like re-encode | | 3 of 140 (round 3: 45 of 133) |
| DF40 face-swap methods on FaceForensics++ test identities | mean 0.93 | |
| DeepFakeDetection | 0.82 | 7 of 150 |

It still fails three rules fixed in advance:

- Celeb-DF v2 test-list reals: 20 of 178 flagged (limit 8%);
- other Celeb-DF v2 reals: 63 of 711 flagged (limit 5%);
- the frozen list of real non-FaceForensics clips with a face: **61/383 =
  16%** [13, 20] flagged (limit 3%).

That last rate is less than half of round 3's 38% [33, 44] on the same
list, so v2 replaced the round-3 model. It stays evidence only and never
decides. It does not flag the RepresentUs ad either (score +2.71,
threshold +3.20). A whole analysis of a 15 s 1080p clip takes about 20 s
on the 2020 laptop it was built on.

This round also brought the audit fixes (local-only by default; Content
Credentials decide only when their signature validates) and the
extension's v1.3.0 capture: the tab at its full resolution, up to
1920×1080, with only the playing video's area analysed when it stays put.
It was released as v1.3.0 on 2026-11-09
([releases](https://github.com/luckyDuckBuck/trubeat/releases/tag/extension-v1.3.0)). Results: `docs/13` §10.

Human gates of the first build (env, dataset forms/downloads, ROI glance,
gate reviews) are collected in `reports/HUMAN_GATES.md`; each later round
lists its own (docs/11 §6, docs/12 §2, docs/13 §9).

## Datasets and credits

TrueBeat is built on public research datasets. Thank you to their authors.
No video from any of them is redistributed here. The full terms are in
[NOTICE](NOTICE), section 4.

**Trained on** (the shipped models):

| dataset | used for | terms | credit |
|---|---|---|---|
| [FaceForensics++](https://github.com/ondyari/FaceForensics), with the DeepFakes Detection Dataset by Google & Jigsaw | the pulse check (`detector.joblib`), the face-swap detector (`faceswap.joblib`) and the AI-video detector's real clips (`aigen.joblib`); DeepFakeDetection is a test set only | non-commercial research and educational use | A. Rössler, D. Cozzolino, L. Verdoliva, C. Riess, J. Thies, M. Nießner, "FaceForensics++: Learning to Detect Manipulated Facial Images", ICCV 2019 ([arXiv:1901.08971](https://arxiv.org/abs/1901.08971)); N. Dufour, A. Gully et al., "DeepFakes Detection Dataset", Google & Jigsaw, 2019 |
| [DF40](https://github.com/YZY-stack/DF40) | the face-swap detector's fakes (eight face-swap methods); its test sets measure it | CC BY-NC 4.0 | Z. Yan et al., "DF40: Toward Next-Generation Deepfake Detection", NeurIPS 2024 Datasets and Benchmarks Track ([arXiv:2406.13495](https://arxiv.org/abs/2406.13495)) |
| [AIGVDBench](https://huggingface.co/datasets/AIGVDBench/AIGVDBench) | the AI-video detector's training clips | CC BY 4.0 | L. Ma, Z. Xue et al., "Your One-Stop Solution for AI-Generated Video Detection", CVPR 2026 ([arXiv:2601.11035](https://arxiv.org/abs/2601.11035)) |
| [OpenVid-1M](https://huggingface.co/datasets/nkp37/OpenVid-1M) | the real clips inside AIGVDBench | CC BY 4.0; its card says it is for research and non-commercial purposes | K. Nan et al., "OpenVid-1M: A Large-Scale High-Quality Dataset for Text-to-video Generation", ICLR 2025 ([arXiv:2407.02371](https://arxiv.org/abs/2407.02371)) |
| [Rapidata text-to-video human-preference sets](https://huggingface.co/datasets/Rapidata/text-2-video-human-preferences) | 1,492 clips from 14 current generators (Veo, Seedance, Marey, Ray 2, Gen-3, Sora, Pika, HunyuanVideo, Wan, Mochi) for the AI-video detector | Apache-2.0 | [Rapidata](https://rapidata.ai) |

**Tested on** (measurements only; nothing shipped is trained on them):

| dataset | used for | terms | credit |
|---|---|---|---|
| [GenBuster-Bench](https://huggingface.co/datasets/l8cv/GenBuster-Bench) | the AI-video detector's held-out test, and real clips for the face-swap detector's false-alarm test | MIT | H. Wen et al., "BusterX: MLLM-Powered AI-Generated Video Forgery Detection and Explanation", 2025 ([arXiv:2505.12620](https://arxiv.org/abs/2505.12620)) |
| [Celeb-DF v2](https://github.com/yuezunli/celeb-deepfakeforensics) | the face-swap detector's cross-dataset test | research use, under its access form | Y. Li, X. Yang, P. Sun, H. Qi, S. Lyu, "Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics", CVPR 2020 ([arXiv:1909.12962](https://arxiv.org/abs/1909.12962)) |
| UADFV (in DF40's test set) | a report-only face-swap test | research use | X. Yang, Y. Li, S. Lyu, "Exposing Deep Fakes Using Inconsistent Head Poses", ICASSP 2019 ([arXiv:1811.00661](https://arxiv.org/abs/1811.00661)) |
| [UBFC-rPPG](https://sites.google.com/view/ybenezeth/ubfcrppg) | the heart-rate check | its authors' terms (subjects 21 and 27 may not be shown) | S. Bobbia, R. Macwan, Y. Benezeth, A. Mansouri, J. Dubois, "Unsupervised skin tissue segmentation for remote photoplethysmography", Pattern Recognition Letters 124, 2019 |

The models build on DINOv2 (Meta AI), CLIP ViT-B/16 (OpenAI), the MediaPipe
Face Landmarker (Google) and YuNet (OpenCV Zoo). NOTICE sections 2 and 3
list their licences.

## License

The code is MIT: see [LICENSE](LICENSE). The trained models in
`src/trubeat/models/` are not covered by it: `detector.joblib`,
`aigen.joblib` and `faceswap.joblib` were trained with FaceForensics++ data,
whose terms allow non-commercial research and educational use only, and
`faceswap.joblib` also with DF40 fakes (CC BY-NC 4.0). So these models are
for non-commercial use, and the app loads them by default.
[NOTICE](NOTICE) lists every model file, the backbone weights
`trubeat fetch-models` downloads, and each dataset's terms and attribution.

## Contact

Ayushman Tripathi — atr.ayushmantripathi@gmail.com
