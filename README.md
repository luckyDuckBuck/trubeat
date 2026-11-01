
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

## Docs (read first)

The full pre-implementation pack lives in [`docs/`](docs/00_index.md):
project charter, system requirements, tech stack, architecture, data strategy,
risk register, implementation plan, acceptance tests, and the
[agent execution plan](docs/10_agent_execution_plan.md).

**Agents start here:** [`AGENTS.md`](AGENTS.md) is the operating manual
(OpenCode reads it natively), then `docs/09` is the task backlog.

## Quick start (Phase 0)

```bash
uv venv --python 3.11 && source .venv/bin/activate   # 3.12 also fine
uv pip install -e .
pytest                    # smoke tests must stay green
```

Then follow the Phase 0 checklist in `docs/07_implementation_plan.md`:
submit the FF++ / Celeb-DF access forms, download UBFC-rPPG.

## Run the app (Phase 4a)

A one-click web interface over the *same* pipeline the CLI runs.

```bash
uv pip install -e ".[serving]"      # gradio (optional extra)
# AI-video detector (CPU only; install torch from the CPU index first,
# or pip pulls ~3 GB of CUDA wheels):
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install -e ".[aigen]"
trubeat fetch-models                # one-time weight download: the ONLY network step
trubeat serve                       # http://127.0.0.1:7860
```

Upload a video (**2–120 s**). Each clip goes through three independent
checks, and the verdict says which one decided (docs/11 §4):

| check | what it looks at | needs |
|-------|------------------|-------|
| **Content Credentials** | a C2PA manifest declaring AI generation (offline read; re-encoding usually strips it) | — |
| **AI-video detector** | the whole frame over time: frozen DINOv2 features, how the clip moves through them (`src/trubeat/aigen.py`) | ≥ 2 s |
| **face-swap check** | rPPG consistency across face regions, at a strict threshold (weak: about a quarter of Deepfakes-style swaps, ~2% of real face videos flagged) | ≥ 10 s, visible face |

| verdict | meaning |
|---------|---------|
| **Likely AI-generated** / **AI-generated: declared by its Content Credentials** / **Possible face-swap** | a check fired; the detail says which, with its score and threshold |
| **No sign of AI generation found** | the AI-video detector ran and nothing fired. *Not* proof the video is real |
| **Inconclusive** | no check could decide; the detail lists why for each one |

Measured accuracy, including false alarms on real videos, is in
`reports/aigen_detector.md` (written by `scripts/train_aigen.py`) and
summarised in docs/11. **Score it on your own videos:** put clips you know
the answer for in `data/raw/user_eval/ai/` and `data/raw/user_eval/real/`,
then run `python scripts/eval_user_set.py`.

From the command line: `trubeat aigen clip.mp4` (provenance + AI-video
detector, JSON).

The pulse itself no longer decides anything. With the standard pulse SNR,
real faces in web-quality video read like noise (docs/11, Phase B result),
so heart rate and pulse SNR are shown as evidence only.

**Privacy.** The local app is **fully private** — videos are analysed on this
machine and never leave it. `--share` publishes a temporary Gradio link:
**uploads to a shared instance leave the machine**, so use it for demos only,
never for real faces.

**Free hosting.** The app is a plain Gradio app, so it also runs on
**Hugging Face Spaces** (free CPU tier) for a public demo — with the same
caveat that anything uploaded to a hosted instance leaves your machine.
Deploy by adding a `spaces.yaml` (or `app.py` entrypoint) that calls
`trubeat.app.build_demo()`.

**Extension hook.** The analysis is a pure function, separate from the UI:

```python
from trubeat.app import analyze_video
analysis = analyze_video("clip.mp4")   # -> PulseAnalysis (verdict + evidence)
```

`analyze_video` is what the UI calls and what a future browser extension
should call — Gradio's built-in API already exposes the same function over
HTTP, so an extension can POST a video without a second implementation.

## Browser extension (Phase 4b)

A Chrome extension (Manifest V3, vanilla JS, no build step) in
[`extension/`](extension/README.md): one click checks the video playing in
the current tab and overlays the verdict on the page.

```bash
# load it: chrome://extensions → Developer mode → Load unpacked → extension/
```

Prefer a download? Grab the packaged zip from the releases page — the
stable link, always the latest build:

<https://github.com/tripathiayushman/trubeat/releases/latest/download/trubeat-extension.zip>

It captures ~20 s of the tab, uploads it to **your** backend, and renders
the three-state verdict with the evidence numbers. It never computes a
verdict itself and never recalibrates a threshold — it calls the same
`analyze_video()` the web app uses, over the Gradio API. No telemetry, no
storage, no network traffic to anything but the backend URL you set.

## Website

The website (marketing pages, documentation, research summary and the
"try it online" demo page) lives in its own repo,
[tripathiayushman/trubeat-website](https://github.com/tripathiayushman/trubeat-website)
and is served at <https://trubeat.vercel.app>. Nothing about the tool
changed with the split: this repo is the product; the website is its
front door.

## Layout

```
src/trubeat/   # one module per pipeline stage (interfaces: docs/04 §4)
configs/       # p1_hr / p2_detector / p3_cnn — every run is config-driven
data/          # DVC-tracked; raw faces NEVER committed (see .gitignore)
docs/          # the full pre-implementation pack (00–09)
tests/         # smoke + golden-set regression + synthetic-signal property tests
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
GenBuster-Bench clips, never used for training or the threshold, it flags
5.7% [4.4, 7.3] of real clips and 0/135 FF++ real test clips. It catches
78% [76, 81] of 2024-era commercial generators (Sora, Kling, Gen-3, Luma,
Pika), 45% [38, 53] of the newest (Veo 3.1, Sora 2, Kling 2.6) and 50%
[47, 53] of open-source ones. It does not detect face-swaps. The round-2
audit and the next steps are in `docs/11` §9.

Human gates (env, dataset forms/downloads, ROI glance, gate reviews) are
collected in `reports/HUMAN_GATES.md`.

## License

MIT — see [LICENSE](LICENSE).

## Contact

Ayushman Tripathi — atr.ayushmantripathi@gmail.com
