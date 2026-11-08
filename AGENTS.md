# AGENTS.md — TrueBeat agent operating manual

You are the coding agent building TrueBeat. This file is your law. The build
spec lives in `docs/`; this file tells you *how* to work.

## 1. What TrueBeat is

A from-scratch rebuild of FakeCatcher (Ciftci, Demir & Yin, 2020): read the
human pulse from ordinary video via remote photoplethysmography (rPPG), check
that facial blood-flow signals are consistent across face regions, and flag
videos that fail. Python, laptop-first, free tooling only.

That is where the project started, not what decides today. `app.verdict_for`
takes the first layer that fires: Content Credentials (C2PA) whose signature
validates; the whole-frame AI-video detector ("Likely AI-generated"); a
face-swap detector whose bundle is decisive (the shipped detector v2 of
round 4 is evidence only: shown, never deciding); else the pulse-consistency check at
a strict threshold ("Possible face-swap", weak). If the AI-video detector
ran and nothing fired, the headline is "No sign of AI generation found",
never a claim that the video is real or authentic.
Heart rate and pulse SNR are evidence only. Plans and measured results since
the first build: `docs/11`, `docs/12`, and `docs/13` (round 4; result in §10).

## 2. First-read order (every session)

1. `docs/00_index.md` — the doc map.
2. The current plan, today `docs/13_faceswap_v2_and_audit_plan.md` (its §0
   is the hand-off brief, §7 the order of work). Work top-down.
3. The doc relevant to your current task (interfaces and warning codes →
   `docs/04` §4 and §8; data → `docs/05`; the rounds the plan builds on →
   `docs/11`, `docs/12`).

`docs/07` (Phases 0 → 4) and the `docs/10` backlog record the first build,
which is over: read them as history, not as the task list.

If this file and `docs/` ever conflict on *what* to build, `docs/` wins.
If they conflict on *how* to work, this file wins.

## 3. Autonomy

You have full access: install packages (uv/pip), run code and tests, create /
edit / delete files in this repo, clone public repos, use the network for
public resources, commit locally on your branch.

Never without explicit human approval: create accounts, submit forms, upload
or share data (especially faces/videos) to any external service, spend money,
push to `main`, delete `data/raw`.

## 4. Build order (the law)

The current plan's phases, in its order (today `docs/13` §7, phases 0 → 6).
Never start a phase until its gate is met or the human waives it in writing.
The plan's order of work is the task list — take the topmost unfinished
item, finish it, commit, repeat. The first build ran Phase 0 → 1 → 2 → 3 → 4
in `docs/07` order from the `docs/10` backlog: Phases 1 and 2 failed their
gates honestly, Phase 3 was not pursued, Phase 4 built the app and the
extension (README, "Status").

## 5. Coding rules

- Typed Python. One module per pipeline stage (`src/trubeat/`). The interfaces
  in `docs/04` §4 are frozen — change one only with a doc edit and a note in
  the commit message.
- Config-driven: bands, windows, ROI sets, methods live in `configs/*.yaml`.
  No magic numbers in code. Every run records its config hash.
- Errors: use the taxonomy in `src/trubeat/errors.py`. Never return silent NaN —
  raise `E_*`, or append to `warnings` and continue for `W_*`.
- Tests for every module. Synthetic-signal property tests FIRST
  (e.g. 72 bpm sine in → 72 ± frequency resolution out), real data second.
- Trace extraction is the expensive step: cache by content hash
  (video bytes + ROI def + method) under `data/processed/traces/`, never
  recompute silently.

## 6. Honesty rules (non-negotiable)

- Every prediction carries confidence + warnings. Every evaluation carries
  dataset + split protocol + grouping key.
- Report metrics with bootstrap confidence intervals. Show the full ROC;
  never a single cherry-picked accuracy.
- Never quote "96%" as our number. Never claim a metric without naming the
  dataset and split next to it.
- The manifest validator (`eval.validate_manifest`) runs before every training
  job and fails the run on identity leakage. No exceptions.

## 7. Data rules

- `data/raw/`, `data/processed/`, `runs/`, `reports/` are gitignored and stay
  that way. The text reports the docs cite are force-added one by one
  (`git add -f`; `git ls-files reports` lists them): only reports with no
  faces or images, no names or paths of the user's own videos and no
  dataset download links. Raw face videos are never committed, never
  uploaded.
- Faces are biometric data. Local-only by default; the inference path makes
  zero network calls.

## 8. Git hygiene

- One commit per finished backlog item. Conventional messages:
  `feat:`, `fix:`, `test:`, `docs:`, `chore:`.
- 2026-10-18 (human-approved): push directly to `origin/main` (= upstream, in the luckyDuckBuck organization since 2026-11-06:
  `luckyDuckBuck/trubeat`). No fork round-trip, no PRs for routine work.
  Conventional, small, honest commits — one per finished task. Phase-gate
  reports still go through human review (§9) before the next phase starts.
- 2026-11-06 (human-requested): the website (`tripathiayushman/trubeat-website`,
  cloned at `../trubeat-website`, deployed to trubeat.vercel.app by Vercel
  on push) describes this tool. A major user-facing change (a new check, a
  new way in, new measured results, changed install steps) gets a site
  update in the same push cycle: one entry on its Updates page plus any
  page the change makes wrong. Steps: that repo's README, "When the tool
  changes". Routine fixes do not. Push the tool first, so the site never
  describes code that is not public yet.

## 9. Human gates (only the human does these — prepare each as a single action)

1. Creating the Python env and installing deps.
2. Submitting dataset access forms (done so far: FaceForensics++ with
   DeepFakeDetection, DF40, Celeb-DF v2; `docs/13` §5).
3. Downloading datasets into `data/raw/`.
4. Glancing at the ROI check images (`reports/roi_check_subject01.png`,
   `roi_check_subject15.png`, `roi_check_subject38.png`; local only, they
   show faces) to confirm ROI polygons. Passed 2026-10-18
   (`reports/HUMAN_GATES.md` B4).
5. Reviewing phase-gate reports and pasting results back.

Each plan lists its own gates as single actions (`docs/11` §6, `docs/12` §2,
`docs/13` §9); `reports/HUMAN_GATES.md` is the first build's checklist.

Everything else is yours. The initial direction: do every task that needs no
human, in the current plan's order, until you hit a gate — then present the
gate checklist and stop.

## 10. When blocked

Write `reports/BLOCKERS.md` (what / what you tried / what's needed to unblock),
commit it (`git add -f`: `reports/` is gitignored), and continue with the next
independent item of the plan. Never spin on a blocked task. Never fake a
result to clear a gate.
