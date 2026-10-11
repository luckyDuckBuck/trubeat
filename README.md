# TrueBeat (`trubeat`)

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

Scaffold + docs complete. Pipeline modules are typed stubs raising
`NotImplementedError`, sequenced by `docs/09` Track A for autonomous
implementation. Human gates (env, dataset forms/downloads, ROI glance,
gate reviews) are collected in `reports/HUMAN_GATES.md` once Track A lands.

## License

MIT — see [LICENSE](LICENSE).

## Contact

Ayushman Tripathi — tripathiayushman@example.com
