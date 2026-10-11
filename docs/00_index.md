# PulseFake — Pre-Implementation Document Pack

A FakeCatcher-style deepfake detector built on remote photoplethysmography (rPPG):
read the human pulse from ordinary video, check that facial blood-flow signals are
consistent across regions, and flag videos that fail the test.

Based on Hari's "Deepfake detection from your pulse" brief, with every major claim
re-verified against the cited papers, datasets, and tools (Oct 2026).

## The pack

| # | Document | Answers |
|---|----------|---------|
| 01 | [Project Charter](01_project_charter.md) | What are we building, why, and what does "done" look like? |
| 02 | [System Requirements (SRS)](02_system_requirements.md) | Exact functional + non-functional requirements with IDs |
| 03 | [Tech Stack Recommendation](03_tech_stack.md) | What to use for every layer, why, and what was rejected |
| 04 | [System Architecture](04_system_architecture.md) | Components, data flow, module interfaces, deployment |
| 05 | [Data Strategy](05_data_strategy.md) | Datasets, access, storage, versioning, split protocol |
| 06 | [Risk Register](06_risk_register.md) | Everything known to go wrong, with mitigations |
| 07 | [Implementation Plan](07_implementation_plan.md) | Phases mapped to the brief's build steps, with gates |
| 08 | [Acceptance Test Plan](08_acceptance_test_plan.md) | How you prove it works, honestly |
| 09 | [Data Access Starter](09_data_access_starter.md) | Tonight's 10-minute actions: verified form links, ToS friction, download commands |

## Read order

1. `01` for the why, `03` for the what-to-install, `07` for the build order.
2. Keep `02`, `04`, `05`, `06`, `08` open as reference while implementing.
3. `09` is the do-tonight checklist (dataset access forms + UBFC pull).

## One-line summary of the whole project

Green-channel pulse from 3–6 face patches → classical classifier (weekend project);
32-region PPG maps → CNN classifier (the real FakeCatcher); local spatiotemporal
consistency is the feature that survives 2025-era deepfakes, not the global heart rate.
