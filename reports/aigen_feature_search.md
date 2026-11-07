# AI-video detector: descriptor search — 2026-11-01 12:13

Selection on the pool only (macro leave-one-generator-family-out AUC), from cached frame embeddings. Rule fixed in advance (script docstring): adopt only if macro AUC >= shipped + 0.01 and worst family >= shipped - 0.02.

| descriptor | C | dims | macro LOGO AUC | worst family |
|---|---|---|---|---|
| mean | 0.003 | 384 | 0.8127 | 0.531 |
| mean | 0.01 | 384 | 0.8090 | 0.538 |
| mean | 0.1 | 384 | 0.7975 | 0.534 |
| mean+std | 0.003 | 768 | 0.8059 | 0.506 |
| mean+std | 0.01 | 768 | 0.7948 | 0.510 |
| mean+std | 0.1 | 768 | 0.7667 | 0.513 |
| mean+diff | 0.003 | 768 | 0.8067 | 0.505 |
| mean+diff | 0.01 | 768 | 0.7961 | 0.511 |
| mean+diff | 0.1 | 768 | 0.7626 | 0.508 |
| all | 0.003 | 1152 | 0.8028 | 0.509 |
| all | 0.01 | 1152 | 0.7892 | 0.515 |
| all | 0.1 | 1152 | 0.7483 | 0.521 |

Shipped: mean, C=0.01: 0.8090 (worst 0.538). Best: mean, C=0.003: 0.8127 (worst 0.531). **Rule says: keep the shipped model.**
