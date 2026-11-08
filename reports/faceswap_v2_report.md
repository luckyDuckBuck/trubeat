# Face-swap detector v2 — 2026-11-08 22:35

Protocol: docs/13 §6.2-6.3. Selection on FaceForensics++ (+ DF40 FF-domain) train+val only; every held-out rate is over judged clips (a face in >= 4 of 16 frames) and printed with the total; AUC intervals resample identity groups; rate intervals are Wilson. Config hash `aaaae705`; versions {"torch": "2.15.0+cpu", "timm": "1.0.30", "mediapipe": "0.10.21", "sklearn": "1.9.1", "numpy": "1.26.4", "cv2": "4.9.0", "pandas": "3.0.6", "git_commit": "676d4c392df9fe3cf256ab4fa8ff63151c8c42d5"}.

Sampling: every number in this report was measured on 16 frames sampled within the first 20 s of each clip and processing variant (`window_s`). The app samples within the first 120 s (`max_seconds`, the config hash above); most training and test clips are shorter than 20 s, and gate 4a is measured on the laptop the app's way (docs/13 §6.6 item 7).

Dedupe (docs/13 §3.2): 0 of 2400 DF40 training clips removed (mean v0 face embedding, vit_base_patch16_clip_224.openai, cosine >= 0.97 with one of 5383 held-out clips); 0 more excluded after the laptop's gate-4a overlap check. Nearest-neighbour cosine of the DF40 clips: median 0.842, 99th percentile 0.908, max 0.961.

## Stage A: representation (LOMO macro AUC on the base set, repeated group folds)

| backbone | feature | C | lomo_macro | lomo_sd | Deepfakes | Face2Face | FaceShifter | FaceSwap | NeuralTextures | df40:blendface | df40:e4s | df40:facedancer | df40:faceswap | df40:fsgan | df40:inswap | df40:mobileswap | df40:simswap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vit_base_patch16_clip_224.openai | cls | 0.0100 | 0.8896 | 0.0027 | 0.9449 | 0.8151 | 0.8398 | 0.8367 | 0.6816 | 0.9254 | 0.9755 | 0.8792 | 0.9027 | 0.9581 | 0.9492 | 0.9071 | 0.9495 |
| vit_base_patch16_clip_224.openai | cls+patch | 0.0100 | 0.8872 | 0.0031 | 0.9359 | 0.7997 | 0.8253 | 0.8403 | 0.6839 | 0.9373 | 0.9596 | 0.8672 | 0.9119 | 0.9580 | 0.9418 | 0.9245 | 0.9478 |
| vit_base_patch16_clip_224.openai | cls | 0.1000 | 0.8730 | 0.0046 | 0.9291 | 0.8130 | 0.8173 | 0.8251 | 0.6687 | 0.9068 | 0.9612 | 0.8505 | 0.8900 | 0.9451 | 0.9348 | 0.8795 | 0.9280 |
| vit_base_patch16_clip_224.openai | cls+patch | 0.1000 | 0.8697 | 0.0030 | 0.9178 | 0.7859 | 0.8005 | 0.8294 | 0.6862 | 0.9183 | 0.9362 | 0.8349 | 0.8966 | 0.9415 | 0.9276 | 0.9026 | 0.9280 |
| vit_base_patch16_clip_224.openai | cls | 1.0000 | 0.8557 | 0.0060 | 0.9118 | 0.7981 | 0.7951 | 0.8122 | 0.6501 | 0.8877 | 0.9457 | 0.8285 | 0.8749 | 0.9325 | 0.9177 | 0.8638 | 0.9059 |
| vit_base_patch16_clip_224.openai | cls+patch | 1.0000 | 0.8518 | 0.0027 | 0.8994 | 0.7677 | 0.7758 | 0.8194 | 0.6799 | 0.9007 | 0.9145 | 0.8086 | 0.8782 | 0.9289 | 0.9106 | 0.8812 | 0.9085 |

## Stage B: training set (LOPSO false alarms at the 3% target; LOMO on the training set)

| backbone | feature | C | training_set | lomo_macro | lopso_max | lopso_v0 | lopso_c40 | lopso_v1 | lopso_v2 | lopso_v3 | lopso_v4 | lopso_n_v0 | lopso_n_c40 | lopso_n_v1 | lopso_n_v2 | lopso_n_v3 | lopso_n_v4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vit_base_patch16_clip_224.openai | cls | 0.0100 | base | 0.8902 | 0.4744 | 0.0058 | 0.4744 | 0.0349 | 0.1895 | 0.0267 | 0.0512 | 860 | 860 | 859 | 860 | 860 | 860 |
| vit_base_patch16_clip_224.openai | cls | 0.0100 | video | 0.8748 | 0.2198 | 0.0209 | 0.2198 | 0.0140 | 0.0884 | 0.0256 | 0.0360 | 860 | 860 | 859 | 860 | 860 | 860 |
| vit_base_patch16_clip_224.openai | cls | 0.0100 | video+crop | 0.8765 | 0.2407 | 0.0186 | 0.2407 | 0.0128 | 0.0628 | 0.0209 | 0.0314 | 860 | 860 | 859 | 860 | 860 | 860 |
| vit_base_patch16_clip_224.openai | cls+patch | 0.0100 | base | 0.8887 | 0.3733 | 0.0012 | 0.3733 | 0.0442 | 0.1965 | 0.0267 | 0.0558 | 860 | 860 | 859 | 860 | 860 | 860 |
| vit_base_patch16_clip_224.openai | cls+patch | 0.0100 | video | 0.8614 | 0.1430 | 0.0209 | 0.1430 | 0.0140 | 0.1012 | 0.0233 | 0.0430 | 860 | 860 | 859 | 860 | 860 | 860 |
| vit_base_patch16_clip_224.openai | cls+patch | 0.0100 | video+crop | 0.8638 | 0.1488 | 0.0221 | 0.1488 | 0.0151 | 0.0698 | 0.0233 | 0.0360 | 860 | 860 | 859 | 860 | 860 | 860 |

**Chosen: vit_base_patch16_clip_224.openai, cls+patch, C=0.01, training set `video`** (LOPSO <= 6% **not met**: lowest LOPSO chosen). Threshold +3.1987 (out-of-fold reals of the training set at 3%, sources weighted equally). In-domain out-of-fold AUC on v0 0.898.

## Held-out results

### cdf2 — v0

Reals flagged (test list): **20/178 judged = 11.2% [7.4%, 16.7%] (of 178)**.
Reals flagged (outside the test list): **63/711 judged = 8.9% [7.0%, 11.2%] (of 712)**.

| family | n | judged | errors | AUC vs reals (95% CI, by identity group) | caught (judged, Wilson 95%) |
|---|---|---|---|---|---|
| CDF | 340 | 340 | 0 | 0.778 [0.735, 0.818] | 131/340 = 38.5% [33.5%, 43.8%] |

### df40 — v0


| family | n | judged | errors | AUC vs reals (95% CI, by identity group) | caught (judged, Wilson 95%) |
|---|---|---|---|---|---|
| df40:blendface | 140 | 140 | 0 | 0.916 [0.889, 0.942] | 48/140 = 34.3% [26.9%, 42.5%] |
| df40:facedancer | 140 | 140 | 0 | 0.873 [0.841, 0.905] | 42/140 = 30.0% [23.0%, 38.0%] |
| df40:faceswap | 140 | 140 | 0 | 0.938 [0.916, 0.960] | 79/140 = 56.4% [48.2%, 64.4%] |
| df40:facevid2vid | 140 | 140 | 0 | 0.895 [0.865, 0.925] | 45/140 = 32.1% [25.0%, 40.3%] |
| df40:fomm | 140 | 140 | 0 | 0.955 [0.935, 0.973] | 68/140 = 48.6% [40.4%, 56.8%] |
| df40:inswap | 118 | 118 | 0 | 0.922 [0.891, 0.951] | 51/118 = 43.2% [34.6%, 52.2%] |
| df40:mobileswap | 140 | 140 | 0 | 0.941 [0.917, 0.962] | 66/140 = 47.1% [39.1%, 55.4%] |
| df40:pirender | 140 | 140 | 0 | 0.903 [0.875, 0.929] | 32/140 = 22.9% [16.7%, 30.5%] |
| df40:simswap | 140 | 140 | 0 | 0.923 [0.895, 0.948] | 50/140 = 35.7% [28.3%, 43.9%] |
| df40:uniface | 140 | 140 | 0 | 0.972 [0.956, 0.985] | 84/140 = 60.0% [51.7%, 67.7%] |
| df40:wav2lip | 140 | 140 | 0 | 0.681 [0.651, 0.717] | 2/140 = 1.4% [0.4%, 5.1%] |

### df40cdf — v0


| family | n | judged | errors | AUC vs reals (95% CI, by identity group) | caught (judged, Wilson 95%) |
|---|---|---|---|---|---|
| df40:blendface | 150 | 150 | 0 | 0.733 [0.673, 0.786] | 43/150 = 28.7% [22.0%, 36.4%] |
| df40:facedancer | 150 | 150 | 0 | 0.845 [0.803, 0.885] | 88/150 = 58.7% [50.7%, 66.2%] |
| df40:faceswap | 150 | 149 | 0 | 0.983 [0.974, 0.991] | 139/149 = 93.3% [88.1%, 96.3%] |
| df40:fsgan | 150 | 150 | 0 | 0.904 [0.877, 0.932] | 104/150 = 69.3% [61.5%, 76.2%] |
| df40:inswap | 150 | 150 | 0 | 0.745 [0.705, 0.786] | 49/150 = 32.7% [25.7%, 40.5%] |
| df40:mobileswap | 150 | 150 | 0 | 0.781 [0.740, 0.821] | 60/150 = 40.0% [32.5%, 48.0%] |
| df40:simswap | 150 | 150 | 0 | 0.693 [0.640, 0.750] | 45/150 = 30.0% [23.2%, 37.8%] |
| df40:uniface | 150 | 150 | 0 | 0.797 [0.752, 0.837] | 60/150 = 40.0% [32.5%, 48.0%] |

### dfd — c40

Reals flagged (all): **14/150 judged = 9.3% [5.6%, 15.1%] (of 150)**.

| family | n | judged | errors | AUC vs reals (95% CI, by identity group) | caught (judged, Wilson 95%) |
|---|---|---|---|---|---|
| DFD | 300 | 298 | 0 | 0.682 [0.617, 0.707] | 67/298 = 22.5% [18.1%, 27.6%] |

### dfd — v0

Reals flagged (all): **7/150 judged = 4.7% [2.3%, 9.3%] (of 150)**.

| family | n | judged | errors | AUC vs reals (95% CI, by identity group) | caught (judged, Wilson 95%) |
|---|---|---|---|---|---|
| DFD | 300 | 299 | 0 | 0.819 [0.785, 0.865] | 140/299 = 46.8% [41.2%, 52.5%] |

### dfd — v1

Reals flagged (all): **10/149 judged = 6.7% [3.7%, 11.9%] (of 150)**.

| family | n | judged | errors | AUC vs reals (95% CI, by identity group) | caught (judged, Wilson 95%) |
|---|---|---|---|---|---|
| DFD | 300 | 297 | 0 | 0.580 [0.552, 0.620] | 33/297 = 11.1% [8.0%, 15.2%] |

### dfd — v2

Reals flagged (all): **11/149 judged = 7.4% [4.2%, 12.7%] (of 150)**.

| family | n | judged | errors | AUC vs reals (95% CI, by identity group) | caught (judged, Wilson 95%) |
|---|---|---|---|---|---|
| DFD | 300 | 299 | 0 | 0.731 [0.700, 0.760] | 79/299 = 26.4% [21.7%, 31.7%] |

### dfd — v3

Reals flagged (all): **11/150 judged = 7.3% [4.1%, 12.7%] (of 150)**.

| family | n | judged | errors | AUC vs reals (95% CI, by identity group) | caught (judged, Wilson 95%) |
|---|---|---|---|---|---|
| DFD | 300 | 300 | 0 | 0.794 [0.760, 0.819] | 120/300 = 40.0% [34.6%, 45.6%] |

### dfd — v4

Reals flagged (all): **13/150 judged = 8.7% [5.1%, 14.3%] (of 150)**.

| family | n | judged | errors | AUC vs reals (95% CI, by identity group) | caught (judged, Wilson 95%) |
|---|---|---|---|---|---|
| DFD | 300 | 299 | 0 | 0.702 [0.675, 0.729] | 79/299 = 26.4% [21.7%, 31.7%] |

### ffpp — c40

Reals flagged (all): **3/140 judged = 2.1% [0.7%, 6.1%] (of 140)**.

| family | n | judged | errors | AUC vs reals (95% CI, by identity group) | caught (judged, Wilson 95%) |
|---|---|---|---|---|---|
| Deepfakes | 140 | 140 | 0 | 0.937 [0.915, 0.959] | 88/140 = 62.9% [54.6%, 70.4%] |
| Face2Face | 140 | 140 | 0 | 0.813 [0.773, 0.852] | 40/140 = 28.6% [21.7%, 36.5%] |
| FaceShifter | 140 | 140 | 0 | 0.859 [0.821, 0.894] | 46/140 = 32.9% [25.6%, 41.0%] |
| FaceSwap | 140 | 140 | 0 | 0.894 [0.863, 0.922] | 66/140 = 47.1% [39.1%, 55.4%] |
| NeuralTextures | 140 | 140 | 0 | 0.702 [0.679, 0.730] | 19/140 = 13.6% [8.9%, 20.2%] |

### ffpp — v0

Reals flagged (all): **1/140 judged = 0.7% [0.1%, 3.9%] (of 140)**.

| family | n | judged | errors | AUC vs reals (95% CI, by identity group) | caught (judged, Wilson 95%) |
|---|---|---|---|---|---|
| Deepfakes | 140 | 140 | 0 | 0.973 [0.958, 0.986] | 103/140 = 73.6% [65.7%, 80.2%] |
| Face2Face | 140 | 140 | 0 | 0.911 [0.885, 0.937] | 67/140 = 47.9% [39.8%, 56.1%] |
| FaceShifter | 140 | 140 | 0 | 0.915 [0.890, 0.938] | 52/140 = 37.1% [29.6%, 45.4%] |
| FaceSwap | 140 | 140 | 0 | 0.938 [0.916, 0.960] | 79/140 = 56.4% [48.2%, 64.4%] |
| NeuralTextures | 140 | 140 | 0 | 0.833 [0.806, 0.862] | 23/140 = 16.4% [11.2%, 23.4%] |

### ffpp — v1

Reals flagged (all): **3/140 judged = 2.1% [0.7%, 6.1%] (of 140)**.

| family | n | judged | errors | AUC vs reals (95% CI, by identity group) | caught (judged, Wilson 95%) |
|---|---|---|---|---|---|
| Deepfakes | 140 | 140 | 0 | 0.933 [0.911, 0.954] | 83/140 = 59.3% [51.0%, 67.1%] |
| Face2Face | 140 | 140 | 0 | 0.821 [0.787, 0.858] | 38/140 = 27.1% [20.5%, 35.0%] |
| FaceShifter | 140 | 140 | 0 | 0.813 [0.774, 0.847] | 27/140 = 19.3% [13.6%, 26.6%] |
| FaceSwap | 140 | 140 | 0 | 0.899 [0.866, 0.929] | 51/140 = 36.4% [28.9%, 44.7%] |
| NeuralTextures | 140 | 140 | 0 | 0.713 [0.687, 0.739] | 13/140 = 9.3% [5.5%, 15.2%] |

### ffpp — v2

Reals flagged (all): **2/140 judged = 1.4% [0.4%, 5.1%] (of 140)**.

| family | n | judged | errors | AUC vs reals (95% CI, by identity group) | caught (judged, Wilson 95%) |
|---|---|---|---|---|---|
| Deepfakes | 140 | 140 | 0 | 0.968 [0.952, 0.982] | 104/140 = 74.3% [66.5%, 80.8%] |
| Face2Face | 140 | 140 | 0 | 0.875 [0.843, 0.904] | 53/140 = 37.9% [30.2%, 46.1%] |
| FaceShifter | 140 | 140 | 0 | 0.900 [0.872, 0.926] | 51/140 = 36.4% [28.9%, 44.7%] |
| FaceSwap | 140 | 140 | 0 | 0.934 [0.911, 0.958] | 88/140 = 62.9% [54.6%, 70.4%] |
| NeuralTextures | 140 | 140 | 0 | 0.780 [0.753, 0.812] | 16/140 = 11.4% [7.2%, 17.8%] |

### ffpp — v3

Reals flagged (all): **1/140 judged = 0.7% [0.1%, 3.9%] (of 140)**.

| family | n | judged | errors | AUC vs reals (95% CI, by identity group) | caught (judged, Wilson 95%) |
|---|---|---|---|---|---|
| Deepfakes | 140 | 140 | 0 | 0.975 [0.961, 0.987] | 107/140 = 76.4% [68.8%, 82.7%] |
| Face2Face | 140 | 140 | 0 | 0.903 [0.877, 0.930] | 60/140 = 42.9% [35.0%, 51.1%] |
| FaceShifter | 140 | 140 | 0 | 0.915 [0.889, 0.938] | 57/140 = 40.7% [32.9%, 49.0%] |
| FaceSwap | 140 | 140 | 0 | 0.939 [0.917, 0.960] | 85/140 = 60.7% [52.4%, 68.4%] |
| NeuralTextures | 140 | 140 | 0 | 0.815 [0.788, 0.845] | 20/140 = 14.3% [9.4%, 21.0%] |

### ffpp — v4

Reals flagged (all): **0/140 judged = 0.0% [0.0%, 2.7%] (of 140)**.

| family | n | judged | errors | AUC vs reals (95% CI, by identity group) | caught (judged, Wilson 95%) |
|---|---|---|---|---|---|
| Deepfakes | 140 | 140 | 0 | 0.965 [0.949, 0.980] | 97/140 = 69.3% [61.2%, 76.3%] |
| Face2Face | 140 | 140 | 0 | 0.863 [0.829, 0.897] | 48/140 = 34.3% [26.9%, 42.5%] |
| FaceShifter | 140 | 140 | 0 | 0.876 [0.843, 0.904] | 48/140 = 34.3% [26.9%, 42.5%] |
| FaceSwap | 140 | 140 | 0 | 0.924 [0.900, 0.947] | 74/140 = 52.9% [44.6%, 60.9%] |
| NeuralTextures | 140 | 140 | 0 | 0.772 [0.743, 0.801] | 18/140 = 12.9% [8.3%, 19.4%] |

### uadfv — v0

Reals flagged (all): **3/49 judged = 6.1% [2.1%, 16.5%] (of 49)**.

| family | n | judged | errors | AUC vs reals (95% CI, by identity group) | caught (judged, Wilson 95%) |
|---|---|---|---|---|---|
| UADFV | 100 | 99 | 0 | 0.870 [0.827, 0.917] | 59/99 = 59.6% [49.7%, 68.7%] |

## docs/13 §6.3 gates (cloud part)

| gate | pass | value |
|---|---|---|
| 1. FF++ test AUC >= 0.85 on Deepfakes | yes | 0.973 |
| 1. FF++ test AUC >= 0.85 on FaceSwap | yes | 0.938 |
| 1. FF++ test AUC >= 0.85 on FaceShifter | yes | 0.915 |
| 1b. DF40 face-swap methods on FF++ test identities: mean AUC >= 0.80, none < 0.70 | yes | mean 0.926, min 0.873 over 7 methods |
| 2. FF++ test reals flagged <= 5/134 of judged | yes | 1/140 judged (of 140) |
| 2b. FF++ test reals after v1 flagged <= 5% of judged | yes | 3/140 judged (of 140) |
| 2b. FF++ test reals after c40 flagged <= 5% of judged | yes | 3/140 judged (of 140) |
| 2b. FF++ test reals after v3 flagged <= 5% of judged | yes | 1/140 judged (of 140) |
| 3. DeepFakeDetection AUC >= 0.70 and reals flagged <= 8% of judged | yes | AUC 0.819, 7/150 judged flagged (of 150) |
| 3b. Celeb-DF v2 test list: AUC >= 0.75 and reals flagged <= 8% of judged | **no** | AUC 0.778, 20/178 judged flagged (of 178) |
| 4b. Celeb-DF v2 reals outside the test list flagged <= 5% of judged | **no** | 63/711 judged (of 712) |
| 4a. frozen round-3 gate-4 list flagged <= 3% of judged, n >= 200, no source > 6% | n/a | measured on the laptop (scripts/check_faceswap_reals.py --manifest data/manifests/faceswap_gate4a.csv) |
