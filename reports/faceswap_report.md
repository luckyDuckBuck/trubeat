# Face-swap detector — 2026-11-06 23:38

Protocol: docs/12 §6. Selection on FaceForensics++ train+val only (5000 videos with a face; 160 without enough faces left out). Test and DeepFakeDetection never used for selection or the threshold. Config hash `3c7b7054`.

## Selection (leave-one-manipulation-out AUC, train+val)

| backbone | feature | C | LOMO macro | Deepfakes | Face2Face | FaceShifter | FaceSwap | NeuralTextures |
|---|---|---|---|---|---|---|---|---|
| vit_base_patch16_clip_224.openai | cls | 0.01 | 0.8381 | 0.952 | 0.834 | 0.819 | 0.896 | 0.689 |
| vit_base_patch16_clip_224.openai | cls | 0.1 | 0.8330 | 0.931 | 0.838 | 0.810 | 0.874 | 0.712 |
| vit_base_patch16_clip_224.openai | cls+patch | 0.01 | 0.8226 | 0.945 | 0.819 | 0.784 | 0.883 | 0.682 |
| vit_base_patch16_clip_224.openai | cls+patch | 0.1 | 0.8159 | 0.930 | 0.813 | 0.770 | 0.862 | 0.704 |
| vit_base_patch16_clip_224.openai | cls | 1.0 | 0.8120 | 0.904 | 0.817 | 0.786 | 0.848 | 0.706 |
| vit_base_patch16_clip_224.openai | cls+patch | 1.0 | 0.8088 | 0.920 | 0.799 | 0.764 | 0.854 | 0.708 |
| vit_base_patch14_dinov2.lvd142m | cls+patch | 0.01 | 0.7590 | 0.918 | 0.811 | 0.679 | 0.714 | 0.672 |
| vit_base_patch14_dinov2.lvd142m | cls+patch | 0.1 | 0.7559 | 0.908 | 0.792 | 0.701 | 0.699 | 0.679 |
| vit_base_patch14_dinov2.lvd142m | cls+patch | 1.0 | 0.7487 | 0.885 | 0.775 | 0.714 | 0.692 | 0.678 |
| vit_base_patch14_dinov2.lvd142m | cls | 0.01 | 0.7269 | 0.880 | 0.793 | 0.635 | 0.670 | 0.656 |
| vit_base_patch14_dinov2.lvd142m | cls | 0.1 | 0.7146 | 0.860 | 0.765 | 0.643 | 0.660 | 0.645 |
| vit_small_patch14_dinov2.lvd142m | cls+patch | 0.01 | 0.6974 | 0.836 | 0.765 | 0.657 | 0.623 | 0.606 |
| vit_base_patch14_dinov2.lvd142m | cls | 1.0 | 0.6923 | 0.821 | 0.728 | 0.637 | 0.641 | 0.633 |
| vit_small_patch14_dinov2.lvd142m | cls+patch | 0.1 | 0.6889 | 0.815 | 0.764 | 0.648 | 0.608 | 0.610 |
| vit_small_patch14_dinov2.lvd142m | cls+patch | 1.0 | 0.6753 | 0.795 | 0.749 | 0.642 | 0.587 | 0.605 |
| vit_small_patch14_dinov2.lvd142m | cls | 0.1 | 0.6625 | 0.770 | 0.744 | 0.604 | 0.577 | 0.618 |
| vit_small_patch14_dinov2.lvd142m | cls | 0.01 | 0.6614 | 0.782 | 0.734 | 0.610 | 0.575 | 0.606 |
| vit_small_patch14_dinov2.lvd142m | cls | 1.0 | 0.6524 | 0.754 | 0.733 | 0.593 | 0.571 | 0.611 |

Selected: **vit_base_patch16_clip_224.openai, cls, C=0.01**. In-domain out-of-fold AUC 0.913. Threshold +1.5772 (out-of-fold reals at 3% false alarms).

## Held-out results

### FaceForensics++ c23 test (official split)

Reals flagged: **1/140 = 0.7%** [0.1%, 3.9%] (6 without enough faces).

| family | n | no face | AUC vs reals (95% CI) | caught at threshold (95% CI) |
|---|---|---|---|---|
| Deepfakes | 140 | 6 | 0.967 [0.947, 0.984] | 111/140 = 79.3% [71.8%, 85.2%] |
| Face2Face | 140 | 5 | 0.922 [0.890, 0.950] | 75/140 = 53.6% [45.3%, 61.6%] |
| FaceShifter | 140 | 6 | 0.928 [0.898, 0.954] | 73/140 = 52.1% [43.9%, 60.2%] |
| FaceSwap | 140 | 6 | 0.952 [0.923, 0.973] | 96/140 = 68.6% [60.5%, 75.7%] |
| NeuralTextures | 140 | 6 | 0.850 [0.804, 0.892] | 38/140 = 27.1% [20.5%, 35.0%] |

### DeepFakeDetection (cross-dataset)

Reals flagged: **3/150 = 2.0%** [0.7%, 5.7%] (83 without enough faces).

| family | n | no face | AUC vs reals (95% CI) | caught at threshold (95% CI) |
|---|---|---|---|---|
| DFD | 300 | 161 | 0.878 [0.831, 0.924] | 77/300 = 25.7% [21.1%, 30.9%] |

## docs/12 §6 gates

| gate | pass | value |
|---|---|---|
| 1. FaceForensics++ test AUC >= 0.85 on Deepfakes | yes | 0.967 |
| 1. FaceForensics++ test AUC >= 0.85 on FaceSwap | yes | 0.952 |
| 1. FaceForensics++ test AUC >= 0.85 on FaceShifter | yes | 0.928 |
| 2. FaceForensics++ test reals flagged <= 5/140 | yes | 1/140 |
| 3. DeepFakeDetection AUC >= 0.70 and reals flagged <= 8% | yes | AUC 0.878, 3/150 flagged |
| 4. real non-FaceForensics clips with a face flagged <= 3% (n >= 200) | n/a | measured on the laptop (scripts/check_faceswap_reals.py) |
## Laptop: the FaceForensics++ test clips on disk, as downloaded

Bundle `src/trubeat/models/faceswap.joblib`, threshold +1.5772 fixed.

### FaceForensics++ c23 test (official split)

Reals flagged: **1/140 = 0.7%** [0.1%, 3.9%] (6 without enough faces).

| family | n | no face | AUC vs reals (95% CI) | caught at threshold (95% CI) |
|---|---|---|---|---|
| Deepfakes | 31 | 3 | 0.940 [0.864, 0.991] | 21/31 = 67.7% [50.1%, 81.4%] |
| Face2Face | 31 | 0 | 0.893 [0.806, 0.959] | 15/31 = 48.4% [32.0%, 65.2%] |
| FaceSwap | 28 | 1 | 0.950 [0.900, 0.987] | 16/28 = 57.1% [39.1%, 73.5%] |
| NeuralTextures | 29 | 1 | 0.844 [0.764, 0.922] | 7/29 = 24.1% [12.2%, 42.1%] |

## Laptop: the same clips after a WhatsApp-like re-encode (640 px wide, CRF 30)

Bundle `src/trubeat/models/faceswap.joblib`, threshold +1.5772 fixed.

### FaceForensics++ c23 test (official split)

Reals flagged: **45/140 = 32.1%** [25.0%, 40.3%] (7 without enough faces).

| family | n | no face | AUC vs reals (95% CI) | caught at threshold (95% CI) |
|---|---|---|---|---|
| Deepfakes | 31 | 3 | 0.899 [0.819, 0.966] | 25/31 = 80.6% [63.7%, 90.8%] |
| Face2Face | 31 | 0 | 0.736 [0.634, 0.835] | 21/31 = 67.7% [50.1%, 81.4%] |
| FaceSwap | 28 | 1 | 0.859 [0.745, 0.950] | 23/28 = 82.1% [64.4%, 92.1%] |
| NeuralTextures | 29 | 1 | 0.568 [0.452, 0.683] | 11/29 = 37.9% [22.7%, 56.0%] |
