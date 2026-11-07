# AI-video detector: second-round audit

## 1. Metadata shortcut (fps, size, aspect, duration, bitrate of the original files)

Macro leave-one-generator-out AUC from metadata alone: **0.993** (worst generator 0.936). The detector never sees these values (every clip is re-encoded to one format and centre-cropped first), but a high number means the datasets differ in ways unrelated to generation.

| label | m_fps | m_w | m_h | m_aspect | m_dur | m_kbps |
|---|---|---|---|---|---|---|
| ai | 24.0 | 768.0 | 480.0 | 1.67 | 5.0 | 934.35 |
| real | 25.0 | 1920.0 | 1080.0 | 1.78 | 6.28 | 3708.19 |

## 2. Near-duplicate content, external vs pool (cosine >= 0.97)

| external source | label | near-duplicates | clips |
|---|---|---|---|
| ffpp_original | real | 0 | 135 |
| genbuster_id | ai | 0 | 1000 |
| genbuster_id | real | 8 | 1000 |
| genbuster_ood | ai | 54 | 1000 |
| genbuster_wild | ai | 0 | 150 |

## 3. External results: all external (n=3285)

| measure | value |
|---|---|
| auc_id | 0.872 |
| auc_ood | 0.954 |
| auc_wild | 0.859 |
| fpr_ffpp_real | 0.000 |
| fpr_genbuster_real | 0.057 |
| recall_id | 0.499 |
| recall_ood | 0.783 |
| recall_wild | 0.453 |

## 3. External results: near-duplicates removed (n=3223)

| measure | value |
|---|---|
| auc_id | 0.874 |
| auc_ood | 0.954 |
| auc_wild | 0.861 |
| fpr_ffpp_real | 0.000 |
| fpr_genbuster_real | 0.055 |
| recall_id | 0.499 |
| recall_ood | 0.783 |
| recall_wild | 0.453 |

## 4. FF++ c23 test clips through the AI-video detector (flag rate)

| family | n | flagged as AI |
|---|---|---|
| Deepfakes | 29 | 0.000 |
| Face2Face | 31 | 0.000 |
| FaceSwap | 27 | 0.000 |
| NeuralTextures | 28 | 0.000 |
| original | 40 | 0.000 |
