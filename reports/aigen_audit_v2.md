# AI-video detector: second-round audit

## 1. Metadata shortcut (fps, size, aspect, duration, bitrate of the original files)

Macro leave-one-generator-out AUC from metadata alone: **0.994** (worst generator 0.949). The detector never sees these values (every clip is re-encoded to one format and centre-cropped first), but a high number means the datasets differ in ways unrelated to generation.

| label | m_fps | m_w | m_h | m_aspect | m_dur | m_kbps |
|---|---|---|---|---|---|---|
| ai | 24.0 | 864.0 | 512.0 | 1.76 | 5.04 | 2048.39 |
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
| auc_id | 0.880 |
| auc_ood | 0.961 |
| auc_wild | 0.880 |
| fpr_ffpp_real | 0.000 |
| fpr_genbuster_real | 0.054 |
| median_score_ai | 1.828 |
| median_score_real | -1.442 |
| n_ai_id | 1000.000 |
| n_ai_ood | 1000.000 |
| n_ai_wild | 150.000 |
| n_ffpp_real | 135.000 |
| n_genbuster_real | 1000.000 |
| recall_id | 0.522 |
| recall_ood | 0.831 |
| recall_wild | 0.547 |

## 3. External results: near-duplicates removed (n=3223)

| measure | value |
|---|---|
| auc_id | 0.881 |
| auc_ood | 0.961 |
| auc_wild | 0.881 |
| fpr_ffpp_real | 0.000 |
| fpr_genbuster_real | 0.053 |
| median_score_ai | 1.816 |
| median_score_real | -1.447 |
| n_ai_id | 1000.000 |
| n_ai_ood | 946.000 |
| n_ai_wild | 150.000 |
| n_ffpp_real | 135.000 |
| n_genbuster_real | 992.000 |
| recall_id | 0.522 |
| recall_ood | 0.831 |
| recall_wild | 0.547 |

## 4. FF++ c23 test clips through the AI-video detector (flag rate)

| family | n | flagged as AI |
|---|---|---|
| Deepfakes | 29 | 0.000 |
| Face2Face | 31 | 0.000 |
| FaceSwap | 27 | 0.000 |
| NeuralTextures | 28 | 0.000 |
| original | 40 | 0.000 |
