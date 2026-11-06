# 12 — Round 3: a real face-swap detector, and what else it needs (2026-11-06)

**Status: approved to execute (human, 2026-11-06):** *"first plan the face
swap detector and anything else that's needed; build a full-scale plan;
once the plan is ready start implementing; once implementation is done
conduct a review, then push"*, and *"if it's for training I'd prefer doing
[it] online since I don't have much storage, but I do have the Google AI Pro
plan"*. The decision rules in §6 are fixed by this commit, before any
face-swap result exists.

---

## 1. Why

A user clip came back **"No sign of AI generation found"**. It is a
WhatsApp copy of the 2020 RepresentUs ad "Dictators: Kim Jong-Un": a
professional face-swap (an actor's performance with Kim Jong Un's face
put on), not a fully generated video. Measured locally:

| check | result on the clip | why it missed |
|---|---|---|
| AI-video detector (whole frame) | −0.2 to −1.3 on every 2 s segment of the face footage (threshold +1.11) | built for clips generated from scratch; flags 0% of FaceForensics++ face-swaps (docs/11 R2-3) |
| face-swap check (pulse, G2) | −0.28, threshold −0.605 | weak by design: about a quarter of Deepfakes-style swaps; the pulse is unreadable on WhatsApp-quality video |
| deepfakedetection.io (user's screenshot) | "forgery probability 5%" | (external; shows this is a hard case, not only ours) |

Side finding: the AI-video detector scores the ad's **black title cards**
+2.6 to +5.3 (far above threshold). Today only the first 2 s of a clip is
scored, so a real video that **opens** with a title card would be flagged
"Likely AI-generated". That is a false-alarm risk to fix (§5, item A).

## 2. The machine, and where each step runs

| resource | this laptop | consequence |
|---|---|---|
| CPU | Ryzen 5 4600H, 6 cores / 12 threads | fine for inference and small jobs; slow for embedding 100k face crops (~1–2 h per backbone) |
| RAM | 7.1 GB (+11 GB swap) | no large batches; decode only the frames needed |
| GPU | GTX 1650 Mobile, 4 GB, CUDA-capable (driver 615), **unused**: PyTorch is the CPU build | a usable fallback for embedding (needs a ~3 GB CUDA PyTorch install); too small for comfortable fine-tuning |
| disk | 39 GB free | the full FaceForensics++ c23 set plus a cross-dataset set is ~15 GB: possible, but the user prefers not to |
| network | home Wi-Fi (down at the time of writing: "connected, site only") | downloads wait for it |
| Google AI Pro | 5 TB Drive storage ([since April 2026](https://www.techradar.com/ai-platforms-assistants/google-has-doubled-its-storage-allowance-for-its-ai-pro-plan-and-its-just-one-upped-chatgpt-heres-how-they-compare)); Colab benefits reported, possibly region-limited (unverified, see §8) | data and training go to **Colab + Drive**; only the trained model (kilobytes) comes back |

Split of work:

| step | where | who |
|---|---|---|
| code, tests, app integration, docs | laptop | agent |
| smoke run of the training pipeline on the 603 FaceForensics++ clips already on disk | laptop CPU | agent |
| download FaceForensics++ c23 (all six sets) + DeepFakeDetection subset (~15 GB) | Colab runtime disk (never the laptop) | notebook |
| face crops + embeddings for every candidate backbone | Colab GPU | notebook |
| classifier selection, threshold, held-out evaluation | Colab (seconds) | notebook |
| bring back the model + report (< 1 MB) | Drive → `runs/faceswap/` | **human**: one download |
| false-alarm check on non-FaceForensics real video, robustness, the RepresentUs clip | laptop (data already on disk) | agent |

Human steps, each a single action: (1) open `notebooks/train_faceswap.ipynb`
in Colab, pick a GPU runtime, Run all; (2) download the one result file
from Drive into the repo. Fallback if Colab GPU is not available: the
laptop's GTX 1650 (agent installs CUDA PyTorch, ~3 GB, and downloads the
data locally, ~15 GB), or Kaggle's free GPU hours (needs an account: human).

**Privacy:** only public research videos (FaceForensics++, DeepFakeDetection)
are processed in Colab, downloaded there directly from their servers.
Nothing from this laptop is uploaded, user videos never leave it, and
inference stays local (AGENTS.md §7).

## 3. Design: the face-swap detector (FSD)

Same recipe as the AI-video detector, which already works on this CPU: a
frozen vision model describes the input, a linear classifier decides.
Here the input is the **face**, cropped from several frames.

1. **Frames:** 16 frames evenly spaced over the first min(duration, 120 s),
   decoded at **native resolution** (only those 16 frames are kept, so
   memory stays small). Face-swap traces are fine detail that downscaling
   to 480p would erase.
2. **Face crop:** MediaPipe Face Landmarker (already shipped, image mode)
   → largest face → square box around the landmarks, enlarged 1.3× →
   224 × 224. Frames without a face are dropped; fewer than 4 faces in 16
   frames → the check reports "no face" and does not decide.
3. **Features:** a frozen backbone embeds each crop; the clip descriptor is
   the mean over crops. Candidates (selection in §6): DINOv2-S/14 (already
   installed), CLIP ViT-B/16 (OpenAI weights via timm), DINOv2-B/14; each
   as CLS token, and CLS + mean patch token.
4. **Classifier:** StandardScaler + logistic regression, balanced classes.
5. **Bundle:** `src/trubeat/models/faceswap.joblib` (tens of KB): model,
   backbone, crop settings, threshold, real-clip reference scores (for the
   percentile shown in the app), config hash, training data, metrics. A new
   backbone's weights come through `trubeat fetch-models` (the only
   network step stays the only network step).

The same `faceswap.py` code does the sampling and cropping in Colab and in
the app (the notebook installs this repo), so training and inference see
identical crops.

**Stage 2, only if stage 1 fails gate 1 (§6):** fine-tune a CNN
(EfficientNet-B4 at 380 px, the classic FaceForensics++/DFDC recipe) on the
Colab GPU. Its ~75 MB weights would ship as a release asset fetched by
`trubeat fetch-models`; CPU inference ~3 s for 16 crops.

## 4. Data

| set | role | size | access |
|---|---|---|---|
| FaceForensics++ c23: original + Deepfakes, Face2Face, FaceSwap, NeuralTextures, **FaceShifter** (1,000 each) | train / val / test by the official identity-disjoint split (720 / 140 / 140 per set) | ~11 GB, Colab only | existing FaceForensics++ access (terms already accepted for this project) |
| DeepFakeDetection (Google/Jigsaw, actors; different swap method and people) | cross-dataset test only | 150 real + 300 fake, ~5 GB, Colab only | same access |
| GenBuster-Bench + AIGVDBench real clips with a detected face | false alarms on real non-FaceForensics video | ≥ 200 clips, already on the laptop | public |
| Celeb-DF v2 | optional second cross-dataset test | — | **human gate**: access form |
| the RepresentUs clip, the user's own labelled clips (docs/11 P1) | reported case by case, never used to tune | — | laptop only |

The model is trained on FaceForensics++, whose terms allow non-commercial
research use; the README will say so next to the model.

## 5. Everything else this round needs

| # | item | why | done when |
|---|---|---|---|
| A | **Title-card guard** for the AI-video detector: a segment whose frames are mostly flat (≥ 70% of 8×8 blocks with std < 4, median over the segment's frames) is "graphic"; scoring moves to the first natural segment within the first 30 s; if none, the AI layer reports "not checked: graphics only" | a real clip opening on a title card would be flagged "Likely AI-generated" (§1) | the rule flags < 0.5% of the 8,655 dataset clips' first segments (else the rule is revised before use), flags the RepresentUs end cards, and has synthetic tests |
| B | Face-swap layer in the app: runs on clips ≥ 2 s with a face; verdict priority Content Credentials > AI-video > face-swap detector > pulse check (evidence only once the FSD is adopted) > "No sign…"; the detail shows score, threshold, percentile among real faces, faces found | the point of the round | live app run on FaceForensics++ clips and the RepresentUs clip |
| C | Evidence image: the face crop the detector scored most suspicious, with its timestamp | lets a user see what was judged | shown in the app; absent when no face |
| D | Progress bar: the new stage gets its own share and per-frame text | the bar was just fixed; keep it honest | progress test covers the stage |
| E | HEVC note under the upload box ("preview may not play for HEVC / iPhone video; the analysis still runs") | Chrome on Linux cannot play HEVC; users saw "Video not playable" | visible in the app |
| F | Caveat, README, docs/04 (new module interface), docs/11 pointer, extension caveat (+ release if its text changes), website Updates entry | every number shown must be the measured one | text matches the final report |
| G | Review, then push | requested | §7 |

## 6. Decision rules (fixed now, before any result)

**Selection** (FaceForensics++ train + val only; test never touched):
candidates = backbone × feature (CLS, CLS+patch-mean) × C ∈ {0.01, 0.1, 1}.
Choose by the macro AUC of **leave-one-manipulation-out** cross-validation
over the five fake sets (train on four, score the held-out one against held-
out real identities), because real-world fakes come from methods the model
has not seen. Threshold: the 97th percentile of out-of-fold real scores
(target false-alarm rate 3%).

**Adoption** of the selected model as the decisive face-swap check
(replacing the pulse check's decisive role), all on data never used above:

1. FaceForensics++ test: AUC ≥ 0.85 on **each** face-swap set (Deepfakes,
   FaceSwap, FaceShifter);
2. FaceForensics++ test reals flagged ≤ 5 of 140;
3. DeepFakeDetection (cross-dataset): AUC ≥ 0.70 and reals flagged ≤ 8%;
4. real non-FaceForensics clips with a face (GenBuster / AIGVDBench, n ≥ 200):
   flagged ≤ 3%.

Outcomes: all four pass → **decisive** ("Possible face-swap"). Gates 1–2
pass, 3 or 4 fails → **evidence only** (score shown, never decides; the
caveat says why). Gate 1 fails → not shipped; stage 2 (§3) is tried under
the same gates. Reported in every case, never gated: Face2Face and
NeuralTextures AUC (reenactment), a WhatsApp-like re-encode of the test set
(scale to 640 px wide, CRF 30), and the RepresentUs clip's score. 95%
bootstrap intervals on every AUC; Wilson intervals on every rate.

**Title-card guard (A)** is adopted only if it flags < 0.5% of dataset
first segments; held-out AI-video numbers are then recomputed and must not
change by more than the flagged share.

## 7. Order of work

| phase | what | where | done when |
|---|---|---|---|
| 0 | this plan, committed | laptop | — |
| 1 | `src/trubeat/faceswap.py` (sampling, crops, embedding, scoring) + `scripts/train_faceswap.py` (shared with Colab) + tests; smoke run on the 603 local FaceForensics++ clips | laptop | pipeline runs end to end; smoke numbers recorded as *preliminary, in-domain only* |
| 2 | title-card guard (A) + dataset flag-rate check; HEVC note (E); progress stage (D) | laptop | tests green; flag rate measured |
| 3 | `notebooks/train_faceswap.ipynb`: install repo, download data, crops, embeddings, selection, test, DeepFakeDetection, write `faceswap_result.zip` to Drive; resumable from Drive after a disconnect | laptop (write), Colab (run) | **human runs it**; result file in `runs/faceswap/` |
| 4 | apply §6; gate 4 + robustness + RepresentUs on the laptop; app integration (B, C); texts (F) | laptop | live app shows the new layer with measured numbers |
| 5 | review: `/code-review` on the full round diff + an independent read of the notebook; fix; full tests + CI-style core install | laptop | no open findings |
| 6 | push the tool; extension release if its text changed; website Updates entry; push the website | GitHub, Vercel | CI green, site deployed |

Commits follow AGENTS.md §8 and carry no AI attribution (the user's rule,
2026-11-06).

## 8. Risks and honest expectations

- **This may still miss the RepresentUs ad.** Detectors trained on
  FaceForensics++ typically score 0.95+ AUC in-domain and much lower on
  other datasets; a professional, re-compressed swap is the hardest case.
  The clip is reported, never tuned on.
- **Reenactment** (Face2Face, NeuralTextures) keeps the real face's
  identity and is often missed by face-swap detectors; it is reported, not
  promised.
- **Colab access:** the AI Pro plan's Colab benefit is reported (about 200
  compute units a month, [user report](https://www.mydealz.de/deals/google-ai-pro-200-google-colab-compute-units-wert-21eur-monatlich-inklusive-auch-im-5eurmonat-promo-abo80gb-nvidia-a100-96gb-g4-gpu-2857205))
  but may be limited to some countries; free Colab's T4 also suffices for
  this job (embeddings, not fine-tuning). Check the compute-units display
  in Colab's runtime menu.
- **Data access:** the FaceForensics++ download servers were unreachable at
  the time of writing because this laptop had no internet; the notebook
  tries all three official mirrors.
- **Size of evidence:** 140 test reals bound a false-alarm rate only to
  about ±3 points; gate 4 adds ≥ 200 non-FaceForensics reals for that reason.
