# 09 — Data Access Starter (do these tonight)

Form approvals take days; everything below is a 10-minute action that unblocks
Phase 0. Links were verified live on 2026-10-11.

## 1. FaceForensics++ — submit tonight

- **Form:** https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform
  (linked from the official repo README: https://github.com/ondyari/FaceForensics)
- Fields: name, institution, supervisor, email, agreement to the ToS.
- ToS worth knowing before you sign: non-commercial research use only; if you're
  employed by a for-profit entity, that employer is also bound (fine for a
  Forgestone Studios LLP research phase — just be honest in the purpose field).
- If no reply within a week, their README says the email is likely bouncing —
  check spam before re-requesting.
- On approval you get the download script by email. Start small:

```bash
python faceforensics_download_v4.py ff_data --dataset original --compression c23 --type videos --num_videos 25 --server EU2
python faceforensics_download_v4.py ff_data --dataset Deepfakes --compression c23 --type videos --num_videos 25 --server EU2
```

- When you're ready for the full pull: real + all four fake methods (DeepFakes,
  Face2Face, FaceSwap, NeuralTextures) at **c23** only (~40 GB). Never pull raw
  unless a compression-robustness experiment demands it.

## 2. Celeb-DF v2 — submit tonight, one friction

- **Form:** https://forms.gle/2jYBby6y1FBU3u6q9
  (linked from the official repo README: https://github.com/yuezunli/celeb-deepfakeforensics)
- **Friction:** the form explicitly asks for an **academic** email address, and
  the terms are non-commercial research only. As a solo founder without a .edu
  address, don't fake one — use your real email and state the research purpose
  plainly; if it's rejected, email `deepfakeforensics@gmail.com` with a one-line
  research-purpose note. Worst case this stays the Phase-3 held-out exam and you
  defer it; it never blocks Phase 1 or 2.
- ~14 GB. It has an official test split (`List_of_testing_videos.txt`, 518 videos)
  — use it, don't re-split.

## 3. UBFC-rPPG — pull tonight, no form needed

- **Official page:** https://sites.google.com/view/ybenezeth/ubfcrppg
  (cited in the rPPG literature as the dataset page; Drive folder behind it)
- No request form — direct download. Drive **rate-limits** the large files: if a
  download throttles, wait and retry at off-peak hours rather than hammering it;
  Kaggle mirrors exist as a fallback.
- Tiny (~a few GB) and uncompressed — this is the fastest win and unblocks all
  of Phase 1.
- License note already in `05_data_strategy.md`: subjects **21 and 27** may not
  appear in any publication or presentation.

## Suggested purpose text for the forms

"Independent research at Forgestone Studios LLP: building an open rPPG-based
deepfake detector (pulse-consistency signal) and validating it against published
baselines. Non-commercial research use; results will be compared against
FakeCatcher/FaceForensics++ benchmarks."

## Tonight's checklist

- [ ] FF++ form submitted
- [ ] Celeb-DF form submitted (or deferral decided)
- [ ] UBFC-rPPG downloading (Phase 1 unblocks here)
- [ ] Confirm download script email lands in inbox (check spam in 2–3 days)
