# K1 (GPU): FaceForensics++ c23 + c40 and DeepFakeDetection -> faces, the six
# processing variants, embeddings for both candidate backbones, stage-2 crops.
# Sharded: this session handles shard SHARD (I/N) of the videos.
setup(gpu=True)
resume_from_inputs()
TF2 = f"{PY} {REPO}/scripts/train_faceswap_v2.py"
os.chdir(REPO)
sh(f"{PY} -m trubeat.cli fetch-models --backbone vit_base_patch16_clip_224.openai")
sh(f"{PY} -m trubeat.cli fetch-models --backbone vit_large_patch14_clip_224.openai")
sh(f"{TF2} fetch --root {DATA} --datasets ffpp --compression c23 c40 --threads 12" + (f" --shard {SHARD}" if SHARD else ""))
shutil.copy(f"{DATA}/fetch_report.json", f"{W}/fetch_report_k1.json")
sh(f"{TF2} manifest --ffpp {DATA}/ffpp --out {W}/manifest_k1.csv")
budget = max(600, left_s() - 900)
sh(f"{TF2} extract --manifest {W}/manifest_k1.csv --features-dir {W}/features --workers 4 --device cuda "
   f"--backbones vit_base_patch16_clip_224.openai vit_large_patch14_clip_224.openai "
   f"--stage2-dir {W}/crops380 --time-budget-s {budget:.0f}")
log("K1 done")
