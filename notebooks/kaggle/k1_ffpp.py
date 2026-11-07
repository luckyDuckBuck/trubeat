# K1 (CPU sessions): FaceForensics++ c23 + c40 and DeepFakeDetection -> faces,
# the six processing variants, embeddings (BACKBONES), stage-2 crops. Sharded:
# this session handles shard SHARD (I/N) of the videos.
setup(gpu=GPU)
resume_from_inputs()
TF2 = f"{PY} {REPO}/scripts/train_faceswap_v2.py"
os.chdir(REPO)
for b in BACKBONES.split():
    sh(f"{PY} -m trubeat.cli fetch-models --backbone {b}")
sh(f"{TF2} fetch --root {DATA} --datasets ffpp --compression c23 c40 --threads 4" + (f" --shard {SHARD}" if SHARD else ""))
shutil.copy(f"{DATA}/fetch_report.json", f"{W}/fetch_report_k1.json")
sh(f"{TF2} manifest --ffpp {DATA}/ffpp --out {W}/manifest_k1.csv")
budget = max(600, left_s() - 900)
sh(f"{TF2} extract --manifest {W}/manifest_k1.csv --features-dir {W}/features --workers {4 if GPU else 3} "
   f"--device {'cuda' if GPU else 'cpu'} --backbones {BACKBONES} --stage2-dir {W}/crops380 --time-budget-s {budget:.0f}")
log("K1 done")
