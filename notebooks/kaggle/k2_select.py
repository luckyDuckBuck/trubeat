# K2 (CPU): docs/13 §6.2 selection and the cloud part of the §6.3 gates, on the
# features of K1 and K1b (mounted read-only under /kaggle/input/).
setup(gpu=False)
os.chdir(REPO)
import glob  # noqa: E402

feat = "/tmp/features"
os.makedirs(feat, exist_ok=True)
manifests = []
for d in sorted(glob.glob("/kaggle/input/*")):
    for sub in glob.glob(f"{d}/features/*"):
        if os.path.isdir(sub):
            name = os.path.basename(sub)
            if not os.path.exists(f"{feat}/{name}"):
                os.symlink(sub, f"{feat}/{name}")
            log(f"features {name} <- {sub}")
    manifests += sorted(glob.glob(f"{d}/manifest_k1*.csv"))
log(f"manifests {manifests}")
sh(f"{PY} -c \"import pandas as pd, sys; pd.concat([pd.read_csv(m, dtype=str, keep_default_na=False) for m in sys.argv[1:]])"
   f".drop_duplicates('video_id').to_csv('{W}/manifest.csv', index=False)\" " + " ".join(manifests))
for d in sorted(glob.glob("/kaggle/input/*")):
    for f in glob.glob(f"{d}/features/extract_log.json") + glob.glob(f"{d}/fetch_report*.json"):
        shutil.copy(f, f"{W}/{os.path.basename(d)}_{os.path.basename(f)}")
sh(f"{PY} {REPO}/scripts/train_faceswap_v2.py select --manifest {W}/manifest.csv --features-dir {feat} "
   f"--out-dir {W}/result --allow-errors")
log("K2 done")
