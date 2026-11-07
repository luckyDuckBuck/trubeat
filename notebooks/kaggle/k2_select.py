# K2 (CPU): docs/13 §6.2 selection and the cloud part of the §6.3 gates, on the
# features of the K1 shards and K1b (mounted read-only under /kaggle/input/).
setup(gpu=False)
os.chdir(REPO)
import glob  # noqa: E402

feat = "/tmp/features"
manifests = []
n_linked = {}
for src in input_dirs("features"):          # shards share features/ffpp/: link file by file
    for npz in glob.glob(f"{src}/*/*.npz"):
        ds = os.path.basename(os.path.dirname(npz))
        os.makedirs(f"{feat}/{ds}", exist_ok=True)
        dst = f"{feat}/{ds}/{os.path.basename(npz)}"
        if not os.path.exists(dst):
            os.symlink(npz, dst)
            n_linked[src] = n_linked.get(src, 0) + 1
    parent = os.path.dirname(src)
    manifests += sorted(glob.glob(f"{parent}/manifest_k1*.csv"))
    for f in [f"{src}/extract_log.json"] + glob.glob(f"{parent}/fetch_report*.json"):
        if os.path.isfile(f):
            shutil.copy(f, f"{W}/{os.path.basename(parent)}_{os.path.basename(f)}")
for k, v in n_linked.items():
    log(f"features: {v} files from {k}")
log(f"manifests {manifests}")
if not manifests:
    raise SystemExit("no K1/K1b manifests under /kaggle/input")
sh(f"{PY} -c \"import pandas as pd, sys; pd.concat([pd.read_csv(m, dtype=str, keep_default_na=False) for m in sys.argv[1:]])"
   f".drop_duplicates('video_id').to_csv('{W}/manifest.csv', index=False)\" " + " ".join(manifests))
sh(f"{PY} {REPO}/scripts/train_faceswap_v2.py select --manifest {W}/manifest.csv --features-dir {feat} "
   f"--out-dir {W}/result --allow-errors")
log("K2 done")
