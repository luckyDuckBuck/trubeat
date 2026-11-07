# K1b (GPU): DF40 (face-swap training videos on FF++ train/val identities, the
# gate-1b and report-only test videos, UADFV) and Celeb-DF v2 -> faces and
# embeddings (BACKBONES) of the as-downloaded videos (v0).
setup(gpu=True)
resume_from_inputs()
TF2 = f"{PY} {REPO}/scripts/train_faceswap_v2.py"
os.chdir(REPO)
links_path = f"{DATA}/links.json"
os.makedirs(DATA, exist_ok=True)
with open(links_path, "w") as f:
    json.dump(LINKS, f)
for b in BACKBONES.split():
    sh(f"{PY} -m trubeat.cli fetch-models --backbone {b}")
sh(f"{TF2} fetch --root {DATA} --datasets df40 cdf2 --links {links_path} --threads 8")
shutil.copy(f"{DATA}/fetch_report.json", f"{W}/fetch_report_k1b.json")
os.remove(links_path)
sh(f"{TF2} manifest --ffpp {DATA}/ffpp --df40 {DATA}/df40 --cdf2 {DATA}/cdf2 --out {W}/manifest_k1b.csv")
budget = max(600, left_s() - 900)
sh(f"{TF2} extract --manifest {W}/manifest_k1b.csv --features-dir {W}/features --workers {4 if GPU else 3} "
   f"--device {'cuda' if GPU else 'cpu'} --backbones {BACKBONES} --time-budget-s {budget:.0f}")
log("K1b done")
