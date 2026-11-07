"""Shared header of the round-4 Kaggle kernels (docs/13 §2, §7).

scripts/kaggle_run.py pastes this file, then the kernel's own body, into one
script (Kaggle uploads only the code file) and fills LINKS from the local,
gitignored runs/faceswap/round4_links.json. Every step halts the kernel on
failure (docs/13 B-05); outputs go to /kaggle/working, data to scratch disk.
"""
import json
import os
import shutil
import subprocess
import sys
import time

T0 = time.time()
BRANCH = "round4"
REPO_URL = "https://github.com/luckyDuckBuck/trubeat.git"
REPO = "/tmp/trubeat"
VENV = "/tmp/venv"
PY = f"{VENV}/bin/python"
W = "/kaggle/working"
DATA = "/tmp/trubeat_data"
SESSION_S = 11.0 * 3600          # Kaggle ends sessions at 12 h; stop with an hour to spare
LINKS = json.loads(r'''__LINKS__''')
os.environ["MPLBACKEND"] = "Agg"


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} [{(time.time() - T0) / 60:6.1f} min] {msg}", flush=True)


def sh(cmd, **kw):
    log(f"$ {cmd}")
    subprocess.run(cmd, shell=True, check=True, **kw)


def left_s():
    return SESSION_S - (time.time() - T0)


def setup(gpu=True):
    """The repo at the round's branch + an isolated Python 3.12 env with the
    pinned versions the laptop runs (embeddings must match across machines)."""
    sh("nvidia-smi --query-gpu=name,memory.total --format=csv || true")
    sh("nproc; free -g; df -h /kaggle/working /tmp | cat")
    if not os.path.isdir(REPO):
        sh(f"git clone -q --depth 1 --branch {BRANCH} {REPO_URL} {REPO}")
    sh(f"git -C {REPO} log --oneline -1")
    sh("pip install -q uv")
    if not os.path.exists(PY):
        sh(f"uv venv -q {VENV} --python 3.12")
    index = "https://download.pytorch.org/whl/cu128" if gpu else "https://download.pytorch.org/whl/cpu"
    for idx in ([index] if not gpu else [index, "https://download.pytorch.org/whl/cu130", "https://download.pytorch.org/whl/cu126"]):
        if subprocess.run(f"uv pip install -q -p {PY} torch==2.15.0 torchvision --index-url {idx}", shell=True).returncode == 0:
            log(f"torch from {idx}")
            break
    else:
        raise SystemExit("could not install torch==2.15.0")
    sh(f"uv pip install -q -p {PY} -e '{REPO}[aigen]' timm==1.0.30 remotezip")
    r = subprocess.run([PY, "-c", "import torch, timm, mediapipe, numpy, cv2; "
                        "print('cuda', torch.cuda.is_available(), torch.cuda.device_count(), 'torch', torch.__version__, "
                        "'timm', timm.__version__, 'numpy', numpy.__version__, 'cv2', cv2.__version__)"],
                       capture_output=True, text=True)
    print(r.stdout, r.stderr[-2000:], flush=True)
    if r.returncode != 0:
        raise SystemExit("the training environment is broken: see above")
    if gpu and "cuda True" not in r.stdout:
        raise SystemExit("no CUDA in the training environment")


def resume_from_inputs():
    """Copy feature files from an earlier kernel's output (kernel_sources) so an
    interrupted extraction continues instead of restarting."""
    base = "/kaggle/input"
    if not os.path.isdir(base):
        return
    for d in sorted(os.listdir(base)):
        src = os.path.join(base, d, "features")
        if os.path.isdir(src):
            log(f"resuming from {src}")
            sh(f"mkdir -p {W}/features && cp -rn {src}/. {W}/features/")
