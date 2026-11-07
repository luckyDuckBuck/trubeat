"""Shared header of the round-4 Kaggle kernels (docs/13 §2, §7).

scripts/kaggle_run.py pastes this file, then the kernel's own body, into one
script (Kaggle uploads only the code file) and fills LINKS from the local,
gitignored runs/faceswap/round4_links.json. Every step halts the kernel on
failure (docs/13 B-05); outputs go to /kaggle/working, data to scratch disk.
"""
import json
import os
import shutil  # noqa: F401 - used by the kernel bodies
import subprocess
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
SHARD = "__SHARD__"              # "I/N" for a K1 shard (scripts/kaggle_run.py), else empty
GPU = "__GPU__" == "1"           # the session has GPUs
BACKBONES = "__BACKBONES__"      # the backbones this kernel embeds with (space-separated timm names)
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
    sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true")
    sh("nproc; free -g; df -h /kaggle/working /tmp | cat")
    if not os.path.isdir(REPO):
        sh(f"git clone -q --depth 1 --branch {BRANCH} {REPO_URL} {REPO}")
    sh(f"git -C {REPO} log --oneline -1")
    sh("pip install -q uv")
    if not os.path.exists(PY):
        sh(f"uv venv -q {VENV} --python 3.12")
    # The laptop runs torch 2.15.0. On GPU, PyPI's 2.15.0 is a CUDA 13.2 build (driver >= 580); when it cannot run
    # a kernel on this GPU, fall back to 2.14.1+cu126. The version used is written to extract_log.json.
    if gpu:
        candidates = [("torch==2.15.0 torchvision", ""),
                      ("torch==2.14.1 torchvision", "--index-url https://download.pytorch.org/whl/cu126")]
    else:
        candidates = [("torch==2.15.0 torchvision", "--index-url https://download.pytorch.org/whl/cpu")]
    check = ("import torch; x = torch.randn(64, 64, device='cuda'); print(torch.__version__, float((x @ x).sum()))"
             if gpu else "import torch; print(torch.__version__)")
    for spec, idx in candidates:
        if subprocess.run(f"uv pip install -q -p {PY} {spec} {idx}", shell=True, check=False).returncode != 0:
            continue
        r = subprocess.run([PY, "-c", check], capture_output=True, text=True, check=False)
        if r.returncode == 0:
            log(f"torch ok: {spec} {idx} -> {r.stdout.strip()}")
            break
        log(f"torch {spec} {idx} cannot run here: {r.stderr.strip()[-300:]}")
        subprocess.run(f"uv pip uninstall -q -p {PY} torch torchvision", shell=True, check=False)
    else:
        raise SystemExit("could not install a working torch")
    sh(f"uv pip install -q -p {PY} -e '{REPO}[aigen]' timm==1.0.30 remotezip")
    probe = ("import torch, timm, mediapipe, numpy, cv2; "
             "print('cuda', torch.cuda.is_available(), torch.cuda.device_count(), 'torch', torch.__version__, "
             "'timm', timm.__version__, 'numpy', numpy.__version__, 'cv2', cv2.__version__)")
    r = subprocess.run([PY, "-c", probe], capture_output=True, text=True, check=False)
    print(r.stdout, r.stderr[-2000:], flush=True)
    if r.returncode != 0:
        raise SystemExit("the training environment is broken: see above")
    if gpu and "cuda True" not in r.stdout:
        raise SystemExit("no CUDA in the training environment")


def input_dirs(name):
    """Every directory called ``name`` under /kaggle/input (kernel outputs are
    mounted there; the depth has changed over time, so search)."""
    import glob
    return sorted(d for d in glob.glob(f"/kaggle/input/**/{name}", recursive=True) if os.path.isdir(d))


def resume_from_inputs():
    """Copy feature files from an earlier kernel's output (kernel_sources) so an
    interrupted extraction continues instead of restarting."""
    for src in input_dirs("features"):
        log(f"resuming from {src}")
        sh(f"mkdir -p {W}/features && cp -rn {src}/. {W}/features/")
