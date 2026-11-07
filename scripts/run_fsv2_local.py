"""Detector v2 on the laptop, every step (docs/13 §6.6 item 10).

Kaggle cancels this pipeline (five sessions, 2026-11-07), so it runs here,
step by step; every step can be rerun and continues where it stopped.

  env     runs/venv-cuda: the app environment's package versions plus torch
          2.15.0's CUDA build, for B/16 embeddings on the GTX 1650.
  ffpp    FF++ c23 + c40 and 450 DFD clips into data/raw/ffpp (official
          server, the user's access); manifest; extract (six variants,
          stage-2 crops).
  df40    DF40 (and UADFV, in its test/deepfacelab.zip) zip by zip from
          Google Drive through runs/fsv2/local/df40_tmp: fetch the planned
          members, manifest, extract (v0), delete those videos. Nothing under
          data/raw is deleted.
  cdf2    Celeb-DF v2's test list and reals into data/raw/cdf2; manifest;
          extract.
  select  merge the manifests; selection in the app's environment (CPU).

Usage (from the repo root, in the app's environment):
    uv run python scripts/run_fsv2_local.py env
    uv run python scripts/run_fsv2_local.py ffpp df40 cdf2 select
Outputs: runs/fsv2/local/ (manifests/, features/, crops380/, result/, run.log).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import train_faceswap_v2 as t

OUT = REPO / "runs" / "fsv2" / "local"
VENV = REPO / "runs" / "venv-cuda"
VPY = VENV / "bin" / "python"
TF2 = str(REPO / "scripts" / "train_faceswap_v2.py")
LINKS = REPO / "runs" / "faceswap" / "round4_links.json"
DATA = REPO / "data" / "raw"
TMP = OUT / "df40_tmp"
EXCLUDE = REPO / "runs" / "faceswap" / "v2_exclude.txt"
B16 = "vit_base_patch16_clip_224.openai"
WORKERS = 3          # 7 GB of RAM: three workers next to the GPU process


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "run.log", "a") as f:
        f.write(line + "\n")


def run(cmd: list) -> None:
    cmd = [str(c) for c in cmd]
    log("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO)


def need_gb(gb: float, what: str) -> None:
    free = shutil.disk_usage(REPO).free / 1e9
    if free < gb:
        raise SystemExit(f"{what} needs about {gb:.0f} GB free; {free:.0f} GB left")


def extract(manifest: Path, stage2: bool = False) -> None:
    cmd = [VPY, TF2, "extract", "--manifest", manifest, "--features-dir", OUT / "features", "--workers", WORKERS,
           "--device", "cuda", "--backbones", B16]
    if stage2:
        cmd += ["--stage2-dir", OUT / "crops380"]
    run(cmd)


def step_env() -> None:
    check = "import torch; x = torch.randn(64, 64, device='cuda'); print(torch.__version__, float((x @ x).sum()))"
    if not VPY.is_file():
        run(["uv", "venv", "-q", VENV, "--python", "3.12"])
    freeze = subprocess.run(["uv", "pip", "freeze", "-p", REPO / ".venv" / "bin" / "python"], capture_output=True,
                            text=True, check=True).stdout.splitlines()
    pins, tv = [], None
    for ln in freeze:
        name = ln.split("==")[0].strip().lower()
        if not ln or ln.startswith("-e") or " @ " in ln or name in ("trubeat", "torch"):
            continue
        if name == "torchvision":
            tv = ln.split("==")[1].split("+")[0]
            continue
        pins.append(ln)
    req = OUT / "venv_cuda_requirements.txt"
    req.parent.mkdir(parents=True, exist_ok=True)
    req.write_text("\n".join(pins) + "\n")
    run(["uv", "pip", "install", "-q", "-p", VPY, "torch==2.15.0"] + ([f"torchvision=={tv}"] if tv else ["torchvision"]))
    run(["uv", "pip", "install", "-q", "-p", VPY, "-r", req])
    run(["uv", "pip", "install", "-q", "-p", VPY, "--no-deps", "-e", REPO])
    run(["uv", "pip", "install", "-q", "-p", VPY, "remotezip"])      # range reads of the DF40 / Celeb-DF zips
    run([VPY, "-c", check])


def step_ffpp() -> None:
    need_gb(30, "FF++ c23 + c40 and DFD")
    run([sys.executable, TF2, "fetch", "--root", DATA, "--datasets", "ffpp", "--compression", "c23", "c40",
         "--threads", "4"])
    man = OUT / "manifests" / "ffpp.csv"
    t.finish_manifest(t.manifest_ffpp(DATA / "ffpp"), man)
    extract(man, stage2=True)


def step_df40() -> None:
    if not LINKS.is_file():
        raise SystemExit(f"{LINKS} is missing (the dataset links, kept out of the repo)")
    plan_path = OUT / "df40_plan.json"
    if not plan_path.is_file():  # the same picks `fetch` makes in one call (fixed seed), saved once
        run([VPY, TF2, "fetch", "--root", TMP, "--datasets", "df40", "--links", LINKS, "--df40-plan", plan_path,
             "--plan-only"])
    plan = json.loads(plan_path.read_text())
    log(f"DF40 plan: {sum(len(v) for v in plan.values())} videos from {len(plan)} zips")
    for key, members in plan.items():
        tag = key.replace("/", "_").removesuffix(".zip")
        done = OUT / "manifests" / f"df40_{tag}.done"
        if done.is_file():
            continue
        zdir = TMP / "df40" / key.removesuffix(".zip")
        if members:
            need_gb(8, key)
            run([VPY, TF2, "fetch", "--root", TMP, "--datasets", "df40", "--links", LINKS,
                 "--df40-plan", plan_path, "--only", key])
            man = OUT / "manifests" / f"df40_{tag}.csv"
            rows = [r for r in t.manifest_df40(TMP / "df40", DATA / "ffpp") if str(zdir) in r["path"]]
            df = t.finish_manifest(rows, man)
            if len(df):
                extract(man)
        if zdir.is_dir():
            shutil.rmtree(zdir)   # a streaming cache under runs/, never data/raw
        done.write_text(f"{len(members)} planned\n")
        log(f"DF40 {key}: done")


def step_cdf2() -> None:
    need_gb(4, "Celeb-DF v2 (test list and reals)")
    run([VPY, TF2, "fetch", "--root", DATA, "--datasets", "cdf2", "--links", LINKS])
    man = OUT / "manifests" / "cdf2.csv"
    t.finish_manifest(t.manifest_cdf2(DATA / "cdf2"), man)
    extract(man)


def step_select() -> None:
    parts = sorted((OUT / "manifests").glob("*.csv"))
    m = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False) for f in parts]).drop_duplicates("video_id")
    m = m.fillna("")
    allm = OUT / "manifest_all.csv"
    m.to_csv(allm, index=False)
    log(f"manifest: {len(m)} videos from {len(parts)} parts")
    cmd = [sys.executable, TF2, "select", "--manifest", allm, "--features-dir", OUT / "features",
           "--out-dir", OUT / "result", "--allow-errors"]
    if EXCLUDE.is_file() and EXCLUDE.read_text().strip():
        cmd += ["--exclude", EXCLUDE]
    run(cmd)


STEPS = {"env": step_env, "ffpp": step_ffpp, "df40": step_df40, "cdf2": step_cdf2, "select": step_select}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("steps", nargs="+", choices=[*STEPS, "all"])
    a = ap.parse_args()
    steps = list(STEPS) if "all" in a.steps else a.steps
    for s in steps:
        log(f"== step {s}")
        STEPS[s]()
        log(f"== step {s} done")


if __name__ == "__main__":
    main()
