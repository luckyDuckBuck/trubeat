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

While Google Drive refuses downloads (its "Quota exceeded" notice, which
can last up to 24 h: every DF40 zip from 06:12 on 2026-11-08), the df40 and
cdf2 steps wait, one 100-byte probe per half hour for up to 36 h, then go
on. The df40 step stops the run if any zip is still incomplete, so select
never runs on part of DF40.

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


DRIVE_WAIT_S = 30 * 60        # Drive's "Quota exceeded" can last up to 24 h: look again every half hour
DRIVE_MAX_WAIT_S = 36 * 3600


def fetch(cmd: list, url: str, what: str) -> None:
    """Run a `fetch` call that reads from Google Drive. While Drive refuses
    downloads (a failed 100-byte probe, or `fetch` exiting with QUOTA_EXIT),
    wait and try again, one probe per half hour, up to 36 h; any other
    failure stops the run."""
    t0 = time.time()
    while True:
        if t.drive_ready(url):
            cmd = [str(c) for c in cmd]
            log("$ " + " ".join(cmd))
            code = subprocess.run(cmd, cwd=REPO, check=False).returncode
            if code == 0:
                return
            if code != t.QUOTA_EXIT:
                raise SystemExit(f"{what}: fetch failed (exit {code})")
        if time.time() - t0 > DRIVE_MAX_WAIT_S:
            raise SystemExit(f"{what}: Google Drive still refuses downloads after {DRIVE_MAX_WAIT_S // 3600} h")
        log(f"{what}: Google Drive refuses downloads for now (quota exceeded); next try in {DRIVE_WAIT_S // 60} min")
        time.sleep(DRIVE_WAIT_S)


def need_gb(gb: float, what: str) -> None:
    free = shutil.disk_usage(REPO).free / 1e9
    if free < gb:
        raise SystemExit(f"{what} needs about {gb:.0f} GB free; {free:.0f} GB left")


def extract(manifest: Path, stage2: bool = False) -> None:
    """Extract, resuming after a worker died (exit 3: most likely out of
    memory) with one worker fewer each time; any other failure stops."""
    workers = WORKERS
    for attempt in range(4):
        cmd = [VPY, TF2, "extract", "--manifest", manifest, "--features-dir", OUT / "features", "--workers", workers,
               "--device", "cuda", "--backbones", B16]
        if stage2:
            cmd += ["--stage2-dir", OUT / "crops380"]
        cmd = [str(c) for c in cmd]
        log("$ " + " ".join(cmd))
        code = subprocess.run(cmd, cwd=REPO, check=False).returncode
        if code == 0:
            return
        if code == 3 and attempt < 3:
            workers = max(1, workers - 1)
            log(f"extract: a worker process died; resuming with {workers} worker(s)")
            continue
        raise SystemExit(f"extract failed (exit {code}) on {manifest.name}")


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
    run(["uv", "pip", "install", "-q", "-p", VPY, "-r", req])
    run(["uv", "pip", "install", "-q", "-p", VPY, "--no-deps", "-e", REPO])
    run(["uv", "pip", "install", "-q", "-p", VPY, "remotezip"])      # range reads of the DF40 / Celeb-DF zips
    # torch 2.15.0 (the app's version; PyPI's build is CUDA 13.2), else 2.14.1+cu126 for an older driver
    for spec, idx in ((["torch==2.15.0"] + ([f"torchvision=={tv}"] if tv else ["torchvision"]), []),
                      (["torch==2.14.1", "torchvision"], ["--index-url", "https://download.pytorch.org/whl/cu126"])):
        run(["uv", "pip", "install", "-q", "-p", VPY, *spec, *idx])
        if subprocess.run([str(VPY), "-c", check], cwd=REPO, check=False).returncode == 0:
            log(f"env: torch {spec[0]} runs on the GPU")
            return
        log(f"env: {spec[0]} cannot run on this GPU; trying the next build")
    raise SystemExit("no torch build runs on this GPU")


def step_ffpp() -> None:
    need_gb(30, "FF++ c23 + c40 and DFD")
    # the TUM server gives each connection ~0.6 MB/s from here: 8 at a time
    run([sys.executable, TF2, "fetch", "--root", DATA, "--datasets", "ffpp", "--compression", "c23", "c40",
         "--threads", "8"])
    man = OUT / "manifests" / "ffpp.csv"
    t.finish_manifest(t.manifest_ffpp(DATA / "ffpp"), man)
    extract(man, stage2=True)


def step_df40() -> None:
    if not LINKS.is_file():
        raise SystemExit(f"{LINKS} is missing (the dataset links, kept out of the repo)")
    plan_path = OUT / "df40_plan.json"
    ids = json.loads(LINKS.read_text())["df40_ids"]
    if not plan_path.is_file():  # the same picks `fetch` makes in one call (fixed seed), saved once
        fetch([VPY, TF2, "fetch", "--root", TMP, "--datasets", "df40", "--links", LINKS, "--df40-plan", plan_path,
               "--plan-only"], t.drive_url(next(iter(ids.values()))), "DF40 plan")
    plan = json.loads(plan_path.read_text())
    log(f"DF40 plan: {sum(len(v) for v in plan.values())} videos from {len(plan)} zips")
    incomplete = []
    for key, members in plan.items():
        tag = key.replace("/", "_").removesuffix(".zip")
        done = OUT / "manifests" / f"df40_{tag}.done"
        if done.is_file():
            continue
        zdir = TMP / "df40" / key.removesuffix(".zip")
        missing: list[str] = []
        failed: list[str] = []
        if members:
            need_gb(8, key)
            for attempt in range(3):      # Drive may refuse reads for a while (503): pause and fetch again
                fetch([VPY, TF2, "fetch", "--root", TMP, "--datasets", "df40", "--links", LINKS,
                       "--df40-plan", plan_path, "--only", key], t.drive_url(ids[key]), f"DF40 {key}")
                missing = [m for m in members if not (zdir / m).is_file()]
                if not missing:
                    break
                log(f"DF40 {key}: {len(missing)} of {len(members)} planned clips missing; retry in {60 * 3 ** attempt} s")
                time.sleep(60 * 3 ** attempt)
            man = OUT / "manifests" / f"df40_{tag}.csv"
            rows = [r for r in t.manifest_df40(TMP / "df40", DATA / "ffpp") if str(zdir) in r["path"]]
            df = t.finish_manifest(rows, man)
            if len(df):
                extract(man)
            for r in df.itertuples(index=False):   # delete each clip once its features are complete
                if t.extracted_v2(OUT / "features", r.video_id, [B16]):
                    Path(r.path).unlink(missing_ok=True)
                else:
                    failed.append(r.video_id)
        if missing or failed:   # kept on disk and not marked done: a rerun of this step retries them
            incomplete.append({"zip": key, "missing": len(missing), "not_extracted": len(failed)})
            log(f"DF40 {key}: NOT done ({len(missing)} clips missing, {len(failed)} not fully extracted); "
                "rerun the df40 step to retry")
            continue
        if zdir.is_dir():
            shutil.rmtree(zdir)   # a streaming cache under runs/, never data/raw
        done.write_text(f"{len(members)} planned\n")
        log(f"DF40 {key}: done")
    (OUT / "df40_incomplete.json").write_text(json.dumps(incomplete, indent=1))
    if incomplete:   # stop here: the steps after this one (select) must not run on part of DF40
        raise SystemExit(f"DF40: {len(incomplete)} zips incomplete (runs/fsv2/local/df40_incomplete.json); "
                         "rerun the df40 step")


def step_cdf2() -> None:
    need_gb(4, "Celeb-DF v2 (test list and reals)")
    fetch([VPY, TF2, "fetch", "--root", DATA, "--datasets", "cdf2", "--links", LINKS],
          t.drive_url(json.loads(LINKS.read_text())["cdf2_id"]), "Celeb-DF v2")
    man = OUT / "manifests" / "cdf2.csv"
    t.finish_manifest(t.manifest_cdf2(DATA / "cdf2"), man)
    extract(man)


def step_select() -> None:
    parts = sorted((OUT / "manifests").glob("*.csv"))
    m = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False) for f in parts]).fillna("")
    # no de-duplication here: the parts are disjoint by construction, and select's leakage check
    # (check_manifest_v2) fails on any video id listed twice
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
        try:
            STEPS[s]()
        except SystemExit as exc:   # recorded in run.log, and the steps after it do not run
            log(f"== step {s} stopped: {exc}")
            raise
        log(f"== step {s} done")


if __name__ == "__main__":
    main()
