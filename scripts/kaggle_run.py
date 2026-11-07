"""Push, watch and pull the round-4 Kaggle kernels (docs/13 §2, §7).

A kernel is notebooks/kaggle/common.py + notebooks/kaggle/<body>.py joined
into one private script (Kaggle uploads only the code file). Download links
come from the local, gitignored runs/faceswap/round4_links.json and live only
in the private kernel, never in the repo.

Usage:
    python scripts/kaggle_run.py push k1c0 k1c1 k1c2 k1c3 k1c4 k1b   (K1 is sharded: K1C_SHARDS sessions)
    python scripts/kaggle_run.py push k1c2 --slug trubeat-r4-k1c2b --resume-from trubeat-r4-k1c2   (a continuation)
    python scripts/kaggle_run.py status k1c0 k1c1 k1c2 k1c3 k1c4 k1b
    python scripts/kaggle_run.py wait k2 [--poll 300]
    python scripts/kaggle_run.py pull k2 --dest runs/fsv2/k2 [--pattern 'manifest|report|extract_log']
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KDIR = REPO / "notebooks" / "kaggle"
LINKS = REPO / "runs" / "faceswap" / "round4_links.json"
B16 = "vit_base_patch16_clip_224.openai"
# FF++ + DFD in CPU sessions: Kaggle cancelled every GPU session that started downloading FF++
# (3 times, 2026-11-07). ~93 vCPU-hours of work in total (six variants, faces, B/16 on CPU).
K1C_SHARDS = 5
KERNELS = {
    **{f"k1c{i}": {"body": "k1_ffpp.py", "slug": f"trubeat-r4-k1c{i}", "gpu": False, "links": False, "sources": [],
                   "shard": f"{i}/{K1C_SHARDS}", "backbones": B16} for i in range(K1C_SHARDS)},
    "k1b": {"body": "k1b_more.py", "slug": "trubeat-r4-k1b", "gpu": True, "links": True, "sources": [],
            "backbones": B16},
    "k2": {"body": "k2_select.py", "slug": "trubeat-r4-k2", "gpu": False, "links": False, "backbones": B16,
           "sources": [f"trubeat-r4-k1c{i}" for i in range(K1C_SHARDS)] + ["trubeat-r4-k1b"]},
}


def kaggle(*args: str, check: bool = True) -> str:
    r = subprocess.run(["kaggle", *args], capture_output=True, text=True, check=False)
    if check and r.returncode != 0:
        raise SystemExit(f"kaggle {' '.join(args)} failed: {r.stdout}{r.stderr}")
    return r.stdout + r.stderr


def username() -> str:
    m = re.search(r"username:\s*(\S+)", kaggle("config", "view"))
    if not m:
        raise SystemExit("no Kaggle username configured (kaggle config view)")
    return m.group(1)


def build(name: str, resume_from: list[str], slug: str | None) -> Path:
    k = KERNELS[name]
    links = "{}"
    if k["links"]:
        if not LINKS.is_file():
            raise SystemExit(f"{LINKS} is missing (the dataset links, kept out of the repo)")
        links = LINKS.read_text()
    common = (KDIR / "common.py").read_text().replace("__LINKS__", links).replace("__SHARD__", k.get("shard", ""))
    common = common.replace("__GPU__", "1" if k["gpu"] else "0").replace("__BACKBONES__", k["backbones"])
    code = common + "\n\n" + (KDIR / k["body"]).read_text()
    user = username()
    d = Path(tempfile.mkdtemp(prefix=f"kaggle_{name}_"))
    (d / f"{name}.py").write_text(code)
    meta = {
        "id": f"{user}/{slug or k['slug']}", "title": slug or k["slug"], "code_file": f"{name}.py",
        "language": "python", "kernel_type": "script", "is_private": "true",
        "enable_gpu": "true" if k["gpu"] else "false", "enable_internet": "true",
        "dataset_sources": [], "competition_sources": [], "model_sources": [],
        "kernel_sources": [f"{user}/{s}" for s in (k["sources"] + resume_from)],
    }
    if k["gpu"]:
        meta["machine_shape"] = "NvidiaTeslaT4"
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    return d


def ref(name: str, slug: str | None) -> str:
    return f"{username()}/{slug or KERNELS[name]['slug']}"


def status(name: str, slug: str | None) -> str:
    out = kaggle("kernels", "status", ref(name, slug), check=False)
    m = re.search(r'status "?([A-Za-z_.]+)"?', out)
    return m.group(1) if m else out.strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=["push", "status", "wait", "pull", "build"])
    ap.add_argument("kernels", nargs="+", choices=sorted(KERNELS))
    ap.add_argument("--slug", default=None, help="push under another slug (e.g. a continuation run); one kernel only")
    ap.add_argument("--resume-from", nargs="*", default=[], help="earlier kernels whose features to continue from")
    ap.add_argument("--dest", default=None)
    ap.add_argument("--pattern", default=None)
    ap.add_argument("--poll", type=int, default=300)
    a = ap.parse_args()
    if a.slug and len(a.kernels) > 1:
        raise SystemExit("--slug applies to one kernel at a time")
    if a.action in ("push", "build"):
        for name in a.kernels:
            d = build(name, a.resume_from, a.slug)
            print(f"built {d}")
            if a.action == "push":
                print(kaggle("kernels", "push", "-p", str(d)))
    elif a.action == "status":
        for name in a.kernels:
            print(name, status(name, a.slug))
    elif a.action == "wait":  # until every named kernel has ended
        while True:
            st = {name: status(name, a.slug) for name in a.kernels}
            print(time.strftime("%H:%M:%S"), " ".join(f"{n}={s}" for n, s in st.items()), flush=True)
            if all(any(x in s.upper() for x in ("COMPLETE", "ERROR", "CANCEL")) for s in st.values()):
                break
            time.sleep(a.poll)
    elif a.action == "pull":
        for name in a.kernels:
            dest = Path(a.dest or REPO / "runs" / "fsv2" / name)
            dest.mkdir(parents=True, exist_ok=True)
            args = ["kernels", "output", ref(name, a.slug), "-p", str(dest), "-o"]
            if a.pattern:
                args += ["--file-pattern", a.pattern]
            print(kaggle(*args))


if __name__ == "__main__":
    sys.exit(main())
