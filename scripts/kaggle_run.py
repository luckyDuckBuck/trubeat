"""Push, watch and pull the round-4 Kaggle kernels (docs/13 §2, §7).

A kernel is notebooks/kaggle/common.py + notebooks/kaggle/<body>.py joined
into one private script (Kaggle uploads only the code file). Download links
come from the local, gitignored runs/faceswap/round4_links.json and live only
in the private kernel, never in the repo.

Usage:
    python scripts/kaggle_run.py push k1 [--resume-from trubeat-r4-k1]
    python scripts/kaggle_run.py status k1
    python scripts/kaggle_run.py wait k1 [--poll 300]
    python scripts/kaggle_run.py pull k1 --dest runs/fsv2/k1 [--pattern 'manifest|report|extract_log']
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
KERNELS = {
    "k1": {"body": "k1_ffpp.py", "slug": "trubeat-r4-k1", "gpu": True, "links": False, "sources": []},
    "k1b": {"body": "k1b_more.py", "slug": "trubeat-r4-k1b", "gpu": True, "links": True, "sources": []},
    "k2": {"body": "k2_select.py", "slug": "trubeat-r4-k2", "gpu": False, "links": False,
           "sources": ["trubeat-r4-k1", "trubeat-r4-k1b"]},
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
    code = (KDIR / "common.py").read_text().replace("__LINKS__", links) + "\n\n" + (KDIR / k["body"]).read_text()
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
    ap.add_argument("kernel", choices=sorted(KERNELS))
    ap.add_argument("--slug", default=None, help="push under another slug (e.g. a continuation run)")
    ap.add_argument("--resume-from", nargs="*", default=[], help="earlier kernels whose features to continue from")
    ap.add_argument("--dest", default=None)
    ap.add_argument("--pattern", default=None)
    ap.add_argument("--poll", type=int, default=300)
    a = ap.parse_args()
    if a.action in ("push", "build"):
        d = build(a.kernel, a.resume_from, a.slug)
        print(f"built {d}")
        if a.action == "push":
            print(kaggle("kernels", "push", "-p", str(d)))
    elif a.action == "status":
        print(status(a.kernel, a.slug))
    elif a.action == "wait":
        while True:
            s = status(a.kernel, a.slug)
            print(time.strftime("%H:%M:%S"), s, flush=True)
            if any(x in s.upper() for x in ("COMPLETE", "ERROR", "CANCEL")):
                break
            time.sleep(a.poll)
    elif a.action == "pull":
        dest = Path(a.dest or REPO / "runs" / "fsv2" / a.kernel)
        dest.mkdir(parents=True, exist_ok=True)
        args = ["kernels", "output", ref(a.kernel, a.slug), "-p", str(dest), "-o"]
        if a.pattern:
            args += ["--file-pattern", a.pattern]
        print(kaggle(*args))


if __name__ == "__main__":
    sys.exit(main())
