"""Fetch a subset of public AI-generated-video datasets (docs/11 D1).

Sources (verified anonymous / ungated on 2026-11-01 via the HF API):

* AIGVDBench (HF ``AIGVDBench/AIGVDBench``, CC-BY-4.0): one zip per
  generator, every member an mp4. Clips are pulled *out of the remote zip*
  with HTTP range requests, so a 10 GB generator zip costs only the clips
  taken. Open-source fakes are named by their OpenVid source clip id, and
  the real clip of the same id is fetched too (content-matched pairs; the
  pair shares one ``split_group``).
* GenBuster-Bench (HF ``l8cv/GenBuster-Bench``, MIT): one 2.2 GB zip with
  real clips, in-distribution, out-of-distribution and in-the-wild fakes
  (Sora 2, Veo 3.1, Kling 2.5/2.6, ...). Downloaded whole.

Writes clips under ``data/raw/aigen/`` and the manifest
``data/manifests/aigen.csv`` (path, label, generator, source, family,
split_group). Resumable: existing files are skipped.

Usage:
    python scripts/download_aigen_data.py [--per-generator 100] [--extra-real 300]
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[1]
HF = "https://huggingface.co/datasets"
AIGVD = "AIGVDBench/AIGVDBench"
GENBUSTER = "l8cv/GenBuster-Bench"
OUT = REPO / "data" / "raw" / "aigen"
MANIFEST = REPO / "data" / "manifests" / "aigen.csv"
UA = {"User-Agent": "trubeat-research-downloader/0.1"}


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


class HTTPRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests (for zipfile)."""

    def __init__(self, url: str) -> None:
        self.url = url
        self._resolve()
        self.pos = 0

    def _resolve(self) -> None:
        r = requests.head(self.url, allow_redirects=True, headers=UA, timeout=60)
        r.raise_for_status()
        self.final = r.url
        self.size = int(r.headers["Content-Length"])

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        else:
            self.pos = self.size + offset
        return self.pos

    def readinto(self, b) -> int:
        n = len(b)
        if self.pos >= self.size or n == 0:
            return 0
        end = min(self.pos + n, self.size) - 1
        for attempt in range(6):
            try:
                r = requests.get(
                    self.final,
                    headers={**UA, "Range": f"bytes={self.pos}-{end}"},
                    timeout=120,
                )
                if r.status_code in (401, 403):  # signed CDN URL expired
                    self._resolve()
                    continue
                r.raise_for_status()
                data = r.content
                b[: len(data)] = data
                self.pos += len(data)
                return len(data)
            except requests.RequestException as exc:
                wait = 2**attempt
                log(f"  range read failed ({exc}); retry in {wait}s")
                time.sleep(wait)
        raise OSError(f"range read failed repeatedly: {self.url}")


def open_remote_zip(path_in_repo: str, repo: str = AIGVD) -> zipfile.ZipFile:
    raw = HTTPRangeFile(f"{HF}/{repo}/resolve/main/{path_in_repo}")
    return zipfile.ZipFile(io.BufferedReader(raw, buffer_size=4 << 20))


def extract_member(zf: zipfile.ZipFile, name: str, dest: Path) -> bool:
    if dest.is_file() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with zf.open(name) as src, open(tmp, "wb") as dst:
            while chunk := src.read(4 << 20):
                dst.write(chunk)
        tmp.replace(dest)
        return True
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        log(f"  could not extract {name}: {exc}")
        tmp.unlink(missing_ok=True)
        return False


def aigvd_zips() -> list[str]:
    j = requests.get(f"https://huggingface.co/api/datasets/{AIGVD}", headers=UA, timeout=60).json()
    return sorted(
        s["rfilename"] for s in j["siblings"]
        if s["rfilename"].endswith(".zip") and "/Real/" not in s["rfilename"]
    )


def test_split_ids() -> set[str]:
    r = requests.get(f"{HF}/{AIGVD}/resolve/main/Split/test.jsonl", headers=UA, timeout=120)
    r.raise_for_status()
    return {Path(json.loads(line)["Video_id"]).stem for line in r.text.splitlines() if line.strip()}


def generator_name(zip_path: str) -> tuple[str, str]:
    parts = Path(zip_path).parts  # AIGVDBench/<ClosedSource|OpenSource>/[T2V/]X.zip
    stem = Path(zip_path).stem
    if parts[1] == "ClosedSource":
        return stem.lower().replace("_24fps", ""), "closed"
    return f"{stem.lower()}_{parts[2].lower()}", parts[2].lower()


def _zip_seed(seed: int, name: str) -> int:
    """Deterministic per-zip seed, so parallel workers sample reproducibly."""
    return seed * 1_000_003 + sum((i + 1) * ord(c) for i, c in enumerate(name))


def _fetch_one_zip(zp: str, per_gen: int, test_ids: set[str], seed: int) -> list[dict]:
    gen, family = generator_name(zp)
    dest_dir = OUT / "aigvdbench" / gen
    try:
        zf = open_remote_zip(zp)
    except Exception as exc:  # noqa: BLE001 - one bad zip must not stop the run
        log(f"skip {zp}: {exc}")
        return []
    mp4s = [n for n in zf.namelist() if n.lower().endswith(".mp4") and "__MACOSX" not in n]
    by_base = {Path(n).name: n for n in mp4s}
    # Resume: clips already on disk count toward the quota first.
    have = [by_base[f.name] for f in sorted(dest_dir.glob("*.mp4")) if f.name in by_base]
    in_test = [n for n in mp4s if Path(n).stem in test_ids and n not in have]
    rest = [n for n in mp4s if n not in have]
    pool = in_test if len(in_test) >= per_gen - len(have) else rest
    rng = random.Random(_zip_seed(seed, zp))
    need = max(0, per_gen - len(have))
    pick = have[:per_gen] + rng.sample(pool, min(need, len(pool)))
    log(f"{zp}: {len(mp4s)} clips ({len(in_test)} in test split), {len(have)} on disk, "
        f"taking {len(pick)}")
    rows = []
    for name in pick:
        dest = dest_dir / Path(name).name
        if not extract_member(zf, name, dest):
            continue
        src_id = Path(name).stem
        paired = family != "closed" and src_id in test_ids
        rows.append({
            "path": str(dest.relative_to(REPO)), "label": "ai", "generator": gen,
            "source": "aigvdbench", "family": family,
            "split_group": f"openvid:{src_id}" if paired else f"aigvd:{gen}:{src_id}",
        })
    log(f"done {zp}: {len(rows)} clips")
    return rows


def _fetch_real_chunk(names: list[str]) -> list[dict]:
    zf = open_remote_zip("AIGVDBench/Real/Real.zip")
    rows = []
    for name in names:
        dest = OUT / "aigvdbench" / "real" / Path(name).name
        if not extract_member(zf, name, dest):
            continue
        rows.append({
            "path": str(dest.relative_to(REPO)), "label": "real", "generator": "real",
            "source": "aigvdbench", "family": "real",
            "split_group": f"openvid:{Path(name).stem}",
        })
    return rows


def fetch_aigvd(per_gen: int, extra_real: int, seed: int, threads: int) -> list[dict]:
    from concurrent.futures import ThreadPoolExecutor

    test_ids = test_split_ids()
    log(f"AIGVDBench test split: {len(test_ids)} source ids; {threads} parallel zips")
    rows: list[dict] = []
    with ThreadPoolExecutor(threads) as ex:
        for part in ex.map(lambda z: _fetch_one_zip(z, per_gen, test_ids, seed), aigvd_zips()):
            rows += part
    # Real clips: the OpenVid sources of the fetched open-source fakes first
    # (content-matched pairs), then extras.
    wanted = {r["split_group"].split(":", 1)[1] for r in rows if r["split_group"].startswith("openvid:")}
    zf = open_remote_zip("AIGVDBench/Real/Real.zip")
    real_names = {Path(n).stem: n for n in zf.namelist() if n.lower().endswith(".mp4")}
    others = sorted(set(real_names) - wanted)
    rng = random.Random(seed)
    picks = sorted(wanted & set(real_names)) + rng.sample(others, min(extra_real, len(others)))
    log(f"Real.zip: {len(real_names)} clips; taking {len(picks)} "
        f"({len(wanted & set(real_names))} paired with fetched fakes)")
    names = [real_names[sid] for sid in picks]
    chunks = [names[i::threads] for i in range(threads)]
    with ThreadPoolExecutor(threads) as ex:
        for part in ex.map(_fetch_real_chunk, chunks):
            rows += part
    return rows


def fetch_genbuster() -> list[dict]:
    from huggingface_hub import hf_hub_download

    root = OUT / "genbuster"
    on_disk = sorted(root.glob("GenBuster-Bench/**/*.mp4"))
    if len(on_disk) >= 3000:  # already extracted by an earlier run
        names = [str(f.relative_to(root)) for f in on_disk]
        zpath = None
    else:
        zpath = Path(hf_hub_download(GENBUSTER, "GenBuster-Bench.zip", repo_type="dataset",
                                     local_dir=root / "_zip"))
        with zipfile.ZipFile(zpath) as zf:
            names = [n for n in zf.namelist()
                     if n.lower().endswith(".mp4") and "__MACOSX" not in n]
            for name in names:
                dest = root / name
                if not (dest.is_file() and dest.stat().st_size > 0):
                    extract_member(zf, name, dest)
    rows: list[dict] = []
    for name in names:
        dest = root / name
        parts = [p.lower() for p in Path(name).parts]
        label = "real" if "real" in parts else "ai"
        # .../<id|ood|wild>/<real|fake>/<generator?>/clip.mp4
        tier = next((p for p in parts if p in ("id", "ood", "wild")), "unknown")
        gen = "real"
        if label == "ai":
            i = parts.index("fake") if "fake" in parts else len(parts) - 2
            gen = parts[i + 1] if i + 1 < len(parts) - 1 else "unknown"
        rows.append({
            "path": str(dest.relative_to(REPO)), "label": label,
            "generator": gen if label == "ai" else "real",
            "source": f"genbuster_{tier}", "family": tier,
            "split_group": f"genbuster:{tier}:{gen}:{Path(name).stem}",
        })
    if zpath is not None:
        zpath.unlink(missing_ok=True)  # extracted; keep disk free
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--per-generator", type=int, default=100)
    ap.add_argument("--extra-real", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--skip-genbuster", action="store_true")
    ap.add_argument("--skip-aigvd", action="store_true")
    ap.add_argument("--threads", type=int, default=8, help="parallel remote zips")
    args = ap.parse_args()
    rows: list[dict] = []
    if not args.skip_genbuster:
        log("GenBuster-Bench: downloading the 2.2 GB zip")
        rows += fetch_genbuster()
        log(f"GenBuster-Bench: {len(rows)} clips")
    if not args.skip_aigvd:
        rows += fetch_aigvd(args.per_generator, args.extra_real, args.seed, args.threads)
    df = pd.DataFrame(rows).drop_duplicates("path")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    if MANIFEST.is_file():  # merge with an earlier partial run
        old = pd.read_csv(MANIFEST)
        df = pd.concat([old, df]).drop_duplicates("path", keep="last")
    df.to_csv(MANIFEST, index=False)
    log(f"manifest: {len(df)} rows -> {MANIFEST}")
    print(df.groupby(["source", "label"]).size().to_string())


if __name__ == "__main__":
    sys.exit(main())
