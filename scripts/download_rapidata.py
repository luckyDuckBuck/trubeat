"""Fetch Rapidata text-to-video clips of current generators (docs/11 §9.7).

Rapidata's human-preference sets on Hugging Face (Apache-2.0, ungated) ship
the MP4s they asked people to compare. The repos share one ~100-prompt
benchmark, so the same baseline clip (e.g. ``ray2`` prompt 0099 take 1)
appears in several repos under different paths; clips are de-duplicated by
(generator, prompt, take). GIF-only sets (Sora 2, Kling 2.1, ...) are not
used: a GIF palette would teach the detector "GIF = AI".

Public download only (AGENTS.md §3); AI-generated clips, no real faces of
identifiable people beyond what the generators produced. Files land in
``data/raw/aigen/rapidata/`` (gitignored); rows go to
``data/manifests/aigen_rapidata.csv`` with ``split_group`` = prompt index,
so clips of one prompt never straddle a cross-validation fold.

Usage:
    python scripts/download_rapidata.py [--per-generator 200] [--threads 8]
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data/raw/aigen/rapidata"
REPOS = [f"Rapidata/text-2-video-human-preferences{s}" for s in (
    "-veo3.1", "-seedance-1-pro", "-moonvalley-marey", "-veo3", "-pika2.2", "-wan2.1",
    "-veo2", "-genmo-mochi-1", "")]
CANON = {"alpha": "gen3-alpha", "veo31": "veo3.1", "hunyuan": "hunyuan"}


def parse(filename: str) -> tuple[str, int, str] | None:
    """'videos/0062_ray2_1.mp4' / 'Videos/ray2/ray2_0099_1.mp4' /
    'videos/veo31_49_0.mp4' -> (generator, prompt index, take)."""
    stem = Path(filename).stem
    tok = stem.split("_")
    if len(tok) < 3:
        return None
    if tok[0].isdigit():
        idx, model, take = tok[0], "_".join(tok[1:-1]), tok[-1]
    elif tok[-2].isdigit():
        idx, model, take = tok[-2], "_".join(tok[:-2]), tok[-1]
    else:
        return None
    model = re.sub(r"-\d{1,2}-\d{1,2}-\d{2}$", "", model.lower())  # drop "-24-7-25" dates
    return CANON.get(model, model), int(idx), take


def list_clips() -> pd.DataFrame:
    from huggingface_hub import HfApi

    api = HfApi()
    rows, seen = [], set()
    for rid in REPOS:
        info = api.dataset_info(rid, files_metadata=True)
        for s in info.siblings:
            if not s.rfilename.lower().endswith(".mp4"):
                continue
            key = parse(s.rfilename)
            if key is None or key in seen:
                continue
            seen.add(key)
            rows.append({"repo": rid, "file": s.rfilename, "generator": key[0],
                         "prompt": key[1], "take": key[2], "size": s.size or 0})
    return pd.DataFrame(rows)


def fetch(repo: str, file: str) -> Path | None:
    from huggingface_hub import hf_hub_download

    dest_dir = OUT / repo.split("/")[1]
    target = dest_dir / file
    if target.is_file() and target.stat().st_size > 0:
        return target
    for attempt in range(3):
        try:
            return Path(hf_hub_download(repo, file, repo_type="dataset", local_dir=dest_dir))
        except Exception as exc:  # noqa: BLE001 - retry, then skip this clip
            if attempt == 2:
                print(f"skip {repo}/{file}: {type(exc).__name__}", flush=True)
            time.sleep(2 * (attempt + 1))
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--per-generator", type=int, default=200)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--manifest", default=str(REPO / "data/manifests/aigen_rapidata.csv"))
    args = ap.parse_args()
    clips = list_clips()
    clips = (clips.sample(frac=1.0, random_state=7)
             .groupby("generator", sort=True).head(args.per_generator)
             .sort_values(["generator", "prompt", "take"]).reset_index(drop=True))
    print(clips.groupby("generator").agg(n=("file", "size"), mb=("size", lambda x: x.sum() / 1e6))
          .round(0).to_string(), flush=True)
    print(f"total {len(clips)} clips, {clips['size'].sum() / 1e9:.2f} GB", flush=True)
    paths: dict[int, Path] = {}
    t0 = time.time()
    with ThreadPoolExecutor(args.threads) as ex:
        futs = {ex.submit(fetch, r.repo, r.file): i for i, r in clips.iterrows()}
        for k, f in enumerate(as_completed(futs), 1):
            p = f.result()
            if p is not None:
                paths[futs[f]] = p
            if k % 100 == 0:
                print(f"  {k}/{len(clips)} in {time.time() - t0:.0f}s", flush=True)
    clips = clips.loc[sorted(paths)].assign(path=[str(paths[i].relative_to(REPO)) for i in sorted(paths)])
    man = pd.DataFrame({
        "path": clips.path, "label": "ai", "generator": clips.generator, "source": "rapidata",
        "family": "rapidata", "split_group": [f"rapidata:prompt:{p:04d}" for p in clips.prompt],
    })
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    man.to_csv(args.manifest, index=False)
    print(f"{len(man)} rows -> {args.manifest}")


if __name__ == "__main__":
    sys.exit(main())
