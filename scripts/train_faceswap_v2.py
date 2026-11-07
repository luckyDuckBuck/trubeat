"""Face-swap detector v2 (docs/13): processing-invariant training, modern
swap methods, small-face finding, and the selection and gates fixed in
docs/13 §6. Runs on Kaggle (driven by scripts/kaggle_run.py) and locally.

Commands (each resumable; run in order):

  fetch      FaceForensics++ c23 + c40 and DeepFakeDetection from the official
             servers; DF40 and Celeb-DF v2 members straight out of their zips
             on Google Drive by HTTP range requests (only the videos needed).
  manifest   one CSV for every dataset: video_id, dataset, family, label,
             path, path_c40, split, group, source_id. Identity groups are
             unioned across datasets (DF40's FF++-domain videos share
             FaceForensics++'s groups); the leakage check runs on the result.
  extract    per video and per processing variant (docs/13 §3.2): sample
             frames within the first window_s, find faces (the two-pass
             finder), crop, embed with every candidate backbone on the GPU,
             plus K crop-augmented copies of the training rows; one npz per
             video, written atomically. Errors are recorded per variant and
             are never confused with "no face".
  select     docs/13 §6.2: stage A (representation by repeated
             leave-one-manipulation-out AUC on the base set), stage B
             (training set by leave-one-processing-source-out false alarms),
             the threshold, the final fit, the held-out tables over judged
             clips with group-bootstrap intervals, and the §6.3 gates that
             the cloud can measure; writes the bundle and a report.

Usage (local smoke run):
    python scripts/train_faceswap_v2.py manifest --ffpp data/raw/ffpp --out runs/fsv2/manifest.csv
    python scripts/train_faceswap_v2.py extract --manifest runs/fsv2/manifest.csv --features-dir runs/fsv2/features \\
        --backbones vit_base_patch16_clip_224.openai --limit 10 --device cpu
    python scripts/train_faceswap_v2.py select --manifest runs/fsv2/manifest.csv --features-dir runs/fsv2/features \\
        --out-dir runs/fsv2/result --smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

if os.environ.get("MPLBACKEND", "").startswith("module://"):  # Colab/Kaggle inline backends break MediaPipe
    os.environ["MPLBACKEND"] = "Agg"

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
from trubeat import faceswap as fs

FFPP_METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter")
SWAP_FAMILIES = ("Deepfakes", "FaceSwap", "FaceShifter")
DFD_REAL, DFD_FAKE = "original_sequences/actors", "manipulated_sequences/DeepFakeDetection"
CDF_TEST_LIST = "List_of_testing_videos.txt"


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def load_cfg(path: str | None = None) -> dict:
    p = Path(path) if path else REPO / "configs/p7_faceswap_v2.yaml"
    return yaml.safe_load(p.read_text())


def inference_cfg(cfg: dict) -> dict:
    """The keys the app reads (faceswap.DEFAULT_CONFIG), taken from the v2 config."""
    return {k: cfg[k] for k in fs.DEFAULT_CONFIG if k in cfg}


# ----------------------------------------------------------- identity groups ---

class UnionFind:
    def __init__(self) -> None:
        self.p: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def shard_of(video_id: str, n: int) -> int:
    """Stable shard of a video: the cloud fetch + extraction of FF++ and DFD is
    split across parallel sessions by this (a session holds ~4 vCPU)."""
    return int(hashlib.sha256(video_id.encode()).hexdigest()[:8], 16) % n


def parse_shard(text: str | None) -> tuple[int, int] | None:
    if not text:
        return None
    i, n = (int(x) for x in text.split("/"))
    if not 0 <= i < n:
        raise SystemExit(f"bad --shard {text}: want I/N with 0 <= I < N")
    return i, n


def dfd_groups(stems: list[str]) -> dict[str, str]:
    """DFD stem -> identity group over every DFD video given: both actors of a
    swap are joined (docs/13 B-17). Run on the whole manifest, so shards
    fetched in separate sessions end up with the same groups."""
    uf = UnionFind()
    first = {}
    for stem in stems:
        actors = [x for x in stem.split("__")[0].split("_") if x]
        first[stem] = f"dfd:{actors[0]}"
        for x in actors[1:]:
            uf.union(f"dfd:{actors[0]}", f"dfd:{x}")
    return {s: uf.find(g) for s, g in first.items()}


def regroup_dfd(df: pd.DataFrame) -> pd.DataFrame:
    m = df.dataset == "dfd"
    if m.any():
        g = dfd_groups(df.loc[m, "source_id"].tolist())
        df.loc[m, "group"] = df.loc[m, "source_id"].map(g)
    return df


def ffpp_splits(root: Path) -> tuple[dict[str, str], dict[str, str]]:
    """FaceForensics++ id -> split, id -> its official pair group."""
    split_of, group_of = {}, {}
    for split in ("train", "val", "test"):
        f = root / "splits" / f"{split}.json"
        if not f.is_file():
            raise SystemExit(f"missing {f}: run `fetch` first (or copy the official splits)")
        for a_, b_ in json.loads(f.read_text()):
            a, b = str(a_), str(b_)
            g = "ffpp:" + "_".join(sorted((a, b)))
            for x in (a, b):
                split_of[x], group_of[x] = split, g
    return split_of, group_of


# --------------------------------------------------------------------- fetch ---

def drive_url(file_id: str) -> str:
    return f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"


def fetch_zip_members(url: str, members: list[str], dest: Path, threads: int = 6) -> tuple[int, list[str]]:
    """Copy the named members of a remote stored zip into ``dest`` (same
    relative paths) by HTTP range requests; existing files are kept."""
    from remotezip import RemoteZip

    todo = [m for m in members if not (dest / m).is_file()]
    if not todo:
        return 0, []
    failed: list[str] = []

    def one(m: str) -> bool:
        for attempt in range(4):
            try:
                with RemoteZip(url) as z:
                    data = z.read(m)
                out = dest / m
                out.parent.mkdir(parents=True, exist_ok=True)
                tmp = out.with_name(out.name + ".part")
                tmp.write_bytes(data)
                os.replace(tmp, out)
                return True
            except Exception as exc:  # noqa: BLE001 - retried, then reported
                if attempt == 3:
                    log(f"  failed {m}: {type(exc).__name__}: {str(exc)[:200]}")
                time.sleep(2 + 3 * attempt)
        return False

    with ThreadPoolExecutor(threads) as ex:
        futs = {ex.submit(one, m): m for m in todo}
        for f in as_completed(futs):
            if not f.result():
                failed.append(futs[f])
    return len(todo) - len(failed), failed


def zip_index(url: str) -> list[str]:
    from remotezip import RemoteZip

    with RemoteZip(url) as z:
        return [i.filename for i in z.infolist()]


def mp4s(names: list[str]) -> list[str]:
    return sorted(n for n in names if n.endswith(".mp4") and "__MACOSX" not in n and "/._" not in n)


def df40_pair_ids(stem: str) -> list[str]:
    """FaceForensics++ ids in a DF40 FF-domain file name: '001_870' -> ['001', '870'];
    talking-head names '970_4mJ1aHFPS1M_50009' -> ['970']."""
    parts = stem.split("_")
    ids = [p for p in parts[:2] if p.isdigit() and len(p) == 3]
    return ids


def plan_df40(cfg: dict, ids: dict[str, str], split_of: dict[str, str], rng: np.random.Generator) -> dict:
    """Which members to fetch from which DF40 zip: {zip_path: [members]}."""
    d = cfg["df40"]
    plan: dict[str, list[str]] = {}
    for m in d["train_methods"]:
        key = f"train/{m}.zip"
        if key not in ids:
            log(f"  DF40 {key} not in the folder listing; skipped")
            continue
        names = mp4s(zip_index(drive_url(ids[key])))
        ok = [n for n in names if (p := df40_pair_ids(Path(n).stem)) and all(split_of.get(i) in ("train", "val") for i in p)]
        pick = sorted(rng.choice(ok, size=min(int(d["train_cap"]), len(ok)), replace=False).tolist()) if ok else []
        plan[key] = pick
        log(f"  DF40 {key}: {len(names)} videos, {len(ok)} on FF++ train/val identities, fetching {len(pick)}")
    test_methods = sorted(set(d["gate_methods"]) | set(d["reenact_methods"]) | set(d["cdf_methods"]))
    for m in test_methods + ["deepfacelab"]:
        cands = [k for k in ids if k.lower() in (f"test/{m}.zip", f"test/{m.replace('one_shot', 'oneshot')}.zip")]
        if not cands:
            log(f"  DF40 test/{m}.zip not found; skipped")
            continue
        key = cands[0]
        names = mp4s(zip_index(drive_url(ids[key])))
        ff = [n for n in names if n.split("/")[-2] == "ff"]
        cdf = [n for n in names if n.split("/")[-2] in ("cdf", "video") and Path(n).stem.startswith("id")]
        pick = []
        if m in d["gate_methods"] or m in d["reenact_methods"]:
            pick += [n for n in ff if all(split_of.get(i) == "test" for i in df40_pair_ids(Path(n).stem))]
        if m in d["cdf_methods"] and cdf:
            pick += sorted(rng.choice(cdf, size=min(int(d["cdf_cap"]), len(cdf)), replace=False).tolist())
        if m == "deepfacelab":
            pick = names  # UADFV: real/ + fake/
        plan[key] = sorted(set(pick))
        log(f"  DF40 {key}: {len(names)} videos ({len(ff)} ff, {len(cdf)} cdf), fetching {len(plan[key])}")
    return plan


def cmd_fetch(a) -> None:
    import train_faceswap as tf3

    cfg = load_cfg(a.config)
    root = Path(a.root)
    report: dict = {"created": time.strftime("%Y-%m-%d %H:%M")}
    for split in ("train", "val", "test"):  # the official splits: every dataset's identities are checked against them
        dest = root / "ffpp" / "splits" / f"{split}.json"
        if not dest.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(tf3._get(tf3.SPLITS_URL.format(split)))
    if "ffpp" in a.datasets:
        base = tf3.pick_server()
        log(f"FaceForensics++ server {base}")
        pairs = json.loads(tf3._get(base + "misc/filelist.json"))
        shard = parse_shard(a.shard)

        def keep(video_id: str) -> bool:  # same ids as the manifest; c23 and c40 of a video share a shard
            return shard is None or shard_of(video_id, shard[1]) == shard[0]

        jobs = []
        for comp in a.compression:
            jobs += [f"original_sequences/youtube/{comp}/videos/{x}.mp4" for x in sorted({v for p in pairs for v in p})
                     if keep(f"ffpp__original__{x}")]
            for m in FFPP_METHODS:
                for p in pairs:
                    for t, s in ((p[0], p[1]), (p[1], p[0])):
                        if keep(f"ffpp__{m}__{t}_{s}"):
                            jobs.append(f"manipulated_sequences/{m}/{comp}/videos/{t}_{s}.mp4")
        names = json.loads(tf3._get(base + "misc/deepfake_detection_filenames.json"))
        rng = np.random.default_rng(int(cfg["seed"]))
        real = sorted(names["actors"])
        fake = sorted(names["DeepFakesDetection"])
        pick_r = sorted(rng.choice(real, size=min(a.dfd_real, len(real)), replace=False).tolist())
        pick_f = sorted(rng.choice(fake, size=min(a.dfd_fake, len(fake)), replace=False).tolist())
        for comp in a.compression:
            jobs += [f"{DFD_REAL}/{comp}/videos/{n}.mp4" for n in pick_r if keep(f"dfd__DFD-original__{n}")]
            jobs += [f"{DFD_FAKE}/{comp}/videos/{n}.mp4" for n in pick_f if keep(f"dfd__DFD__{n}")]
        report["ffpp_shard"] = a.shard or "all"
        log(f"FaceForensics++ + DFD: {len(jobs)} files (shard {a.shard or 'all'})")
        missing = []
        with ThreadPoolExecutor(a.threads) as ex:
            futs = {ex.submit(tf3.fetch, base + j, root / "ffpp" / j): j for j in jobs}
            for k, f in enumerate(as_completed(futs), 1):
                if not f.result():
                    missing.append(futs[f])
                if k % 1000 == 0:
                    log(f"  {k}/{len(jobs)} ({len(missing)} missing)")
        report["ffpp"] = {"files": len(jobs), "missing": len(missing), "missing_list": missing[:200]}
        log(f"FaceForensics++ + DFD: {len(jobs) - len(missing)}/{len(jobs)} present")
    links = json.loads(Path(a.links).read_text()) if a.links else {}
    split_of, _ = ffpp_splits(root / "ffpp")
    if "df40" in a.datasets:
        ids = links.get("df40_ids") or {}
        if not ids:
            raise SystemExit("DF40 needs --links with df40_ids (zip path -> Drive file id)")
        rng = np.random.default_rng(int(cfg["seed"]))
        plan = plan_df40(cfg, ids, split_of, rng)
        report["df40"] = {}
        for key, members in plan.items():
            ok, failed = fetch_zip_members(drive_url(ids[key]), members, root / "df40" / key.replace(".zip", ""))
            report["df40"][key] = {"planned": len(members), "fetched_now": ok, "failed": failed[:50]}
            log(f"  DF40 {key}: {len(members) - len(failed)}/{len(members)} present")
    if "cdf2" in a.datasets:
        cid = links.get("cdf2_id")
        if not cid:
            raise SystemExit("Celeb-DF v2 needs --links with cdf2_id")
        url = drive_url(cid)
        names = zip_index(url)
        test_member = next(n for n in names if n.endswith(CDF_TEST_LIST))
        fetch_zip_members(url, [test_member], root / "cdf2")
        test = [ln.split()[1] for ln in (root / "cdf2" / test_member).read_text().splitlines() if ln.strip()]
        prefix = test_member[: -len(CDF_TEST_LIST)]
        reals = [n for n in mp4s(names) if "-real/" in n]
        want = sorted({prefix + t for t in test} | set(reals))
        ok, failed = fetch_zip_members(url, want, root / "cdf2")
        report["cdf2"] = {"planned": len(want), "failed": failed[:50]}
        log(f"Celeb-DF v2: {len(want) - len(failed)}/{len(want)} present")
    out = root / "fetch_report.json"
    old = json.loads(out.read_text()) if out.is_file() else {}
    old.update(report)
    out.write_text(json.dumps(old, indent=1))


# ------------------------------------------------------------------ manifest ---

def manifest_ffpp(root: Path) -> list[dict]:
    split_of, group_of = ffpp_splits(root)
    rows = []
    ids = sorted(split_of)
    for x in ids:
        p = root / f"original_sequences/youtube/c23/videos/{x}.mp4"
        rows.append({"video_id": f"ffpp__original__{x}", "dataset": "ffpp", "family": "original", "label": "real",
                     "path": str(p), "path_c40": str(root / f"original_sequences/youtube/c40/videos/{x}.mp4"),
                     "split": split_of[x], "group": group_of[x], "source_id": x})
    for split in ("train", "val", "test"):
        for a_, b_ in json.loads((root / "splits" / f"{split}.json").read_text()):
            a, b = str(a_), str(b_)
            for m in FFPP_METHODS:
                for t, s in ((a, b), (b, a)):
                    stem = f"{t}_{s}"
                    rows.append({"video_id": f"ffpp__{m}__{stem}", "dataset": "ffpp", "family": m, "label": "fake",
                                 "path": str(root / f"manipulated_sequences/{m}/c23/videos/{stem}.mp4"),
                                 "path_c40": str(root / f"manipulated_sequences/{m}/c40/videos/{stem}.mp4"),
                                 "split": split, "group": group_of[a], "source_id": stem})
    dfd = []
    for sub, label, fam in ((DFD_REAL, "real", "DFD-original"), (DFD_FAKE, "fake", "DFD")):
        for p in sorted((root / sub / "c23/videos").glob("*.mp4")):
            dfd.append({"video_id": f"dfd__{fam}__{p.stem}", "dataset": "dfd", "family": fam, "label": label,
                        "path": str(p), "path_c40": str(root / sub / "c40/videos" / p.name),
                        "split": "external", "group": "", "source_id": p.stem})
    groups = dfd_groups([r["source_id"] for r in dfd])   # docs/13 B-17: both actors of a swap
    for r in dfd:
        r["group"] = groups[r["source_id"]]
    return rows + dfd


def manifest_df40(root: Path, ffpp_root: Path) -> list[dict]:
    split_of, group_of = ffpp_splits(ffpp_root)
    rows = []
    for zdir in sorted((root / "train").glob("*")) + sorted((root / "test").glob("*")):
        part = zdir.parent.name                       # train | test
        for p in sorted(zdir.rglob("*.mp4")):
            sub = p.parent.name
            stem = p.stem
            method = zdir.name.lower().replace("one_shot", "oneshot")
            if method == "deepfacelab":               # UADFV inside DF40's test folder
                rows.append({"video_id": f"uadfv__{sub}__{stem}", "dataset": "uadfv",
                             "family": "UADFV-original" if sub == "real" else "UADFV", "label": "real" if sub == "real" else "fake",
                             "path": str(p), "path_c40": "", "split": "external",
                             "group": f"uadfv:{stem.split('_')[0]}", "source_id": stem})
                continue
            ids = df40_pair_ids(stem)
            if sub in ("cdf", "video") and stem.startswith("id"):
                a_ = stem.split("_")[0]
                rows.append({"video_id": f"df40__{method}__cdf__{stem}", "dataset": "df40cdf", "family": f"df40:{method}",
                             "label": "fake", "path": str(p), "path_c40": "", "split": "external",
                             "group": f"cdf2:{a_}", "source_id": stem})
                continue
            if not ids:
                continue
            splits = {split_of.get(i) for i in ids}
            if None in splits or len(splits) != 1:
                continue                                # unknown id, or identities from two splits: never used
            split = splits.pop()
            if part == "train" and split == "test":
                continue                                # a test identity in a training zip: dropped (leakage guard)
            rows.append({"video_id": f"df40__{method}__{part}__{stem}", "dataset": "df40", "family": f"df40:{method}",
                         "label": "fake", "path": str(p), "path_c40": "", "split": split,
                         "group": group_of[ids[0]], "source_id": stem})
    return rows


def manifest_cdf2(root: Path) -> list[dict]:
    lists = list(root.rglob(CDF_TEST_LIST))
    if not lists:
        return []
    base = lists[0].parent
    test = {ln.split()[1] for ln in lists[0].read_text().splitlines() if ln.strip()}
    rows = []
    for p in sorted(base.rglob("*.mp4")):
        rel = str(p.relative_to(base))
        folder = p.parent.name
        label = "real" if folder.endswith("-real") else "fake"
        stem = p.stem
        if folder == "YouTube-real":
            group = f"cdf2:yt:{stem}"
        else:
            group = f"cdf2:{stem.split('_')[0]}"
        rows.append({"video_id": f"cdf2__{folder}__{stem}", "dataset": "cdf2",
                     "family": "CDF-original" if label == "real" else "CDF",
                     "label": label, "path": str(p), "path_c40": "",
                     "split": "external", "group": group, "source_id": rel,
                     "cdf_set": "test" if rel in test else "outside"})
    return rows


def check_manifest_v2(df: pd.DataFrame) -> None:
    """AGENTS.md §6 before every training job: FaceForensics++ splits are
    identity-disjoint (DF40's FF-domain rows included, they share the
    groups), and no group of a training/val/test row appears in the
    external sets."""
    from trubeat import eval as ev

    inner = df[df.split != "external"]
    ev.validate_manifest(inner.rename(columns={"group": "split_group"}))
    shared = set(inner.group) & set(df[df.split == "external"].group)
    if shared:
        raise ValueError(f"identity leakage: {len(shared)} group(s) in both the training splits and an external "
                         f"set: {sorted(shared)[:5]}")
    dup = df.video_id[df.video_id.duplicated()]
    if len(dup):
        raise ValueError(f"duplicate video ids: {dup.tolist()[:5]}")


def cmd_manifest(a) -> None:
    rows = []
    if a.ffpp:
        rows += manifest_ffpp(Path(a.ffpp))
    if a.df40:
        rows += manifest_df40(Path(a.df40), Path(a.ffpp))
    if a.cdf2:
        rows += manifest_cdf2(Path(a.cdf2))
    df = pd.DataFrame(rows)
    if "cdf_set" not in df:
        df["cdf_set"] = ""
    df["cdf_set"] = df["cdf_set"].fillna("")
    df = df[df.path.map(lambda p: Path(p).is_file())].reset_index(drop=True)
    df["path_c40"] = df.path_c40.where(df.path_c40.map(lambda p: bool(p) and Path(p).is_file()), "")
    check_manifest_v2(df)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    log(f"{len(df)} videos -> {a.out}")
    print(df.groupby(["dataset", "split", "label"]).size().to_string())


# ------------------------------------------------------------------ variants ---

def video_size(path: str) -> tuple[int, int]:
    import cv2

    cap = cv2.VideoCapture(path)
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()


_HEVC: list[str] = []


def hevc_encoder() -> str:
    """hevc_nvenc when it works on this machine, else libx265 (probed once)."""
    if _HEVC:
        return _HEVC[0]
    for enc in ("hevc_nvenc", "libx265"):
        r = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=s=256x256:d=0.5",
                            "-c:v", enc, "-f", "null", "-"], capture_output=True, timeout=60, check=False)
        if r.returncode == 0:
            _HEVC.append(enc)
            return enc
    _HEVC.append("libx264")
    return "libx264"


def variant_cmd(src: str, dst: str, v: str, spec: dict, window: float, w: int, h: int) -> tuple[list[str], str]:
    """ffmpeg command for processing variant ``v`` of ``src`` (first ``window`` s)."""
    base = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-t", f"{window:g}", "-i", src, "-an"]
    if v == "v1":
        return base + ["-vf", f"scale='min({spec['width']},iw)':-2", "-c:v", "libx264", "-crf", str(spec["crf"]),
                       "-preset", spec["preset"], "-pix_fmt", "yuv420p", dst], "libx264"
    if v == "v2":
        enc = hevc_encoder()
        vf = ["-vf", f"scale='min({spec['width']},iw)':-2"]
        if enc == "hevc_nvenc":
            q = ["-c:v", "hevc_nvenc", "-rc", "constqp", "-qp", str(int(spec["crf"]) + 2)]
        elif enc == "libx265":
            q = ["-c:v", "libx265", "-crf", str(spec["crf"]), "-preset", "ultrafast", "-x265-params", "log-level=error"]
        else:
            q = ["-c:v", "libx264", "-crf", str(int(spec["crf"]) - 2), "-preset", "veryfast"]
        return base + vf + q + ["-pix_fmt", "yuv420p", dst], enc
    if v == "v3":
        s = int(spec["short_side"])
        vf = f"scale=-2:'min({s},ih)'" if w >= h else f"scale='min({s},iw)':-2"
        return base + ["-vf", vf, "-c:v", "libvpx-vp9", "-b:v", spec["bitrate"], "-deadline", "realtime",
                       "-cpu-used", "8", "-row-mt", "1", dst], "libvpx-vp9"
    if v == "v4":
        s = int(spec["down_short_side"])
        down = f"scale=-2:'min({s},ih)'" if w >= h else f"scale='min({s},iw)':-2"
        return base + ["-vf", f"{down},scale={w}:{h}", "-c:v", "libx264", "-crf", str(spec["crf"]),
                       "-preset", spec["preset"], "-pix_fmt", "yuv420p", dst], "libx264"
    raise ValueError(f"unknown variant {v}")


# ------------------------------------------------------------------- extract ---

_W: dict = {}


def _init_worker(cfg: dict, stage2: bool) -> None:
    warnings.filterwarnings("ignore")
    _W["cfg"], _W["stage2"] = cfg, stage2
    _W["finder"] = fs.FaceFinder(inference_cfg(cfg))


def augment(crop: np.ndarray, rng: np.random.Generator, spec: dict) -> np.ndarray:
    """One crop-level processing augmentation (docs/13 §3.2): rescale down and
    up, blur, brightness/contrast, then a JPEG round trip."""
    import cv2

    s = crop.shape[0]
    k = rng.uniform(*spec["rescale"])
    small = cv2.resize(crop, (max(8, round(s * k)),) * 2, interpolation=cv2.INTER_AREA)
    out = cv2.resize(small, (s, s), interpolation=cv2.INTER_LINEAR)
    if rng.random() < float(spec["blur_p"]):
        out = cv2.GaussianBlur(out, (0, 0), float(rng.uniform(0.3, spec["blur_sigma_max"])))
    a = 1.0 + rng.uniform(-spec["contrast"], spec["contrast"])
    b = 255.0 * rng.uniform(-spec["brightness"], spec["brightness"])
    out = np.clip(out.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
    q = int(rng.integers(spec["jpeg_quality"][0], spec["jpeg_quality"][1] + 1))
    _, buf = cv2.imencode(".jpg", out[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1].copy()


def stage2_jpegs(sampled, crops, size: int) -> list[bytes]:
    import cv2

    by_index = dict(zip(sampled.index, sampled.frames))
    out = []
    for c in crops:
        x0, y0, s, _ = c.box
        img = fs.cut_square(by_index[c.frame_index], x0 + s / 2.0, y0 + s / 2.0, float(s), size)
        out.append(cv2.imencode(".jpg", img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes())
    return out


def process_video(row: dict) -> dict:
    """Worker: every variant of one video -> crops (or an error), plus the
    augmented copies of the v0 crops for training rows."""
    cfg = _W["cfg"]
    window = float(cfg["window_s"])
    out = {"video_id": row["video_id"], "variants": {}, "aug": [], "jpgs": []}
    w = h = 0
    with tempfile.TemporaryDirectory(prefix="fsv2_") as tmp:
        for v in row["variants"]:
            rec = {"crops": None, "n_sampled": 0, "n_faces": 0, "error": "", "encoder": "", "n_seen_max": 0}
            try:
                if v == "v0":
                    src = row["path"]
                elif v == "c40":
                    if not row.get("path_c40"):
                        continue                         # no c40 file for this video (not an error)
                    src = row["path_c40"]
                else:
                    if not w:
                        w, h = video_size(row["path"])
                    dst = str(Path(tmp) / f"{v}.{'webm' if v == 'v3' else 'mp4'}")
                    cmd, rec["encoder"] = variant_cmd(row["path"], dst, v, cfg["variant_specs"][v], window, w, h)
                    subprocess.run(cmd, check=True, capture_output=True, timeout=int(cfg["ffmpeg_timeout_s"]))
                    src = dst
                s = fs.sample_frames(src, int(cfg["n_frames"]), window)
                seen = []
                crops = []
                finder = _W["finder"]
                for rgb, idx, t in zip(s.frames, s.index, s.time_s):
                    box = finder.largest_face(rgb)
                    seen.append(finder.last_seen)
                    if box is None:
                        continue
                    cx, cy, side = box
                    crop = fs.cut_square(rgb, cx, cy, side, int(cfg["crop_size"]))
                    sz = int(np.rint(side))           # the same integer box as faceswap.face_crops
                    crops.append(fs.FaceCrop(crop, idx, t, (int(np.rint(cx - sz / 2)), int(np.rint(cy - sz / 2)), sz, sz)))
                rec["n_sampled"], rec["n_faces"], rec["n_seen_max"] = len(s.frames), len(crops), max(seen or [0])
                if crops:
                    rec["crops"] = np.stack([c.crop for c in crops])
                    if v == "v0" and _W["stage2"]:
                        out["jpgs"] = stage2_jpegs(s, crops, int(cfg["stage2_crop_size"]))
            except subprocess.TimeoutExpired:
                rec["error"] = f"ffmpeg timeout ({cfg['ffmpeg_timeout_s']} s) for {v}"
            except subprocess.CalledProcessError as exc:
                rec["error"] = f"ffmpeg failed for {v}: {exc.stderr.decode(errors='replace')[-300:]}"
            except Exception as exc:  # noqa: BLE001 - recorded per variant, never silent
                rec["error"] = f"{type(exc).__name__}: {exc}"
            out["variants"][v] = rec
    v0 = out["variants"].get("v0", {})
    if row["split"] in ("train", "val") and v0.get("crops") is not None:
        spec = cfg["crop_aug"]
        rng = np.random.default_rng(int(hashlib.sha256(row["video_id"].encode()).hexdigest()[:8], 16))
        for _ in range(int(spec["k"])):
            out["aug"].append(np.stack([augment(c, rng, spec) for c in v0["crops"]]))
    return out


def feature_path(fdir: Path, video_id: str) -> Path:
    return fdir / video_id.split("__")[0] / f"{video_id}.npz"


def extracted_v2(fdir: Path, video_id: str, backbones: list[str]) -> bool:
    """Done = the npz exists, loads, has every backbone, and no variant failed
    (failed videos are retried on the next run)."""
    p = feature_path(fdir, video_id)
    if not p.is_file():
        return False
    try:
        with np.load(p, allow_pickle=False) as z:
            if any(str(z[k]) for k in z.files if k.endswith("|error")):
                return False
            have = {k.split("|")[0] for k in z.files if k.endswith("|cls_mean")}
            return all(b in have for b in backbones) or not any(k.endswith("|cls_mean") for k in z.files)
    except Exception:  # noqa: BLE001 - a truncated file from a killed run is "not extracted"
        return False


def save_npz(path: Path, arrays: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def cmd_extract(a) -> None:
    cfg = load_cfg(a.config)
    if a.window:
        cfg["window_s"] = float(a.window)
    df = pd.read_csv(a.manifest, dtype=str, keep_default_na=False)
    if a.datasets:
        df = df[df.dataset.isin(a.datasets)]
    if a.splits:
        df = df[df.split.isin(a.splits)]
    if a.limit:
        df = df.groupby(["dataset", "split", "family"], group_keys=False).head(a.limit)
    fdir = Path(a.features_dir)
    device = a.device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    devices = [device]
    if device == "cuda":
        import torch
        devices = [f"cuda:{i}" for i in range(max(1, torch.cuda.device_count()))]
    embedders = {b: fs.FaceEmbedder(b, "cls+patch", int(cfg["torch_threads"]), device=devices[i % len(devices)],
                                    img_size=int(cfg["crop_size"])) for i, b in enumerate(a.backbones)}
    for b, e in embedders.items():  # load every backbone now: a missing one stops here, not mid-run
        log(f"{b}: {e.dim}-dim features on {e.device}")
    variants = cfg["variants"]
    rows = []
    for r in df.to_dict("records"):
        if extracted_v2(fdir, r["video_id"], a.backbones):
            continue
        r["variants"] = [v for v in variants.get(r["dataset"], ["v0"]) if not a.variants or v in a.variants]
        rows.append(r)
    stage2_dir = Path(a.stage2_dir) if a.stage2_dir else None
    log(f"{len(df)} videos, {len(rows)} to extract, backbones {a.backbones}, devices {devices}, "
        f"hevc encoder {hevc_encoder()}")
    t0, n_err, n_done = time.time(), 0, 0
    log_rows = []
    stopped_early = False
    with mp.get_context("spawn").Pool(a.workers, initializer=_init_worker,
                                      initargs=(cfg, stage2_dir is not None)) as pool:
        for k, res in enumerate(pool.imap_unordered(process_video, rows, chunksize=1), 1):
            vid = res["video_id"]
            arrays: dict = {"variants": np.array(list(res["variants"]))}
            errs = {v: r["error"] for v, r in res["variants"].items() if r["error"]}
            for v, r in res["variants"].items():
                arrays[f"{v}|error"] = np.array(r["error"])
                arrays[f"{v}|n_sampled"] = np.array(r["n_sampled"])
                arrays[f"{v}|n_faces"] = np.array(r["n_faces"])
                arrays[f"{v}|n_seen_max"] = np.array(r["n_seen_max"])
                arrays[f"{v}|encoder"] = np.array(r["encoder"])
            for b, e in embedders.items():
                for v, r in res["variants"].items():
                    if r["crops"] is None:
                        continue
                    emb = e.embed(r["crops"])
                    d = emb.shape[1] // 2
                    arrays[f"{b}|{v}|cls_mean"] = emb[:, :d].mean(axis=0).astype(np.float32)
                    arrays[f"{b}|{v}|patch_mean"] = emb[:, d:].mean(axis=0).astype(np.float32)
                    if v == "v0":
                        arrays[f"{b}|v0|cls"] = emb[:, :d].astype(np.float16)
                        arrays[f"{b}|v0|patch"] = emb[:, d:].astype(np.float16)
                for j, aug in enumerate(res["aug"]):
                    emb = e.embed(aug)
                    d = emb.shape[1] // 2
                    arrays[f"{b}|aug{j}|cls_mean"] = emb[:, :d].mean(axis=0).astype(np.float32)
                    arrays[f"{b}|aug{j}|patch_mean"] = emb[:, d:].mean(axis=0).astype(np.float32)
            if stage2_dir is not None and res["jpgs"]:
                dd = stage2_dir / vid
                dd.mkdir(parents=True, exist_ok=True)
                for j, data in enumerate(res["jpgs"]):
                    (dd / f"{j:02d}.jpg").write_bytes(data)
                (dd / "done").write_text(str(len(res["jpgs"])))
            save_npz(feature_path(fdir, vid), arrays)
            n_done += 1
            if errs:
                n_err += 1
                if n_err <= 10:
                    log(f"  error on {vid}: {json.dumps(errs)[:400]}")
                log_rows.append({"video_id": vid, "errors": errs})
                if n_err == k and k >= 30:
                    raise SystemExit(f"the first {k} videos all failed: the setup is broken; last: {json.dumps(errs)[:400]}")
            if k % 100 == 0 or k == len(rows):
                rate = k / max(time.time() - t0, 1e-9)
                log(f"  {k}/{len(rows)} ({rate:.2f} videos/s, {n_err} with errors, ~{(len(rows) - k) / max(rate, 1e-9) / 60:.0f} min left)")
            if a.time_budget_s and time.time() - t0 > a.time_budget_s:
                stopped_early = True   # a cloud session must end cleanly, with its outputs saved; a rerun resumes
                log(f"  time budget ({a.time_budget_s:.0f} s) reached after {k}/{len(rows)} videos: stopping cleanly")
                break
    lp = fdir / "extract_log.json"
    old = json.loads(lp.read_text()) if lp.is_file() else {"runs": []}
    old["runs"].append({"finished": time.strftime("%Y-%m-%d %H:%M"), "videos": len(rows), "written": n_done,
                        "stopped_early": stopped_early, "with_errors": n_err,
                        "errors": log_rows[:500], "hevc_encoder": hevc_encoder(), "window_s": cfg["window_s"]})
    lp.parent.mkdir(parents=True, exist_ok=True)
    lp.write_text(json.dumps(old, indent=1))
    log(f"done: {n_done} videos written, {n_err} with an error in at least one variant (see {lp})")
    if n_err and a.strict:
        raise SystemExit(f"{n_err} videos had errors (strict mode)")


# -------------------------------------------------------------------- store ---

def load_store(df: pd.DataFrame, fdir: Path, backbone: str, min_faces: int, k_aug: int) -> pd.DataFrame:
    """One row per (video, source) with a judged descriptor (a face in >= min_faces
    frames) or None, plus the error text. Sources: the variants and aug0..augK-1."""
    rows = []
    for r in df.itertuples(index=False):
        p = feature_path(fdir, r.video_id)
        if not p.is_file():
            rows.append({"video_id": r.video_id, "source": "v0", "cls": None, "patch": None, "n_faces": -1,
                         "error": "no feature file"})
            continue
        with np.load(p, allow_pickle=False) as z:
            files = set(z.files)
            for v in [str(x) for x in z["variants"]] + [f"aug{j}" for j in range(k_aug)]:
                err = str(z[f"{v}|error"]) if f"{v}|error" in files else ""
                if v.startswith("aug"):
                    nf = int(z["v0|n_faces"]) if "v0|n_faces" in files else 0
                    if f"{backbone}|{v}|cls_mean" not in files:
                        continue
                else:
                    nf = int(z[f"{v}|n_faces"])
                ok = nf >= min_faces and f"{backbone}|{v}|cls_mean" in files and not err
                rows.append({"video_id": r.video_id, "source": v,
                             "cls": z[f"{backbone}|{v}|cls_mean"] if ok else None,
                             "patch": z[f"{backbone}|{v}|patch_mean"] if ok else None,
                             "n_faces": nf, "error": err})
    st = pd.DataFrame(rows)
    meta = df.set_index("video_id")
    for c in ("dataset", "family", "label", "split", "group", "cdf_set"):
        st[c] = st.video_id.map(meta[c]) if c in meta else ""
    st["judged"] = st.cls.map(lambda x: x is not None)
    return st


def descriptors(st: pd.DataFrame, feature: str) -> np.ndarray:
    parts = [np.stack(st.cls.to_numpy())]
    if feature == "cls+patch":
        parts.append(np.stack(st.patch.to_numpy()))
    return np.hstack(parts).astype(np.float32)


# ------------------------------------------------------------------- select ---

def make_model(C: float):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=C, max_iter=3000))])


def balance_weights(st: pd.DataFrame) -> np.ndarray:
    """Equal total weight per (label, source) cell, so that no processing
    source carries label information (docs/13 §3.2)."""
    key = st.label + "|" + st.source
    counts = key.map(key.value_counts())
    w = 1.0 / counts.to_numpy(dtype=float)
    return w * (len(w) / w.sum())


def fit(X: np.ndarray, st: pd.DataFrame, C: float):
    m = make_model(C)
    m.fit(X, st.label.to_numpy(), clf__sample_weight=balance_weights(st))
    return m


def swap_score(model, X: np.ndarray) -> np.ndarray:
    s = model.decision_function(X)
    return s if list(model.classes_)[1] == "fake" else -s


def group_folds(groups: pd.Series, n_folds: int, seed: int) -> np.ndarray:
    """Fold per row with whole groups kept together, groups shuffled by ``seed``."""
    uniq = np.array(sorted(groups.unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    fold_of = {g: i % n_folds for i, g in enumerate(uniq)}
    return groups.map(fold_of).to_numpy()


def auc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def training_rows(st: pd.DataFrame, tset: str) -> pd.DataFrame:
    srcs = {"base": {"v0", "c40"}, "video": {"v0", "c40", "v1", "v2", "v3", "v4"}}.get(tset)
    tv = st[st.split.isin(["train", "val"]) & st.judged]
    if tset == "video+crop":
        return tv[tv.source.isin({"v0", "c40", "v1", "v2", "v3", "v4"}) | tv.source.str.startswith("aug")]
    return tv[tv.source.isin(srcs)]


def lomo(st_t: pd.DataFrame, X: np.ndarray, C: float, folds: np.ndarray, n_folds: int) -> dict[str, float]:
    """Leave-one-manipulation-out AUC per fake family, on v0 rows of the held-out folds."""
    y = (st_t.label == "fake").to_numpy().astype(int)
    fams = sorted(f for f in st_t.family.unique() if st_t.loc[st_t.family == f, "label"].iloc[0] == "fake")
    out = {}
    for fam in fams:
        vals = []
        for f in range(n_folds):
            trm = (folds != f) & (st_t.family != fam).to_numpy()
            tem = (folds == f) & ((st_t.label == "real") | (st_t.family == fam)).to_numpy() & (st_t.source == "v0").to_numpy()
            if trm.sum() == 0 or y[tem].min() == y[tem].max():
                continue
            m = fit(X[trm], st_t[trm], C)
            vals.append(auc(y[tem], swap_score(m, X[tem])))
        out[fam] = float(np.nanmean(vals)) if vals else float("nan")
    return out


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    o = np.argsort(values)
    v, w = values[o], weights[o]
    c = np.cumsum(w) / w.sum()
    return float(np.interp(q, c, v))


def source_weights(st: pd.DataFrame) -> np.ndarray:
    counts = st.source.map(st.source.value_counts()).to_numpy(dtype=float)
    return 1.0 / counts


def lopso(st_all: pd.DataFrame, X_all: np.ndarray, tset: str, C: float, folds_all: np.ndarray, n_folds: int,
          sources: list[str], target: float) -> dict[str, tuple[int, int]]:
    """Leave-one-processing-source-out false alarms: for each source s, train on
    the training set without s, set the threshold on out-of-fold reals of the
    other sources, and count source-s reals above it (docs/13 §6.2)."""
    tv = st_all.split.isin(["train", "val"]).to_numpy() & st_all.judged.to_numpy()
    in_t = st_all.index.isin(training_rows(st_all, tset).index)
    real = (st_all.label == "real").to_numpy()
    src = st_all.source.to_numpy()
    out = {}
    for s in sources:
        k = n = 0
        for f in range(n_folds):
            trm = in_t & (folds_all != f) & (src != s)
            te_other = tv & (folds_all == f) & real & in_t & (src != s) & ~st_all.source.str.startswith("aug").to_numpy()
            te_s = tv & (folds_all == f) & real & (src == s)
            if trm.sum() == 0 or te_other.sum() == 0 or te_s.sum() == 0:
                continue
            m = fit(X_all[trm], st_all[trm], C)
            sc_other = swap_score(m, X_all[te_other])
            thr = weighted_quantile(sc_other, source_weights(st_all[te_other]), 1 - target)
            sc_s = swap_score(m, X_all[te_s])
            k += int((sc_s >= thr).sum())
            n += int(te_s.sum())
        out[s] = (k, n)
    return out


def stage_b_run(backbone: str, feature: str, C: float, tset: str, meta: pd.DataFrame, X_all: np.ndarray,
                folds_all: np.ndarray, n_folds: int, sources: list[str], target: float) -> dict | None:
    """One stage-B cell: LOMO on the training set and LOPSO over the sources."""
    warnings.filterwarnings("ignore")
    tr = training_rows(meta, tset)
    if tr.empty:
        return None
    idx = tr.index.to_numpy()
    per = lomo(tr, X_all[idx], C, folds_all[idx], n_folds)
    lp = lopso(meta, X_all, tset, C, folds_all, n_folds, sources, target)
    rates = {s: (k / n if n else float("nan")) for s, (k, n) in lp.items()}
    return {"backbone": backbone, "feature": feature, "C": C, "training_set": tset,
            "lomo_macro": float(np.nanmean(list(per.values()))),
            "lopso_max": float(np.nanmax(list(rates.values()))),
            **{f"lopso_{s}": rates[s] for s in rates}, **{f"lopso_n_{s}": lp[s][1] for s in lp}}


def group_bootstrap_auc(y: np.ndarray, s: np.ndarray, groups: np.ndarray, n: int = 1000,
                        seed: int = 7) -> tuple[float, float]:
    """95% interval of the AUC, resampling identity groups (docs/13 B-06)."""
    uniq = np.unique(groups)
    if len(uniq) < 2 or not 0 < y.sum() < len(y):
        return float("nan"), float("nan")
    idx_of = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        pick = np.concatenate([idx_of[g] for g in rng.choice(uniq, len(uniq))])
        if 0 < y[pick].sum() < len(pick):
            vals.append(auc(y[pick], s[pick]))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"),) * 2


def wilson(k: int, n: int) -> tuple[float, float]:
    import train_aigen as ta

    return ta.wilson(k, n)


def rate_text(k: int, n: int, total: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k}/{n} judged = {k / max(n, 1):.1%} [{lo:.1%}, {hi:.1%}] (of {total})"


def md_table(df: pd.DataFrame, digits: int = 4) -> str:
    """A pipe table without extra dependencies."""
    def cell(v):
        if isinstance(v, float):
            return "nan" if np.isnan(v) else f"{v:.{digits}f}"
        return str(v)
    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    rows = ["| " + " | ".join(cell(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def threshold_reals(st_t: pd.DataFrame) -> np.ndarray:
    """Rows that set a threshold: reals of real processing sources (crop-level
    augmented copies are training material, not a processing source)."""
    return ((st_t.label == "real") & ~st_t.source.str.startswith("aug")).to_numpy()


def set_tables(ho: pd.DataFrame) -> tuple[list[str], dict]:
    """Markdown tables + metrics: for each (dataset, source): reals flagged over
    judged clips, and for each fake family the AUC vs that dataset's reals
    (group bootstrap) and the share caught (Wilson)."""
    L, M = [], {}
    real_pool = {"ffpp": ("ffpp", "real"), "dfd": ("dfd", "real"), "cdf2": ("cdf2", "real"), "uadfv": ("uadfv", "real"),
                 "df40": ("ffpp", "real"), "df40cdf": ("cdf2", "real")}
    for (ds, src), g in ho.groupby(["dataset", "source"]):
        rd, _ = real_pool.get(ds, (ds, "real"))
        reals = ho[(ho.dataset == rd) & (ho.source == src) & (ho.label == "real")]
        if ds == "cdf2":
            reals_t = reals[reals.cdf_set == "test"]
        else:
            reals_t = reals
        L += ["", f"### {ds} — {src}", ""]
        if ds in ("ffpp", "dfd", "cdf2", "uadfv"):
            for subset, rr in ((("test list" if ds == "cdf2" else "all"), reals_t),) + (
                    (("outside the test list", reals[reals.cdf_set == "outside"]),) if ds == "cdf2" else ()):
                j = rr[rr.judged]
                k = int(j.flagged.sum())
                n_err = int((rr.error != "").sum())
                M[f"{ds}|{src}|reals|{subset}"] = (k, len(j), len(rr))
                L.append(f"Reals flagged ({subset}): **{rate_text(k, len(j), len(rr))}**"
                         + (f"; {n_err} with extraction errors (not judged, not 'no face')" if n_err else "") + ".")
        L += ["", "| family | n | judged | errors | AUC vs reals (95% CI, by identity group) | caught (judged, Wilson 95%) |",
              "|---|---|---|---|---|---|"]
        fakes = g[g.label == "fake"]
        for fam, fk in fakes.groupby("family"):
            if ds == "cdf2":
                fk = fk[fk.cdf_set == "test"]
            both = pd.concat([reals_t[reals_t.judged], fk[fk.judged]])
            y = (both.label == "fake").to_numpy().astype(int)
            a_ = auc(y, both.score.to_numpy())
            lo, hi = group_bootstrap_auc(y, both.score.to_numpy(), both.group.to_numpy())
            jf = fk[fk.judged]
            kk = int(jf.flagged.sum())
            M[f"{ds}|{src}|{fam}|auc"] = a_
            M[f"{ds}|{src}|{fam}|caught"] = (kk, len(jf), len(fk))
            rlo, rhi = wilson(kk, len(jf))
            L.append(f"| {fam} | {len(fk)} | {len(jf)} | {int((fk.error != '').sum())} | {a_:.3f} [{lo:.3f}, {hi:.3f}] | "
                     f"{kk}/{len(jf)} = {kk / max(len(jf), 1):.1%} [{rlo:.1%}, {rhi:.1%}] |")
    return L, M


def gates_v2(M: dict, cfg: dict) -> list[tuple[str, bool | None, str]]:
    """docs/13 §6.3 gates the cloud measures (4a runs on the laptop)."""
    out = []
    for fam in SWAP_FAMILIES:
        v = M.get(f"ffpp|v0|{fam}|auc")
        out.append((f"1. FF++ test AUC >= 0.85 on {fam}", None if v is None or np.isnan(v) else bool(v >= 0.85),
                    "n/a" if v is None else f"{v:.3f}"))
    df40 = {k.split("|")[2]: v for k, v in M.items() if k.startswith("df40|v0|") and k.endswith("|auc")
            and k.split("|")[2].split(":")[-1] in cfg["df40"]["gate_methods"]}
    if df40:
        vals = [v for v in df40.values() if not np.isnan(v)]
        ok = bool(vals) and float(np.mean(vals)) >= 0.80 and min(vals) >= 0.70
        out.append(("1b. DF40 face-swap methods on FF++ test identities: mean AUC >= 0.80, none < 0.70", ok,
                    f"mean {np.mean(vals):.3f}, min {min(vals):.3f} over {len(vals)} methods"))
    else:
        out.append(("1b. DF40 face-swap methods on FF++ test identities", None, "not applicable: DF40 not available"))
    k, n, t = M.get("ffpp|v0|reals|all", (None, None, None))
    out.append(("2. FF++ test reals flagged <= 5/134 of judged", None if not n else bool(k / n <= 5 / 134),
                "n/a" if not n else f"{k}/{n} judged (of {t})"))
    for src in ("v1", "c40", "v3"):
        k, n, t = M.get(f"ffpp|{src}|reals|all", (None, None, None))
        out.append((f"2b. FF++ test reals after {src} flagged <= 5% of judged", None if not n else bool(k / n <= 0.05),
                    "n/a" if not n else f"{k}/{n} judged (of {t})"))
    v = M.get("dfd|v0|DFD|auc")
    k, n, t = M.get("dfd|v0|reals|all", (None, None, None))
    ok = None if v is None or not n else bool(v >= 0.70 and k / n <= 0.08)
    out.append(("3. DeepFakeDetection AUC >= 0.70 and reals flagged <= 8% of judged", ok,
                "n/a" if ok is None else f"AUC {v:.3f}, {k}/{n} judged flagged (of {t})"))
    v = M.get("cdf2|v0|CDF|auc")
    k, n, t = M.get("cdf2|v0|reals|test list", (None, None, None))
    ok = None if v is None or not n else bool(v >= 0.75 and k / n <= 0.08)
    out.append(("3b. Celeb-DF v2 test list: AUC >= 0.75 and reals flagged <= 8% of judged", ok,
                "not applicable: Celeb-DF not available" if ok is None else f"AUC {v:.3f}, {k}/{n} judged flagged (of {t})"))
    k, n, t = M.get("cdf2|v0|reals|outside the test list", (None, None, None))
    out.append(("4b. Celeb-DF v2 reals outside the test list flagged <= 5% of judged",
                None if not n else bool(k / n <= 0.05),
                "not applicable" if not n else f"{k}/{n} judged (of {t})"))
    out.append(("4a. frozen round-3 gate-4 list flagged <= 3% of judged, n >= 200, no source > 6%", None,
                "measured on the laptop (scripts/check_faceswap_reals.py --manifest data/manifests/faceswap_gate4a.csv)"))
    return out


def versions() -> dict:
    out = {}
    for mod in ("torch", "timm", "mediapipe", "sklearn", "numpy", "cv2", "pandas"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "?")
        except Exception:  # noqa: BLE001
            out[mod] = None
    try:
        out["git_commit"] = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                                           text=True, check=False).stdout.strip()
    except Exception:  # noqa: BLE001
        out["git_commit"] = None
    return out


def weights_sha(backbone: str) -> str | None:
    from trubeat import aigen

    p = aigen.cached_weights(backbone)
    if not p:
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_select(a) -> None:
    from joblib import Parallel, delayed

    warnings.filterwarnings("ignore")
    cfg = load_cfg(a.config)
    sel = cfg["selection"]
    if a.smoke:  # tiny settings for a local end-to-end check; never for a decision
        sel = dict(sel, repeats=1, folds=3, n_jobs=2)
    df = pd.read_csv(a.manifest, dtype=str, keep_default_na=False)
    df = regroup_dfd(df)        # shards were grouped separately; join DFD actors over the whole manifest
    check_manifest_v2(df)
    fdir, out = Path(a.features_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cand = cfg["candidates"]
    backbones = a.backbones or cand["backbones"]
    k_aug = int(cfg["crop_aug"]["k"])
    min_faces = int(cfg["min_faces"])
    target = float(cfg["target_fpr"])
    stores = {}
    for b in backbones:
        st = load_store(df, fdir, b, min_faces, k_aug)
        n_err = int((st.error != "").sum())
        if not st.judged.any():
            raise SystemExit(f"no features for {b} in {fdir} (docs/13 B-11: the grid may not shrink)")
        if n_err and not a.allow_errors:
            raise SystemExit(f"{n_err} (video, variant) rows carry extraction errors; see extract_log.json "
                             "(docs/13 B-02; --allow-errors reports them as errors, never as 'no face')")
        stores[b] = st.reset_index(drop=True)
        log(f"{b}: {len(st)} (video, source) rows, {int(st.judged.sum())} judged, {n_err} errors")
    # stage A: representation on the base set, repeated folds
    stage_a = []
    jobs = []
    for b in backbones:
        st = stores[b]
        base = training_rows(st, "base")
        meta = base.drop(columns=["cls", "patch"])      # workers get the metadata, not the arrays twice
        for feat in cand["features"]:
            X = descriptors(base, feat)
            for C in cand["C"]:
                for rep in range(int(sel["repeats"])):
                    folds = group_folds(base.group, int(sel["folds"]), int(sel["seeds"][rep]))
                    jobs.append(((b, feat, float(C), rep), meta, X, float(C), folds))
    log(f"stage A: {len(jobs)} LOMO runs")
    res = Parallel(n_jobs=int(sel["n_jobs"]))(delayed(lomo)(st_t, X, C, folds, int(sel["folds"]))
                                              for _, st_t, X, C, folds in jobs)
    agg: dict = {}
    for (key, *_), per in zip(jobs, res):
        agg.setdefault(key[:3], []).append(per)
    for (b, feat, C), pers in agg.items():
        fams = sorted({f for p in pers for f in p})
        per_f = {f: float(np.nanmean([p.get(f, np.nan) for p in pers])) for f in fams}
        macro_by_rep = [float(np.nanmean(list(p.values()))) for p in pers]
        stage_a.append({"backbone": b, "feature": feat, "C": C, "lomo_macro": float(np.mean(macro_by_rep)),
                        "lomo_sd": float(np.std(macro_by_rep)), **per_f})
    sa = pd.DataFrame(stage_a).sort_values("lomo_macro", ascending=False).reset_index(drop=True)
    best = float(sa.lomo_macro.iloc[0])
    keep = sa[sa.lomo_macro >= best - float(sel["tolerance"])].head(int(sel["stage_b_keep"]))
    log("stage A kept:\n" + keep[["backbone", "feature", "C", "lomo_macro", "lomo_sd"]].to_string())
    # stage B: training set by leave-one-processing-source-out false alarms (parallel over candidates)
    tasks = []
    for r in keep.itertuples(index=False):
        st = stores[r.backbone]
        jm = st.judged.to_numpy()
        Xj = descriptors(st[jm], r.feature)
        X_all = np.zeros((len(st), Xj.shape[1]), np.float32)
        X_all[jm] = Xj
        meta = st.drop(columns=["cls", "patch"])
        folds_all = group_folds(st.group, int(sel["folds"]), int(sel["seeds"][0]))
        for tset in cand["training_sets"]:
            tasks.append((r.backbone, r.feature, float(r.C), tset, meta, X_all, folds_all))
    log(f"stage B: {len(tasks)} (representation, training set) runs")
    stage_b = [row for row in Parallel(n_jobs=int(sel["n_jobs"]))(
        delayed(stage_b_run)(b_, f_, c_, t_, m_, x_, fo_, int(sel["folds"]), list(sel["sources"]), target)
        for b_, f_, c_, t_, m_, x_, fo_ in tasks) if row is not None]
    for row in stage_b:
        log(f"  stage B {row['backbone']} {row['feature']} C={row['C']} {row['training_set']}: LOMO "
            f"{row['lomo_macro']:.4f}, LOPSO max {row['lopso_max']:.3f}")
    sb = pd.DataFrame(stage_b)
    order = cfg["cost_order"]
    sb["cost"] = sb.apply(lambda x: (order["backbones"].index(x.backbone), order["features"].index(x.feature),
                                     order["training_sets"].index(x.training_set)), axis=1)
    eligible = sb[sb.lopso_max <= float(sel["lopso_max_fpr"])]
    constraint_met = not eligible.empty
    pool = eligible if constraint_met else sb[sb.lopso_max == sb.lopso_max.min()]
    top = float(pool.lomo_macro.max())
    tied = pool[pool.lomo_macro >= top - float(sel["tolerance"])]
    choice = tied.sort_values(["cost", "lomo_macro"], ascending=[True, False]).iloc[0]
    b, feat, C, tset = choice.backbone, choice.feature, float(choice.C), choice.training_set
    log(f"chosen: {b} {feat} C={C} training set {tset} (LOPSO constraint {'met' if constraint_met else 'NOT met'})")
    # threshold from out-of-fold reals of the chosen training set, each source weighted equally
    st = stores[b]
    jm = st.judged.to_numpy()
    Xj = descriptors(st[jm], feat)
    X_all = np.zeros((len(st), Xj.shape[1]), np.float32)
    X_all[jm] = Xj
    tr = training_rows(st, tset)
    Xt = X_all[tr.index.to_numpy()]
    folds = group_folds(tr.group, int(sel["folds"]), int(sel["seeds"][0]))
    oof = np.full(len(tr), np.nan)
    for f in range(int(sel["folds"])):
        m = fit(Xt[folds != f], tr[folds != f], C)
        oof[folds == f] = swap_score(m, Xt[folds == f])
    real = threshold_reals(tr)
    thr = weighted_quantile(oof[real], source_weights(tr[real]), 1 - target)
    y_tr = (tr.label == "fake").to_numpy().astype(int)
    in_domain = auc(y_tr[tr.source.to_numpy() == "v0"], oof[tr.source.to_numpy() == "v0"])
    model = fit(Xt, tr, C)
    ho_mask = ~st.split.isin(["train", "val"]).to_numpy()
    ho = st[ho_mask].copy()
    sc = np.full(len(ho), np.nan)
    hj = ho.judged.to_numpy()
    if hj.any():
        sc[hj] = swap_score(model, X_all[ho.index.to_numpy()[hj]])
    ho["score"] = sc
    ho["flagged"] = np.nan_to_num(sc, nan=-np.inf) >= thr
    tables, M = set_tables(ho)
    g = gates_v2(M, cfg)
    icfg = dict(inference_cfg(cfg), backbone=b, feature=feat)
    ref_scores = np.sort(oof[real & (tr.source.to_numpy() == "v0")]).astype(np.float32)
    bundle = {
        "model": model, "config": icfg, "config_hash": fs.config_hash(icfg), "threshold": float(thr),
        "real_reference_scores": ref_scores, "training_set": tset, "C": C,
        "trained_on": tr.groupby(["dataset", "family"]).size().to_dict(),
        "stage_a": sa.to_dict("records"), "stage_b": sb.drop(columns=["cost"]).to_dict("records"),
        "lopso_constraint_met": constraint_met,
        "metrics": {k: (list(v) if isinstance(v, tuple) else v) for k, v in M.items()},
        "gates": g, "in_domain_oof_auc": in_domain, "created": time.strftime("%Y-%m-%d %H:%M"),
        "versions": versions(), "backbone_sha256": weights_sha(b), "round": 4,
        "terms": ("trained on FaceForensics++ (non-commercial research use, per its terms of use) and DF40 "
                  "(CC BY-NC 4.0)"),
    }
    import joblib

    joblib.dump(bundle, out / "faceswap_v2.joblib")
    ho.drop(columns=["cls", "patch"]).to_csv(out / "held_out_scores.csv", index=False)
    L = [f"# Face-swap detector v2 — {time.strftime('%Y-%m-%d %H:%M')}", "",
         (f"Protocol: docs/13 §6.2-6.3. Selection on FaceForensics++ (+ DF40 FF-domain) train+val only; "
          f"every held-out rate is over judged clips (a face in >= {min_faces} of {cfg['n_frames']} frames) and "
          "printed with the total; AUC intervals resample identity groups; rate intervals are Wilson. "
          f"Config hash `{fs.config_hash(icfg)}`; versions {json.dumps(bundle['versions'])}."), "",
         "## Stage A: representation (LOMO macro AUC on the base set, repeated group folds)", "",
         md_table(sa), "",
         "## Stage B: training set (LOPSO false alarms at the 3% target; LOMO on the training set)", "",
         md_table(sb.drop(columns=["cost"])), "",
         (f"**Chosen: {b}, {feat}, C={C}, training set `{tset}`** (LOPSO <= {sel['lopso_max_fpr']:.0%} "
          f"{'met' if constraint_met else '**not met**: lowest LOPSO chosen'}). Threshold {thr:+.4f} "
          f"(out-of-fold reals of the training set at {target:.0%}, sources weighted equally). In-domain "
          f"out-of-fold AUC on v0 {in_domain:.3f}."), "",
         "## Held-out results", *tables, "", "## docs/13 §6.3 gates (cloud part)", "",
         "| gate | pass | value |", "|---|---|---|"]
    L += [f"| {n_} | {'n/a' if ok_ is None else ('yes' if ok_ else '**no**')} | {v_} |" for n_, ok_, v_ in g]
    (out / "faceswap_v2_report.md").write_text("\n".join(L) + "\n")
    (out / "faceswap_v2_metrics.json").write_text(json.dumps(bundle["metrics"], indent=1, default=str))
    print("\n".join(L))


# ---------------------------------------------------------------------- ship ---

CORE_GATES = ("1.", "2.", "2b.", "3.", "4a.")      # a missing measurement here is a failure, never "n/a"


def ship_decision(gates: list[tuple[str, bool | None, str]], g4a: dict,
                  round3_g4: dict | None) -> tuple[str, str]:
    """docs/13 §6.3 outcome: ("decisive" | "evidence" | "keep-round3" | "stage2", why)."""
    rows = [(n, ok) for n, ok, _ in gates if not n.startswith("4a.")] + [("4a. frozen list", bool(g4a["passed"]))]

    def ok_of(prefix: str) -> list[bool | None]:
        return [ok for n, ok in rows if n.startswith(prefix)]

    core_fail = [n for n, ok in rows if n.startswith(CORE_GATES) and ok is not True]
    other_fail = [n for n, ok in rows if not n.startswith(CORE_GATES) and ok is False]
    if not core_fail and not other_fail:
        return "decisive", "every applicable gate passed"
    first = [ok for p in ("1.", "2.", "2b.") for ok in ok_of(p)]
    if first and all(ok is True for ok in first):
        if round3_g4 is None:
            return "keep-round3", "evidence-only result, but no round-3 gate-4a measurement to compare with"
        v2_hi, r3_lo = float(g4a["ci"][1]), float(round3_g4["ci"][0])
        if v2_hi < r3_lo:
            return "evidence", (f"gates 1, 2, 2b passed; {', '.join(core_fail + other_fail)} failed; gate-4a rate "
                                f"{g4a['rate']:.1%} [..{v2_hi:.1%}] is below round 3's {round3_g4['rate']:.1%} "
                                f"[{r3_lo:.1%}..] with non-overlapping intervals")
        return "keep-round3", (f"evidence-only result whose gate-4a interval (upper {v2_hi:.1%}) overlaps round 3's "
                               f"(lower {r3_lo:.1%}): the round-3 bundle stays")
    return "stage2", f"gate 1, 2 or 2b failed ({', '.join(core_fail)}): stage 2 is tried under the same gates"


def cmd_ship(a) -> None:
    import joblib

    src = Path(a.bundle)
    g4 = json.loads(Path(a.gate4a).read_text())
    if g4.get("bundle_sha256") != hashlib.sha256(src.read_bytes()).hexdigest():
        raise SystemExit("the gate-4a result was measured on a different bundle file; rerun "
                         "scripts/check_faceswap_reals.py --manifest data/manifests/faceswap_gate4a.csv on this one")
    if not g4.get("frozen_manifest"):
        raise SystemExit("gate 4a must be measured on the frozen list (--manifest data/manifests/faceswap_gate4a.csv)")
    r3 = json.loads(Path(a.round3_gate4).read_text()) if a.round3_gate4 else None
    b = joblib.load(src)
    b.pop("_path", None)
    outcome, why = ship_decision(b["gates"], g4, r3)
    log(f"docs/13 §6.3 outcome: {outcome} — {why}")
    if outcome in ("keep-round3", "stage2"):
        Path(a.decision).write_text(json.dumps({"outcome": outcome, "why": why, "created": time.strftime("%Y-%m-%d %H:%M")},
                                              indent=1))
        raise SystemExit(f"not shipped: {outcome} ({why})")
    b["gates"] = [g for g in b["gates"] if not g[0].startswith("4a.")] + [
        ("4a. frozen round-3 gate-4 list flagged <= 3% of judged, n >= 200, no source > 6%", bool(g4["passed"]),
         f"{g4['k']}/{g4['n']} judged flagged")]
    b.update(mode=outcome, gate4=g4, shipped=time.strftime("%Y-%m-%d %H:%M"), ship_reason=why)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(b, a.out)
    Path(a.decision).write_text(json.dumps({"outcome": outcome, "why": why, "out": a.out,
                                            "created": time.strftime("%Y-%m-%d %H:%M")}, indent=1))
    log(f"shipped {src.name} -> {a.out}: mode {outcome}")


# ---------------------------------------------------------------------- main ---

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--root", required=True)
    f.add_argument("--datasets", nargs="+", default=["ffpp"], choices=["ffpp", "df40", "cdf2"])
    f.add_argument("--compression", nargs="+", default=["c23", "c40"])
    f.add_argument("--dfd-real", type=int, default=150)
    f.add_argument("--dfd-fake", type=int, default=300)
    f.add_argument("--threads", type=int, default=8)
    f.add_argument("--shard", default=None, help="I/N: fetch only shard I of N of the FF++ and DFD videos")
    f.add_argument("--links", default=None, help="JSON with df40_ids and cdf2_id (kept out of the repo)")
    f.add_argument("--config", default=None)
    m = sub.add_parser("manifest")
    m.add_argument("--ffpp", required=True, help="the FaceForensics++ root (splits/ inside)")
    m.add_argument("--df40", default=None)
    m.add_argument("--cdf2", default=None)
    m.add_argument("--out", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--manifest", required=True)
    e.add_argument("--features-dir", required=True)
    e.add_argument("--backbones", nargs="+", required=True)
    e.add_argument("--config", default=None)
    e.add_argument("--workers", type=int, default=3)
    e.add_argument("--device", default="auto")
    e.add_argument("--datasets", nargs="*", default=None)
    e.add_argument("--splits", nargs="*", default=None)
    e.add_argument("--variants", nargs="*", default=None, help="only these variants (default: the config's)")
    e.add_argument("--limit", type=int, default=0, help="at most N videos per dataset, split and family")
    e.add_argument("--window", type=float, default=None)
    e.add_argument("--stage2-dir", default=None, help="also save the v0 crops as JPEG for stage 2")
    e.add_argument("--strict", action="store_true", help="exit non-zero when any video had an error")
    e.add_argument("--time-budget-s", type=float, default=0, help="stop cleanly after this many seconds (cloud sessions)")
    s = sub.add_parser("select")
    s.add_argument("--manifest", required=True)
    s.add_argument("--features-dir", required=True)
    s.add_argument("--out-dir", required=True)
    s.add_argument("--backbones", nargs="*", default=None)
    s.add_argument("--config", default=None)
    s.add_argument("--allow-errors", action="store_true")
    s.add_argument("--smoke", action="store_true", help="tiny fold settings for a local pipeline check")
    sh = sub.add_parser("ship", help="apply docs/13 §6.3 and write the bundle the app loads")
    sh.add_argument("--bundle", required=True)
    sh.add_argument("--gate4a", required=True, help="JSON from check_faceswap_reals.py --manifest (frozen list)")
    sh.add_argument("--round3-gate4", default=str(REPO / "reports/faceswap_gate4a_round3.json"),
                    help="the round-3 bundle measured on the same frozen list")
    sh.add_argument("--out", default=str(REPO / "src/trubeat/models/faceswap.joblib"))
    sh.add_argument("--decision", default=str(REPO / "reports/faceswap_v2_decision.json"))
    a = ap.parse_args()
    {"fetch": cmd_fetch, "manifest": cmd_manifest, "extract": cmd_extract, "select": cmd_select,
     "ship": cmd_ship}[a.cmd](a)


if __name__ == "__main__":
    main()
