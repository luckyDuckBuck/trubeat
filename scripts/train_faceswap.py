"""Train and evaluate the face-swap detector (docs/12). Runs in Colab and locally.

Steps (each resumable; run in order):

  download  FaceForensics++ c23 (official servers, three mirrors tried) and
            a DeepFakeDetection subset; needs the FaceForensics++ terms,
            already accepted for this project.
  manifest  one row per video: path, label, family, split, identity group.
            FaceForensics++ splits are the official identity-disjoint ones;
            DeepFakeDetection is external (cross-dataset) only.
  extract   per video: sampled frames -> face crops -> embeddings for every
            backbone, saved per video (``features/<backbone>/<id>.npz``).
            ``--transform whatsapp`` re-encodes first (robustness check).
  select    the docs/12 §6 selection on train + val only (leave-one-
            manipulation-out AUC), the threshold from out-of-fold reals,
            the final fit, then the held-out test and DeepFakeDetection
            numbers and the §6 gates; writes the bundle and a report.
  evaluate  score an existing bundle on other features (e.g. the whatsapp
            variant) and append the numbers to the report.

Usage (local smoke run on the clips already on disk):
    python scripts/train_faceswap.py manifest --root data/raw/ffpp --out runs/faceswap/manifest.csv
    python scripts/train_faceswap.py extract --manifest runs/faceswap/manifest.csv \\
        --features-dir data/processed/faceswap --backbones vit_small_patch14_dinov2.lvd142m
    python scripts/train_faceswap.py select --manifest runs/faceswap/manifest.csv \\
        --features-dir data/processed/faceswap --out-dir runs/faceswap
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import random
import subprocess
import sys
import tempfile
import time
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Colab's kernel exports MPLBACKEND=module://matplotlib_inline...; MediaPipe
# imports pyplot, and that backend is not installed in the training venv, so
# every face lookup failed (2026-11-07, Colab run 1). Scripts are headless.
if os.environ.get("MPLBACKEND", "").startswith("module://"):
    os.environ["MPLBACKEND"] = "Agg"

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
from trubeat import faceswap as fs

SERVERS = ("http://kaldir.vc.in.tum.de/faceforensics/v3/",
           "http://canis.vc.in.tum.de:8100/v3/",
           "http://falas.cmpt.sfu.ca:8100/v3/")
SPLITS_URL = "https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/{}.json"
METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter")
SWAP_FAMILIES = ("Deepfakes", "FaceSwap", "FaceShifter")   # docs/12 §6 gate 1
DFD_REAL, DFD_FAKE = "original_sequences/actors", "manipulated_sequences/DeepFakeDetection"
Cs = (0.01, 0.1, 1.0)


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def load_cfg(path: str | None) -> dict:
    p = Path(path) if path else REPO / "configs/p6_faceswap.yaml"
    return yaml.safe_load(p.read_text())


# --------------------------------------------------------------- download ---

def _get(url: str, timeout: int = 60) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def pick_server() -> str:
    for base in SERVERS:
        try:
            _get(base + "misc/filelist.json", timeout=20)
            return base
        except Exception as exc:  # noqa: BLE001 - try the next mirror
            log(f"server {base} unreachable: {type(exc).__name__}")
    raise SystemExit("no FaceForensics++ server reachable")


def fetch(url: str, dest: Path) -> bool:
    if dest.is_file() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            data = _get(url, timeout=120)
            fd, tmp = tempfile.mkstemp(dir=str(dest.parent))
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, dest)
            return True
        except Exception:  # noqa: BLE001 - retry, then give up on this file
            time.sleep(2 * (attempt + 1))
    return False


def cmd_download(a) -> None:
    root = Path(a.root)
    base = pick_server()
    log(f"server {base}")
    for split in ("train", "val", "test"):
        dest = root / "splits" / f"{split}.json"
        if not dest.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(_get(SPLITS_URL.format(split)))
    pairs = json.loads(_get(base + "misc/filelist.json"))
    jobs = []
    if "original" in a.datasets:
        for x in sorted({v for p in pairs for v in p}):
            jobs.append(f"original_sequences/youtube/c23/videos/{x}.mp4")
    for m in METHODS:
        if m in a.datasets:
            for p in pairs:
                jobs += [f"manipulated_sequences/{m}/c23/videos/{p[0]}_{p[1]}.mp4",
                         f"manipulated_sequences/{m}/c23/videos/{p[1]}_{p[0]}.mp4"]
    if a.dfd_real or a.dfd_fake:
        names = json.loads(_get(base + "misc/deepfake_detection_filenames.json"))
        rng = random.Random(7)
        real = sorted(names["actors"])
        fake = sorted(names["DeepFakesDetection"])
        jobs += [f"{DFD_REAL}/c23/videos/{n}.mp4" for n in rng.sample(real, min(a.dfd_real, len(real)))]
        jobs += [f"{DFD_FAKE}/c23/videos/{n}.mp4" for n in rng.sample(fake, min(a.dfd_fake, len(fake)))]
    log(f"{len(jobs)} files")
    ok = 0
    t0 = time.time()
    with ThreadPoolExecutor(a.threads) as ex:
        futs = {ex.submit(fetch, base + j, root / j): j for j in jobs}
        for k, f in enumerate(as_completed(futs), 1):
            ok += bool(f.result())
            if k % 250 == 0:
                log(f"  {k}/{len(jobs)} ({ok} ok) in {time.time() - t0:.0f}s")
    log(f"downloaded {ok}/{len(jobs)}")


# --------------------------------------------------------------- manifest ---

def build_manifest(root: Path) -> pd.DataFrame:
    rows = []
    for split in ("train", "val", "test"):
        f = root / "splits" / f"{split}.json"
        if not f.is_file():
            raise SystemExit(f"missing {f}: run `download` first (or copy the official splits)")
        for a_, b_ in json.loads(f.read_text()):
            a, b = str(a_), str(b_)
            group = "ffpp:" + "_".join(sorted((a, b)))
            for x in (a, b):
                rows.append({"video_id": f"original_{x}", "family": "original", "label": "real",
                             "path": root / f"original_sequences/youtube/c23/videos/{x}.mp4",
                             "split": split, "group": group, "dataset": "ffpp"})
            for m in METHODS:
                for t, s in ((a, b), (b, a)):
                    rows.append({"video_id": f"{m}_{t}_{s}", "family": m, "label": "fake",
                                 "path": root / f"manipulated_sequences/{m}/c23/videos/{t}_{s}.mp4",
                                 "split": split, "group": group, "dataset": "ffpp"})
    for sub, label, fam in ((DFD_REAL, "real", "DFD-original"), (DFD_FAKE, "fake", "DFD")):
        for p in sorted((root / sub / "c23/videos").glob("*.mp4")):
            actor = p.stem.split("_")[0]
            rows.append({"video_id": f"{fam}_{p.stem}", "family": fam, "label": label, "path": p,
                         "split": "external", "group": f"dfd:{actor}", "dataset": "dfd"})
    df = pd.DataFrame(rows)
    df = df[df.path.map(lambda p: Path(p).is_file())].reset_index(drop=True)
    df["path"] = df.path.map(str)
    return df


def cmd_manifest(a) -> None:
    df = build_manifest(Path(a.root))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    log(f"{len(df)} videos -> {a.out}")
    print(df.groupby(["split", "family"]).size().unstack(fill_value=0).to_string())


# ---------------------------------------------------------------- extract ---

_W: dict = {}


def _init_worker(cfg: dict, transform: str | None, cnn_size: int = 0) -> None:
    warnings.filterwarnings("ignore")
    _W["cfg"], _W["transform"], _W["cnn_size"] = cfg, transform, cnn_size
    _W["finder"] = fs.FaceFinder(cfg)


def whatsapp_like(src: str, dst: str) -> None:
    """The docs/12 robustness variant: 640 px wide, CRF 30 (a messaging re-share)."""
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", src,
                    "-vf", "scale=640:-2", "-c:v", "libx264", "-crf", "30", "-preset", "veryfast",
                    "-an", dst], check=True)


def _jpegs(sampled, crops, size: int) -> list[bytes]:
    """Stage-2 crops: the same face boxes cut at the CNN's input size, JPEG q95."""
    import cv2

    by_index = dict(zip(sampled.index, sampled.frames))
    out = []
    for c in crops:
        x0, y0, s, _ = c.box
        img = fs.cut_square(by_index[c.frame_index], x0 + s / 2.0, y0 + s / 2.0, float(s), size)
        out.append(cv2.imencode(".jpg", img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes())
    return out


def crops_for(row: tuple[str, str]) -> tuple:
    vid, path = row
    cfg = _W["cfg"]
    try:
        with tempfile.TemporaryDirectory() as tmp:
            if _W["transform"] == "whatsapp":
                out = str(Path(tmp) / "v.mp4")
                whatsapp_like(path, out)
                path = out
            s = fs.sample_frames(path, int(cfg["n_frames"]), float(cfg["max_seconds"]))
        crops = fs.face_crops(s, _W["finder"], cfg)
        if not crops:
            return vid, None, None, len(s.frames), "", []
        jpgs = _jpegs(s, crops, _W["cnn_size"]) if _W["cnn_size"] else []
        return (vid, np.stack([c.crop for c in crops]), np.array([c.time_s for c in crops], np.float32),
                len(s.frames), "", jpgs)
    except Exception as exc:  # noqa: BLE001 - one bad file must not stop the run
        return vid, None, None, 0, f"{type(exc).__name__}: {exc}", []


def extracted(out: Path, backbones: list[str], crops_dir: Path | None, vid: str) -> bool:
    """A video counts as done only when every backbone's file exists, its
    stage-2 crops are saved, and it did not fail (failed videos are retried)."""
    files = [out / b / f"{vid}.npz" for b in backbones]
    if not all(f.is_file() for f in files):
        return False
    if crops_dir is not None and not (crops_dir / vid / "done").is_file():
        return False
    with np.load(files[0], allow_pickle=False) as z:
        return str(z["error"]) == ""


def cmd_extract(a) -> None:
    cfg = fs.merged_config({k: v for k, v in load_cfg(a.config).items() if k != "candidates"})
    df = pd.read_csv(a.manifest)
    if a.splits:
        df = df[df.split.isin(a.splits)]
    if a.limit:
        df = df.groupby(["split", "family"], group_keys=False).head(a.limit)
    out = Path(a.features_dir)
    if a.checkpoint_tar and Path(a.checkpoint_tar).is_file() and not any(out.glob("*/*.npz")):
        log(f"resuming from {a.checkpoint_tar}")
        out.mkdir(parents=True, exist_ok=True)
        subprocess.run(["tar", "-xf", a.checkpoint_tar, "-C", str(out)], check=True)
    device = a.device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    embedders = {b: fs.FaceEmbedder(b, "cls+patch", int(cfg["torch_threads"]), device=device,
                                    img_size=int(cfg["crop_size"])) for b in a.backbones}
    for b, e in embedders.items():  # load every backbone now: a missing one stops here, not mid-run
        log(f"{b}: {e.dim}-dim features")
        (out / b).mkdir(parents=True, exist_ok=True)
    crops_dir = Path(a.save_crops) if a.save_crops else None

    todo = [(r.video_id, r.path) for r in df.itertuples()
            if not extracted(out, a.backbones, crops_dir, r.video_id)]
    log(f"{len(df)} videos, {len(todo)} to extract, backbones {a.backbones}, device {device}")
    t0, errors = time.time(), 0
    cnn_size = int(a.cnn_crop_size) if crops_dir else 0
    with mp.get_context("spawn").Pool(a.workers, initializer=_init_worker,
                                      initargs=(cfg, a.transform, cnn_size)) as pool:
        for k, (vid, crops, times, n_sampled, err, jpgs) in enumerate(
                pool.imap_unordered(crops_for, todo, chunksize=2), 1):
            if err:
                errors += 1
                if errors <= 5:  # show the reason, not just a count
                    log(f"  error on {vid}: {err[:400]}")
                if errors == k >= 50:
                    raise SystemExit(f"the first {k} videos all failed, so the setup is broken; "
                                     f"last error: {err[:400]}")
            if crops_dir is not None:
                d = crops_dir / vid
                d.mkdir(parents=True, exist_ok=True)
                for j, data in enumerate(jpgs):
                    (d / f"{j:02d}.jpg").write_bytes(data)
                (d / "done").write_text(str(len(jpgs)))
            for b, e in embedders.items():
                dest = out / b / f"{vid}.npz"
                if crops is None:
                    np.savez(dest, cls=np.zeros((0, 1), np.float16), patch=np.zeros((0, 1), np.float16),
                             time_s=np.zeros(0, np.float32), n_sampled=n_sampled, error=err)
                    continue
                emb = e.embed(crops)
                d = emb.shape[1] // 2
                np.savez(dest, cls=emb[:, :d].astype(np.float16), patch=emb[:, d:].astype(np.float16),
                         time_s=times, n_sampled=n_sampled, error="")
            if a.checkpoint_tar and (k % a.checkpoint_every == 0 or k == len(todo)):
                checkpoint(out, Path(a.checkpoint_tar))
            if k % 100 == 0 or k == len(todo):
                rate = k / (time.time() - t0)
                log(f"  {k}/{len(todo)} ({rate:.2f} videos/s, {errors} errors, "
                    f"~{(len(todo) - k) / max(rate, 1e-9) / 60:.0f} min left)")


def checkpoint(features_dir: Path, tar_path: Path) -> None:
    """Pack the feature files into one tar on (slow, FUSE-mounted) Drive:
    one big write instead of thousands of small ones; resumable."""
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tar_path.with_suffix(".tmp")
    subprocess.run(["tar", "-cf", str(tmp), "-C", str(features_dir), "."], check=True)
    os.replace(tmp, tar_path)
    log(f"  checkpoint -> {tar_path}")


# ----------------------------------------------------------------- select ---

def load_features(df: pd.DataFrame, fdir: Path, backbone: str, min_faces: int) -> dict:
    """Per video: mean CLS and mean patch token over its crops (each crop's
    parts were L2-normalised at extraction), and the face count."""
    cls, patch, nf = [], [], []
    for vid in df.video_id:
        f = fdir / backbone / f"{vid}.npz"
        if not f.is_file():
            cls.append(None), patch.append(None), nf.append(-1)
            continue
        with np.load(f, allow_pickle=False) as z:
            c, p = z["cls"].astype(np.float32), z["patch"].astype(np.float32)
        n = len(c) if c.shape[1] > 1 else 0
        nf.append(n)
        cls.append(c.mean(axis=0) if n >= min_faces else None)
        patch.append(p.mean(axis=0) if n >= min_faces else None)
    return {"cls": cls, "patch": patch, "n_faces": np.array(nf)}


def descriptor(feats: dict, feature: str, rows: np.ndarray) -> np.ndarray:
    parts = [np.stack([feats["cls"][i] for i in rows])]
    if feature == "cls+patch":
        parts.append(np.stack([feats["patch"][i] for i in rows]))
    return np.hstack(parts)


def lomo_auc(sel: pd.DataFrame, X: np.ndarray, C: float, n_folds: int = 5) -> dict[str, float]:
    """Leave-one-manipulation-out: train on the other fake families (and the
    reals of the other identity folds), score the held-out family against
    held-out real identities. Mean over folds, per family."""
    import train_aigen as ta
    from sklearn.model_selection import GroupKFold

    y = (sel.label == "fake").to_numpy().astype(int)
    fams = sorted(f for f in sel.family.unique() if f != "original")
    folds = list(GroupKFold(n_folds).split(sel, groups=sel.group))
    out = {}
    for fam in fams:
        aucs = []
        for tr, te in folds:
            trm = np.zeros(len(sel), bool)
            trm[tr] = True
            trm &= (sel.family != fam).to_numpy()
            tem = np.zeros(len(sel), bool)
            tem[te] = True
            tem &= ((sel.label == "real") | (sel.family == fam)).to_numpy()
            if y[tem].min() == y[tem].max():
                continue
            m = ta.make_model(C).fit(X[trm], sel.label.to_numpy()[trm])
            aucs.append(ta.safe_auc(y[tem], swap_score(m, X[tem])))
        out[fam] = float(np.nanmean(aucs)) if aucs else float("nan")
    return out


def swap_score(model, X: np.ndarray) -> np.ndarray:
    s = model.decision_function(X)
    return s if list(model.classes_)[1] == "fake" else -s


def oof_scores(sel: pd.DataFrame, X: np.ndarray, C: float, n_folds: int = 5) -> np.ndarray:
    import train_aigen as ta
    from sklearn.model_selection import GroupKFold

    s = np.full(len(sel), np.nan)
    for tr, te in GroupKFold(n_folds).split(sel, groups=sel.group):
        m = ta.make_model(C).fit(X[tr], sel.label.to_numpy()[tr])
        s[te] = swap_score(m, X[te])
    return s


def held_out_tables(df: pd.DataFrame, scores: np.ndarray, thr: float) -> tuple[list[str], dict]:
    """Per family AUC (vs the same dataset's reals) and recall; FPR on reals.
    Clips without enough faces are never flagged (the app does not decide)."""
    import train_aigen as ta

    df = df.assign(score=scores, flagged=np.nan_to_num(scores, nan=-np.inf) >= thr, has_face=~np.isnan(scores))
    L, metrics = [], {}
    for ds, title in (("ffpp", "FaceForensics++ c23 test (official split)"),
                      ("dfd", "DeepFakeDetection (cross-dataset)")):
        d = df[df.dataset == ds]
        if d.empty:
            continue
        real = d[d.label == "real"]
        k, n = int(real.flagged.sum()), len(real)
        lo, hi = ta.wilson(k, n)
        metrics[f"{ds}_fpr"] = (k, n)
        L += ["", f"### {title}", "",
              (f"Reals flagged: **{k}/{n} = {k / max(n, 1):.1%}** [{lo:.1%}, {hi:.1%}] "
               f"({int((~real.has_face).sum())} without enough faces)."), "",
              "| family | n | no face | AUC vs reals (95% CI) | caught at threshold (95% CI) |",
              "|---|---|---|---|---|"]
        for fam in sorted(f for f in d.family.unique() if d[d.family == f].label.iloc[0] == "fake"):
            fk = d[d.family == fam]
            both = pd.concat([real, fk])
            both = both[both.has_face]
            y = (both.label == "fake").to_numpy().astype(int)
            auc = ta.safe_auc(y, both.score.to_numpy())
            alo, ahi = ta.bootstrap_auc(y, both.score.to_numpy()) if 0 < y.sum() < len(y) else (np.nan, np.nan)
            kk, nn = int(fk.flagged.sum()), len(fk)
            rlo, rhi = ta.wilson(kk, nn)
            metrics[f"{ds}_{fam}_auc"] = auc
            metrics[f"{ds}_{fam}_recall"] = kk / max(nn, 1)
            L.append(f"| {fam} | {nn} | {int((~fk.has_face).sum())} | {auc:.3f} [{alo:.3f}, {ahi:.3f}] | "
                     f"{kk}/{nn} = {kk / max(nn, 1):.1%} [{rlo:.1%}, {rhi:.1%}] |")
    return L, metrics


def gates(metrics: dict) -> list[tuple[str, bool | None, str]]:
    out = []
    for fam in SWAP_FAMILIES:
        v = metrics.get(f"ffpp_{fam}_auc")
        out.append((f"1. FaceForensics++ test AUC >= 0.85 on {fam}",
                    None if v is None else bool(v >= 0.85), "n/a" if v is None else f"{v:.3f}"))
    k, n = metrics.get("ffpp_fpr", (None, None))
    out.append(("2. FaceForensics++ test reals flagged <= 5/140", None if k is None else bool(k <= 5 * n / 140),
                "n/a" if k is None else f"{k}/{n}"))
    v = metrics.get("dfd_DFD_auc")
    k, n = metrics.get("dfd_fpr", (None, None))
    ok = None if v is None or k is None else bool(v >= 0.70 and k / max(n, 1) <= 0.08)
    out.append(("3. DeepFakeDetection AUC >= 0.70 and reals flagged <= 8%", ok,
                "n/a" if ok is None else f"AUC {v:.3f}, {k}/{n} flagged"))
    out.append(("4. real non-FaceForensics clips with a face flagged <= 3% (n >= 200)", None,
                "measured on the laptop (scripts/check_faceswap_reals.py)"))
    return out


def check_manifest(df: pd.DataFrame) -> None:
    """AGENTS.md §6: the identity-leakage validator runs before every
    training job. FaceForensics++ rows carry the official splits (group =
    the source/target pair); DeepFakeDetection rows are external and must
    not share a group with them. Raises ValueError on leakage."""
    from trubeat import eval as ev

    ff = df[df.split != "external"]
    ev.validate_manifest(ff.rename(columns={"group": "split_group"}))
    shared = set(ff.group) & set(df[df.split == "external"].group)
    if shared:
        raise ValueError(f"identity leakage: {len(shared)} group(s) in both the FaceForensics++ splits "
                         f"and the external set: {sorted(shared)[:5]}")


def gate_passed(bundle: dict, k: int) -> bool:
    """True when every entry of docs/12 §6 gate ``k`` in the bundle passed
    (gate 1 has one entry per face-swap set; "n/a" counts as not passed)."""
    vals = [ok for n, ok, _ in bundle["gates"] if n.startswith(f"{k}.")]
    return bool(vals) and all(v is True for v in vals)


def cmd_select(a) -> None:
    import joblib
    import sklearn
    import train_aigen as ta

    warnings.filterwarnings("ignore")
    raw = load_cfg(a.config)
    base_cfg = fs.merged_config({k: v for k, v in raw.items() if k != "candidates"})
    backbones = a.backbones or raw.get("candidates", [base_cfg["backbone"]])
    df = pd.read_csv(a.manifest)
    check_manifest(df)
    fdir, out = Path(a.features_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trainval = (df.split.isin(["train", "val"])).to_numpy()
    feats, search = {}, []
    for b in backbones:
        if not (fdir / b).is_dir():
            log(f"skip {b}: no features")
            continue
        feats[b] = load_features(df, fdir, b, int(base_cfg["min_faces"]))
        ok = np.array([c is not None for c in feats[b]["cls"]])
        sel_rows = np.where(trainval & ok)[0]
        sel = df.iloc[sel_rows].reset_index(drop=True)
        for feature in fs.FEATURES:
            X = descriptor(feats[b], feature, sel_rows)
            for C in Cs:
                per = lomo_auc(sel, X, C)
                macro = float(np.nanmean(list(per.values())))
                search.append({"backbone": b, "feature": feature, "C": C, "lomo_macro": macro, **per})
                log(f"  {b:38s} {feature:9s} C={C:<5} LOMO macro {macro:.4f}  " +
                    " ".join(f"{k}={v:.3f}" for k, v in per.items()))
    if not search:
        raise SystemExit("no features found")
    res = pd.DataFrame(search).sort_values(["lomo_macro", "C"], ascending=[False, True])
    best = res.iloc[0]
    b, feature, C = best.backbone, best.feature, float(best.C)
    ok = np.array([c is not None for c in feats[b]["cls"]])
    sel_rows = np.where(trainval & ok)[0]
    sel = df.iloc[sel_rows].reset_index(drop=True)
    X = descriptor(feats[b], feature, sel_rows)
    oof = oof_scores(sel, X, C)
    oof_real = oof[(sel.label == "real").to_numpy()]
    thr = float(np.quantile(oof_real, 1 - float(base_cfg["target_fpr"])))
    in_domain_auc = ta.safe_auc((sel.label == "fake").to_numpy().astype(int), oof)
    model = ta.make_model(C).fit(X, sel.label.to_numpy())
    ho = ~trainval
    ho_rows = np.where(ho)[0]
    scores = np.full(len(df), np.nan)
    has = [i for i in ho_rows if feats[b]["cls"][i] is not None]
    if has:
        scores[has] = swap_score(model, descriptor(feats[b], feature, np.array(has)))
    ho_lines, metrics = held_out_tables(df.iloc[ho_rows].reset_index(drop=True), scores[ho_rows], thr)
    g = gates(metrics)
    cfg = dict(base_cfg, backbone=b, feature=feature)
    trained_on = ", ".join(f"{f} ({n})" for f, n in sel.family.value_counts().sort_index().items())
    bundle = {
        "model": model, "config": cfg, "config_hash": fs.config_hash(cfg), "threshold": thr,
        "real_reference_scores": np.sort(oof_real).astype(np.float32),
        "trained_on": f"FaceForensics++ c23 train+val: {trained_on}", "selection": res.to_dict("records"),
        "metrics": {k: (list(v) if isinstance(v, tuple) else v) for k, v in metrics.items()},
        "gates": [(n_, ok_, v_) for n_, ok_, v_ in g], "in_domain_oof_auc": in_domain_auc,
        "created": time.strftime("%Y-%m-%d %H:%M"),
        "terms": "trained on FaceForensics++ (non-commercial research use, per its terms of use)",
        "sklearn_version": sklearn.__version__,  # checked on load (trubeat.bundles, docs/13 D-05)
    }
    joblib.dump(bundle, out / "faceswap.joblib")
    n_noface = int((~ok[trainval]).sum())
    L = [f"# Face-swap detector — {time.strftime('%Y-%m-%d %H:%M')}", "",
         (f"Protocol: docs/12 §6. Selection on FaceForensics++ train+val only ({len(sel)} videos with a "
          f"face; {n_noface} without enough faces left out). Test and DeepFakeDetection never used for "
          f"selection or the threshold. Config hash `{fs.config_hash(cfg)}`."), "",
         "## Selection (leave-one-manipulation-out AUC, train+val)", "",
         "| backbone | feature | C | LOMO macro | " + " | ".join(c for c in res.columns if c not in
                                                              ("backbone", "feature", "C", "lomo_macro")) + " |",
         "|---" * (len(res.columns)) + "|"]
    fam_cols = [c for c in res.columns if c not in ("backbone", "feature", "C", "lomo_macro")]
    for r in res.itertuples(index=False):
        d = r._asdict()
        L.append(f"| {d['backbone']} | {d['feature']} | {d['C']} | {d['lomo_macro']:.4f} | " +
                 " | ".join(f"{d[c]:.3f}" for c in fam_cols) + " |")
    L += ["", (f"Selected: **{b}, {feature}, C={C}**. In-domain out-of-fold AUC {in_domain_auc:.3f}. "
               f"Threshold {thr:+.4f} (out-of-fold reals at {base_cfg['target_fpr']:.0%} false alarms)."),
          "", "## Held-out results", *ho_lines, "", "## docs/12 §6 gates", "",
          "| gate | pass | value |", "|---|---|---|"]
    L += [f"| {n_} | {'n/a' if ok_ is None else ('yes' if ok_ else '**no**')} | {v_} |" for n_, ok_, v_ in g]
    (out / "faceswap_report.md").write_text("\n".join(L) + "\n")
    (out / "faceswap_metrics.json").write_text(json.dumps(bundle["metrics"], indent=1))
    print("\n".join(L))


# --------------------------------------------------------------- finetune ---

def _clip_items(df: pd.DataFrame, crops_dir: Path, min_faces: int) -> dict[str, list[Path]]:
    out = {}
    for vid in df.video_id:
        files = sorted((crops_dir / vid).glob("*.jpg"))
        if len(files) >= min_faces:
            out[vid] = files
    return out


def cmd_finetune(a) -> None:
    """Stage 2 (docs/12 §3): fine-tune a CNN on the face crops. Trains on the
    train split; the validation split picks the epoch and sets the threshold
    (its reals at the target false-alarm rate); test and DeepFakeDetection
    are scored once at the end."""
    import io

    import joblib
    import timm
    import torch
    import train_aigen as ta
    from PIL import Image
    from safetensors.torch import save_file
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    from torchvision import transforms as T

    warnings.filterwarnings("ignore")
    raw = load_cfg(a.config)
    base_cfg = fs.merged_config({k: v for k, v in raw.items() if k != "candidates"})
    min_faces = int(base_cfg["min_faces"])
    df = pd.read_csv(a.manifest)
    check_manifest(df)
    items = _clip_items(df, Path(a.crops_dir), min_faces)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(7)
    model = timm.create_model(a.arch, pretrained=True, num_classes=1).to(dev)
    dcfg = timm.data.resolve_data_config({}, model=model)
    mean, std = list(dcfg["mean"]), list(dcfg["std"])

    class RandomJPEG:
        def __init__(self, p=0.5, q=(50, 95)):
            self.p, self.q = p, q

        def __call__(self, img):
            if random.random() > self.p:
                return img
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=random.randint(*self.q))
            return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")

    train_tf = T.Compose([T.RandomResizedCrop(a.size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
                          T.RandomHorizontalFlip(), T.ColorJitter(0.1, 0.1, 0.1, 0.02), RandomJPEG(),
                          T.RandomApply([T.GaussianBlur(3)], p=0.1), T.ToTensor(), T.Normalize(mean, std)])
    eval_tf = T.Compose([T.Resize((a.size, a.size)), T.ToTensor(), T.Normalize(mean, std)])

    class Crops(Dataset):
        def __init__(self, rows, tf):
            self.rows, self.tf = rows, tf

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            path, y, k = self.rows[i]
            return self.tf(Image.open(path).convert("RGB")), torch.tensor(float(y)), k

    def rows_for(split_mask):
        rows = []
        for r in df[split_mask].itertuples():  # r.Index is the manifest row (RangeIndex)
            for f in items.get(r.video_id, []):
                rows.append((f, int(r.label == "fake"), r.Index))
        return rows

    tr_rows = rows_for(df.split == "train")
    y = np.array([r[1] for r in tr_rows])
    w = np.where(y == 1, 0.5 / max(y.sum(), 1), 0.5 / max((1 - y).sum(), 1))
    sampler = WeightedRandomSampler(torch.tensor(w, dtype=torch.double), num_samples=len(tr_rows), replacement=True)
    tr = DataLoader(Crops(tr_rows, train_tf), batch_size=a.batch, sampler=sampler, num_workers=a.workers,
                    pin_memory=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * len(tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=dev == "cuda")
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def clip_scores(mask) -> np.ndarray:
        """Mean crop logit per manifest row (NaN: not enough faces)."""
        rows = rows_for(mask)
        dl = DataLoader(Crops(rows, eval_tf), batch_size=a.batch * 2, num_workers=a.workers)
        sums, counts = {}, {}
        model.eval()
        with torch.inference_mode(), torch.autocast(dev, enabled=dev == "cuda"):
            for x, _, k in dl:
                s = model(x.to(dev, non_blocking=True)).float().cpu().numpy().reshape(-1)
                for kk, ss in zip(k.tolist(), s):
                    sums[kk] = sums.get(kk, 0.0) + float(ss)
                    counts[kk] = counts.get(kk, 0) + 1
        out = np.full(len(df), np.nan)
        for kk, total in sums.items():
            out[kk] = total / counts[kk]
        return out

    val_mask = (df.split == "val").to_numpy()
    best_auc, best_state, history = -1.0, None, []
    for ep in range(a.epochs):
        model.train()
        t0, run = time.time(), 0.0
        for i, (x, yy, _) in enumerate(tr, 1):
            x, yy = x.to(dev, non_blocking=True), yy.to(dev)
            with torch.autocast(dev, enabled=dev == "cuda"):
                loss = loss_fn(model(x).squeeze(1), yy)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += float(loss)
            if i % 200 == 0:
                log(f"  epoch {ep + 1} step {i}/{len(tr)} loss {run / i:.4f}")
        vs = clip_scores(val_mask)
        v = df[val_mask].assign(s=vs[val_mask]).dropna(subset=["s"])
        auc = ta.safe_auc((v.label == "fake").to_numpy().astype(int), v.s.to_numpy())
        history.append({"epoch": ep + 1, "train_loss": run / max(len(tr), 1), "val_auc": auc,
                        "seconds": time.time() - t0})
        log(f"epoch {ep + 1}: loss {run / max(len(tr), 1):.4f}, val clip AUC {auc:.4f} ({time.time() - t0:.0f}s)")
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().half().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict({k: v.float() for k, v in best_state.items()})
    vs = clip_scores(val_mask)
    val_real = vs[val_mask & (df.label == "real").to_numpy()]
    val_real = val_real[~np.isnan(val_real)]
    thr = float(np.quantile(val_real, 1 - float(base_cfg["target_fpr"])))
    ho = (~df.split.isin(["train", "val"])).to_numpy()
    hs = clip_scores(ho)
    ho_lines, metrics = held_out_tables(df[ho].reset_index(drop=True), hs[ho], thr)
    g = gates(metrics)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    wfile = out / a.weights_name
    save_file({k: v.contiguous() for k, v in best_state.items()}, str(wfile))
    sha = hashlib.sha256(wfile.read_bytes()).hexdigest()
    cfg = dict(base_cfg, crop_size=a.size, backbone=a.arch)
    trained_on = ", ".join(f"{f} ({n})" for f, n in df[df.split == "train"].family.value_counts().sort_index().items())
    bundle = {
        "kind": "cnn", "model": None, "config": cfg, "config_hash": fs.config_hash(cfg), "threshold": thr,
        "cnn": {"arch": a.arch, "file": a.weights_name, "sha256": sha, "url": "", "mean": mean, "std": std},
        "real_reference_scores": np.sort(val_real).astype(np.float32),
        "trained_on": f"FaceForensics++ c23 train: {trained_on}", "history": history, "val_auc": best_auc,
        "metrics": {k: (list(v) if isinstance(v, tuple) else v) for k, v in metrics.items()},
        "gates": [(n_, ok_, v_) for n_, ok_, v_ in g], "created": time.strftime("%Y-%m-%d %H:%M"),
        "terms": "trained on FaceForensics++ (non-commercial research use, per its terms of use)",
    }
    joblib.dump(bundle, out / "faceswap_cnn.joblib")
    L = ["", f"# Stage 2: fine-tuned {a.arch} at {a.size} px — {time.strftime('%Y-%m-%d %H:%M')}", "",
         (f"docs/12 §3 stage 2. Trained on the train split ({len(tr_rows)} crops, balanced sampling, "
          f"{a.epochs} epochs); the epoch with the best validation clip AUC ({best_auc:.3f}) is kept; "
          f"threshold {thr:+.4f} = validation reals at {base_cfg['target_fpr']:.0%} false alarms. "
          "Test and DeepFakeDetection scored once."), "",
         "| epoch | train loss | val clip AUC | seconds |", "|---|---|---|---|"]
    L += [f"| {h['epoch']} | {h['train_loss']:.4f} | {h['val_auc']:.4f} | {h['seconds']:.0f} |" for h in history]
    L += ["", "## Held-out results (stage 2)", *ho_lines, "", "## docs/12 §6 gates (stage 2)", "",
          "| gate | pass | value |", "|---|---|---|"]
    L += [f"| {n_} | {'n/a' if ok_ is None else ('yes' if ok_ else '**no**')} | {v_} |" for n_, ok_, v_ in g]
    with open(out / "faceswap_report.md", "a") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


def cmd_evaluate(a) -> None:
    warnings.filterwarnings("ignore")
    bundle = fs.load_bundle(a.bundle)
    if bundle is None:
        raise SystemExit(f"no bundle at {a.bundle}")
    cfg = bundle["config"]
    df = pd.read_csv(a.manifest)
    ho = df[~df.split.isin(["train", "val"])].reset_index(drop=True)
    scores = np.full(len(ho), np.nan)
    if bundle.get("kind") == "cnn":  # stage 2: score the saved crops directly
        import cv2

        if not a.crops_dir:
            raise SystemExit("a stage-2 bundle needs --crops-dir")
        # weights next to the bundle, checked against its SHA-256
        scorer = fs.CNNScorer.from_bundle(bundle, device=a.device if a.device != "auto" else "cpu")
        for vid, files in _clip_items(ho, Path(a.crops_dir), int(cfg["min_faces"])).items():
            crops = np.stack([cv2.imread(str(f))[:, :, ::-1] for f in files])
            scores[int(np.where(ho.video_id == vid)[0][0])] = float(scorer.score(crops).mean())
    else:
        feats = load_features(ho, Path(a.features_dir), cfg["backbone"], int(cfg["min_faces"]))
        has = [i for i in range(len(ho)) if feats["cls"][i] is not None]
        if has:
            scores[has] = swap_score(bundle["model"], descriptor(feats, cfg["feature"], np.array(has)))
    lines, _ = held_out_tables(ho, scores, float(bundle["threshold"]))
    text = "\n".join([f"## {a.title}", "", f"Bundle `{a.bundle}`, threshold {bundle['threshold']:+.4f} fixed.",
                      *lines]) + "\n"
    with open(a.report, "a") as f:
        f.write("\n" + text)
    print(text)


def cmd_ship(a) -> None:
    """docs/12 §6 outcome -> the installed bundle. Gates 1-3 come from the
    bundle (measured on Colab), gate 4 from the laptop's check of this same
    bundle file. All pass -> ``decisive``; gates 1-2 pass -> ``evidence``;
    otherwise nothing is shipped. A stage-2 bundle's weights must match its
    SHA-256; they are copied into ~/.cache/trubeat and the bundle records
    the release URL ``trubeat fetch-models`` downloads them from."""
    import shutil

    import joblib

    src = Path(a.bundle)
    g4 = json.loads(Path(a.gate4).read_text())
    if g4.get("bundle_sha256") != hashlib.sha256(src.read_bytes()).hexdigest():
        raise SystemExit("the gate-4 result was measured on a different bundle file; rerun "
                         "scripts/check_faceswap_reals.py on this one")
    b = joblib.load(src)
    b.pop("_path", None)
    g1, g2, g3, g4ok = gate_passed(b, 1), gate_passed(b, 2), gate_passed(b, 3), bool(g4["passed"])
    if not (g1 and g2):
        raise SystemExit(f"not shipped (docs/12 §6): gate 1 {'pass' if g1 else 'fail'}, "
                         f"gate 2 {'pass' if g2 else 'fail'}")
    mode = "decisive" if (g3 and g4ok) else "evidence"
    b["gates"] = [(n, g4ok, f"{g4['k']}/{g4['n']} flagged") if n.startswith("4.") else (n, ok, v)
                  for n, ok, v in b["gates"]]
    b.update(mode=mode, gate4=g4, shipped=time.strftime("%Y-%m-%d %H:%M"))
    if b.get("kind") == "cnn":
        w = src.parent / b["cnn"]["file"]
        if not w.is_file() or hashlib.sha256(w.read_bytes()).hexdigest() != b["cnn"]["sha256"]:
            raise SystemExit(f"{w} is missing or does not match the bundle's SHA-256")
        if not a.url:
            raise SystemExit("a stage-2 bundle needs --url (the release asset the weights download from)")
        b["cnn"]["url"] = a.url
        cache = Path.home() / ".cache" / "trubeat"
        cache.mkdir(parents=True, exist_ok=True)
        shutil.copy2(w, cache / b["cnn"]["file"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(b, a.out)
    log(f"shipped {src.name} -> {a.out}: mode {mode} (gates 1 {g1}, 2 {g2}, 3 {g3}, 4 {g4ok})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--root", required=True)
    d.add_argument("--datasets", nargs="*", default=["original", *METHODS])
    d.add_argument("--dfd-real", type=int, default=150)
    d.add_argument("--dfd-fake", type=int, default=300)
    d.add_argument("--threads", type=int, default=8)
    m = sub.add_parser("manifest")
    m.add_argument("--root", required=True)
    m.add_argument("--out", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--manifest", required=True)
    e.add_argument("--features-dir", required=True)
    e.add_argument("--backbones", nargs="+", required=True)
    e.add_argument("--config", default=None)
    e.add_argument("--workers", type=int, default=2)
    e.add_argument("--device", default="auto")
    e.add_argument("--transform", choices=["whatsapp"], default=None)
    e.add_argument("--limit", type=int, default=0, help="at most N videos per split and family")
    e.add_argument("--splits", nargs="*", default=None, help="only these splits (e.g. test external)")
    e.add_argument("--checkpoint-tar", default=None, help="tar the features here every N videos (Drive)")
    e.add_argument("--checkpoint-every", type=int, default=500)
    e.add_argument("--save-crops", default=None, help="also save stage-2 crops (JPEG) under this dir")
    e.add_argument("--cnn-crop-size", type=int, default=380)
    s = sub.add_parser("select")
    s.add_argument("--manifest", required=True)
    s.add_argument("--features-dir", required=True)
    s.add_argument("--out-dir", required=True)
    s.add_argument("--backbones", nargs="*", default=None)
    s.add_argument("--config", default=None)
    f = sub.add_parser("finetune")
    f.add_argument("--manifest", required=True)
    f.add_argument("--crops-dir", required=True)
    f.add_argument("--out-dir", required=True)
    f.add_argument("--arch", default="tf_efficientnet_b4.ns_jft_in1k")
    f.add_argument("--size", type=int, default=380)
    f.add_argument("--epochs", type=int, default=6)
    f.add_argument("--batch", type=int, default=24)
    f.add_argument("--lr", type=float, default=3e-4)
    f.add_argument("--workers", type=int, default=2)
    f.add_argument("--weights-name", default="faceswap_cnn_b4.safetensors")
    f.add_argument("--config", default=None)
    v = sub.add_parser("evaluate")
    v.add_argument("--manifest", required=True)
    v.add_argument("--features-dir", default=None)
    v.add_argument("--bundle", required=True)
    v.add_argument("--report", required=True)
    v.add_argument("--title", default="Robustness: WhatsApp-like re-encode (640 px wide, CRF 30)")
    v.add_argument("--crops-dir", default=None, help="stage-2 bundles: the variant's saved crops")
    v.add_argument("--device", default="auto")
    sh = sub.add_parser("ship", help="apply docs/12 §6 and write the bundle the app loads")
    sh.add_argument("--bundle", required=True, help="faceswap.joblib or faceswap_cnn.joblib from Colab")
    sh.add_argument("--gate4", required=True, help="the JSON written by scripts/check_faceswap_reals.py")
    sh.add_argument("--out", default=str(REPO / "src/trubeat/models/faceswap.joblib"))
    sh.add_argument("--url", default="", help="stage 2: the release URL the weights download from")
    a = ap.parse_args()
    {"download": cmd_download, "manifest": cmd_manifest, "extract": cmd_extract,
     "select": cmd_select, "finetune": cmd_finetune, "evaluate": cmd_evaluate,
     "ship": cmd_ship}[a.cmd](a)


if __name__ == "__main__":
    main()
