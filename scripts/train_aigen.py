"""Train + honestly evaluate the whole-frame AI-video detector (docs/11 D4, D5).

Data roles
  pool      AIGVDBench fakes + reals (content-paired) and FF++ c23 original
            (real) train/val clips. Model selection, leave-one-generator-out
            (LOGO) evaluation and the final fit use the pool only.
  external  GenBuster-Bench (different curators; id / ood / wild tiers incl.
            Sora 2, Veo 3.1, Kling 2.5) and FF++ c23 original *test* clips
            (real talking heads: the false-positive check closest to the
            user's videos). Never used for training, selection or the
            threshold.

Protocol
  1. Every clip is re-encoded to one common format (configs/p5_aigen.yaml
     `normalize`) so bitrate/resolution cannot act as a label shortcut.
  2. Features: aigen.video_descriptor (trajectory geometry + mean
     embedding), cached by content hash.
  3. Selection: feature set x C, scored by macro LOGO AUC on the pool
     (each generator held out in turn, tested against a held-out real fold;
     fakes content-paired with test reals are dropped from training).
  4. Operating threshold: out-of-fold real scores on the pool, quantile
     1 - target_fpr. Never tuned on external data.
  5. External evaluation once: AUC per tier/generator vs held-out reals,
     recall at the threshold, FPR on GenBuster reals and on FF++ real test
     clips, bootstrap CIs; plus a harsher re-encode, a tab-capture-like
     VP9 WebM and the unre-encoded file, for robustness.

    python scripts/train_aigen.py --robustness-only   # robustness rows only,
                                                      # from saved features

Usage:
    python scripts/train_aigen.py --config configs/p5_aigen.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import subprocess
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from trubeat import aigen
from trubeat.errors import TrueBeatError

REPO = Path(__file__).resolve().parents[1]


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def cfg_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:8]


# ------------------------------------------------------------------ data ---


def load_rows(aigen_manifest: Path, ffpp_manifest: Path,
              extra_manifests: tuple[Path, ...] = ()) -> pd.DataFrame:
    a = pd.concat([pd.read_csv(m) for m in (aigen_manifest, *extra_manifests)],
                  ignore_index=True)
    a["role"] = np.where(a["source"].str.startswith("genbuster"), "external", "pool")
    rows = [a]
    if ffpp_manifest.is_file():
        f = pd.read_csv(ffpp_manifest)
        f = f[(f["label"] == "real") & f["usable"].astype(bool)].copy()
        rows.append(pd.DataFrame({
            "path": f["path"],
            "label": "real",
            "generator": "real",
            "source": "ffpp_original",
            "family": "real",
            "split_group": "ffpp:" + f["split_group"].astype(str),
            "role": np.where(f["split"] == "test", "external", "pool"),
        }))
    df = pd.concat(rows, ignore_index=True)
    df["path"] = [str(p if Path(p).is_absolute() else REPO / p) for p in df["path"]]
    df = df[[Path(p).is_file() for p in df["path"]]].reset_index(drop=True)
    return df


ROLE_SPLIT = {"pool": "train", "external": "test"}


def check_roles(rows: pd.DataFrame) -> None:
    """The identity-leakage validator, before any fit or score (AGENTS.md §6,
    docs/13 B-03). The pool trains, selects and sets the threshold; the
    external set is scored once. No ``split_group`` (a source clip with every
    clip generated from it, a prompt, a FaceForensics++ source sequence) and
    no file may sit on both sides. Raises ValueError."""
    from trubeat import eval as ev

    bad = set(rows["role"]) - set(ROLE_SPLIT)
    if bad:
        raise ValueError(f"unknown role(s) {sorted(map(str, bad))}; expected one of {sorted(ROLE_SPLIT)}")
    ev.validate_manifest(rows.assign(split=rows["role"].map(ROLE_SPLIT)))
    both = set(rows.loc[rows.role == "pool", "path"]) & set(rows.loc[rows.role == "external", "path"])
    if both:
        raise ValueError(f"source leakage: {len(both)} file(s) in both the pool and the external set: "
                         f"{sorted(both)[:5]}")


# ------------------------------------------------------------- normalise ---


def normalized_path(src: str, norm: dict) -> Path:
    if norm.get("vcodec"):  # e.g. the tab-capture-like VP9 WebM variant
        sig = f"{src}|{norm['short_side']}|{norm['vcodec']}|{norm['bitrate']}"
        ext = ".webm"
    else:  # libx264 (key kept identical to earlier runs, so caches stay valid)
        sig = f"{src}|{norm['short_side']}|{norm['crf']}|{norm['preset']}"
        ext = ".mp4"
    return REPO / norm["out_dir"] / f"{hashlib.sha1(sig.encode()).hexdigest()[:16]}{ext}"


def normalize_one(args: tuple[str, dict]) -> tuple[str, str | None]:
    src, norm = args
    out = normalized_path(src, norm)
    if out.is_file() and out.stat().st_size > 0:
        return src, str(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    s = int(norm["short_side"])
    vf = f"scale='if(gt(iw,ih),-2,{s})':'if(gt(iw,ih),{s},-2)'"
    tmp = out.with_suffix(".part" + out.suffix)
    if norm.get("vcodec"):
        codec = ["-c:v", str(norm["vcodec"]), "-b:v", str(norm["bitrate"]),
                 "-deadline", "realtime", "-cpu-used", "8"]
    else:
        codec = ["-c:v", "libx264", "-crf", str(norm["crf"]), "-preset", str(norm["preset"])]
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", src, "-t", "20",
        "-vf", vf, *codec, "-pix_fmt", "yuv420p", "-an", "-threads", "2", str(tmp),
    ]
    try:
        subprocess.run(cmd, check=True, timeout=300, capture_output=True)
        tmp.replace(out)
        return src, str(out)
    except (subprocess.SubprocessError, OSError):
        tmp.unlink(missing_ok=True)
        return src, None


def normalize_all(paths: list[str], norm: dict, workers: int) -> dict[str, str]:
    out: dict[str, str] = {}
    with mp.get_context("spawn").Pool(workers) as pool:
        for i, (src, dst) in enumerate(
            pool.imap_unordered(normalize_one, [(p, norm) for p in paths], chunksize=8)
        ):
            if dst:
                out[src] = dst
            if (i + 1) % 250 == 0:
                log(f"  normalised {i + 1}/{len(paths)}")
    return out


# ------------------------------------------------------------- features ---

_EMB: aigen.Embedder | None = None
_CFG: dict = {}


def _init_worker(cfg: dict) -> None:
    global _EMB, _CFG
    warnings.filterwarnings("ignore")
    _CFG = cfg
    _EMB = aigen.Embedder(cfg)


def featurize_one(path: str) -> tuple[str, dict | None, np.ndarray | None, str]:
    try:
        seg, meta = aigen.segment_embeddings(path, _EMB, _CFG, use_cache=True)
        geom, mean_emb = aigen.video_descriptor(seg)
        geom["n_segments"] = float(meta.get("n_segments", 0))
        geom["duration_s"] = float(meta.get("duration_s", 0.0))
        return path, geom, mean_emb, ""
    except TrueBeatError as exc:
        return path, None, None, exc.code
    except Exception as exc:  # noqa: BLE001 - one bad clip must not stop the run
        return path, None, None, f"E_UNKNOWN: {exc}"


def featurize_all(paths: list[str], cfg: dict, workers: int) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame({"feat_path": pd.Series(dtype=str), "error": pd.Series(dtype=str)})
    wcfg = dict(cfg)
    wcfg["torch_threads"] = max(1, int(cfg.get("train_threads", cfg["torch_threads"])) // workers)
    wcfg["max_segments"] = int(cfg.get("train_segments", cfg["max_segments"]))
    recs, failed = [], 0
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(wcfg,)) as pool:
        for i, (p, geom, emb, err) in enumerate(pool.imap_unordered(featurize_one, paths, chunksize=2)):
            if geom is None:
                failed += 1
                recs.append({"feat_path": p, "error": err})
            else:
                r = {"feat_path": p, "error": ""}
                r.update(geom)
                r.update({f"emb_{j}": float(v) for j, v in enumerate(emb)})
                recs.append(r)
            if (i + 1) % 100 == 0:
                rate = (i + 1) / (time.time() - t0)
                log(f"  features {i + 1}/{len(paths)} ({rate:.2f} clips/s, {failed} failed)")
    return pd.DataFrame(recs)


# ------------------------------------------------------------ modelling ---


def columns_for(feature_set: str, df: pd.DataFrame) -> list[str]:
    emb_cols = sorted((c for c in df.columns if c.startswith("emb_")), key=lambda c: int(c[4:]))
    if feature_set == "geometry":
        return list(aigen.GEOMETRY_COLUMNS)
    if feature_set == "embedding":
        return emb_cols
    if feature_set == "combined":
        return list(aigen.GEOMETRY_COLUMNS) + emb_cols
    raise ValueError(feature_set)


def make_model(C: float) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=5000, class_weight="balanced")),
    ])


def ai_score(model: Pipeline, X: np.ndarray) -> np.ndarray:
    s = model.decision_function(X)
    return s if list(model.classes_)[1] == "ai" else -s


def safe_auc(y_ai: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y_ai, s)) if 0 < y_ai.sum() < len(y_ai) else float("nan")


def base_family(generator: str) -> str:
    """Base model family of a generator label, so variants of one model
    (t2v / i2v / v2v, version bumps, the same model in two dataset
    sections) are held out together: holding out ltx_t2v while ltx_i2v
    stays in training would overstate generalisation to unseen generators."""
    g = generator.lower()
    for suffix in ("_t2v", "_i2v", "_v2v"):
        g = g.removesuffix(suffix)
    aliases = {"open-sora": "opensora", "wan2.1": "wan", "wanx": "wan",
               "cogvideox1.5": "cogvideox", "ltxvideo": "ltx", "gen2": "runway",
               "gen3": "runway", "gen4.5": "runway",
               # built on another base model: held out with it
               "causvid": "wan", "accvideo": "hunyuanvideo", "repvideo": "cogvideox",
               # Rapidata names (scripts/download_rapidata.py)
               "gen3-alpha": "runway", "ray2": "luma", "pika2.2": "pika", "veo2": "veo",
               "veo3": "veo", "veo3.1": "veo", "hunyuan": "hunyuanvideo", "mochi-1": "mochi",
               "seedance-1-pro": "seedance", "kling-v2.1-master": "kling",
               # GenBuster wild names (external only; used to label which
               # external generators have an earlier version in training)
               "hailuo2.3": "hailuo", "kling2.5_turbo": "kling", "kling2.6": "kling",
               "pixverse_v5": "pixverse", "seedance1.5_pro": "seedance",
               "seedance2.0": "seedance", "sora2": "sora", "wan2.5": "wan", "wan2.6": "wan"}
    return aliases.get(g, g)


def logo_eval(pool: pd.DataFrame, cols: list[str], C: float, n_folds: int = 5) -> pd.DataFrame:
    """Leave-one-generator-family-out AUC on the pool (see base_family)."""
    reals = pool[pool.label == "real"]
    fold_of = {}
    for k, (_, te) in enumerate(GroupKFold(n_folds).split(reals, groups=reals.split_group)):
        for g in reals.iloc[te].split_group:
            fold_of[g] = k
    pool = pool.assign(real_fold=pool.split_group.map(fold_of),
                       family_id=pool.generator.map(base_family))
    gens = sorted(pool[pool.label == "ai"].family_id.unique())
    out = []
    for i, gen in enumerate(gens):
        k = i % n_folds
        test_real = pool[(pool.label == "real") & (pool.real_fold == k)]
        test_fake = pool[(pool.label == "ai") & (pool.family_id == gen)]
        blocked = set(test_real.split_group) | set(test_fake.split_group)
        train = pool[
            ((pool.label == "real") & (pool.real_fold != k))
            | ((pool.label == "ai") & (pool.family_id != gen))
        ]
        train = train[~train.split_group.isin(blocked)]
        test = pd.concat([test_real, test_fake])
        m = make_model(C).fit(train[cols].to_numpy(), train.label.to_numpy())
        s = ai_score(m, test[cols].to_numpy())
        y = (test.label == "ai").to_numpy().astype(int)
        out.append({"generator": gen, "n_fake": len(test_fake), "n_real": len(test_real),
                    "auc": safe_auc(y, s)})
    return pd.DataFrame(out)


def oof_scores(pool: pd.DataFrame, cols: list[str], C: float, n_folds: int = 5) -> np.ndarray:
    s = np.full(len(pool), np.nan)
    for tr, te in GroupKFold(n_folds).split(pool, groups=pool.split_group):
        m = make_model(C).fit(pool.iloc[tr][cols].to_numpy(), pool.iloc[tr].label.to_numpy())
        s[te] = ai_score(m, pool.iloc[te][cols].to_numpy())
    return s


def bootstrap_auc(y: np.ndarray, s: np.ndarray, n: int = 1000, seed: int = 7) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    idx_pos, idx_neg = np.where(y == 1)[0], np.where(y == 0)[0]
    if len(idx_pos) == 0 or len(idx_neg) == 0:
        return float("nan"), float("nan")
    for _ in range(n):
        b = np.concatenate([rng.choice(idx_pos, len(idx_pos)), rng.choice(idx_neg, len(idx_neg))])
        vals.append(roc_auc_score(y[b], s[b]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, float((c - h) / d)), min(1.0, float((c + h) / d))  # no -0.0% from rounding


# ------------------------------------------------------------------ main ---


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(REPO / "configs/p5_aigen.yaml"))
    ap.add_argument("--manifest", default=str(REPO / "data/manifests/aigen.csv"))
    ap.add_argument("--ffpp-manifest", default=str(REPO / "data/manifests/ffpp_c23_qc.csv"))
    ap.add_argument("--model-out", default=str(REPO / "runs/aigen_detector.joblib"))
    ap.add_argument("--report", default=str(REPO / "reports/aigen_detector.md"))
    ap.add_argument("--skip-robustness", action="store_true")
    ap.add_argument("--max-pool-real", type=int, default=1500,
                    help="cap on pool real clips (paired reals kept first)")
    ap.add_argument("--external-per-group", type=int, default=150,
                    help="cap per (source, generator, label) in the external set")
    ap.add_argument("--variant-limit", type=int, default=40,
                    help="per-(source, generator) cap on AI clips in the robustness subset")
    ap.add_argument("--variant-real-limit", type=int, default=200,
                    help="per-source cap on real clips in the robustness subset")
    ap.add_argument("--robustness-only", action="store_true",
                    help="skip training: re-run the robustness rows from --features and "
                         "--model-out into --robustness-report")
    ap.add_argument("--features", default=str(REPO / "runs/aigen_features.pkl"),
                    help="features table: written by training, read by --robustness-only")
    ap.add_argument("--extra-manifest", nargs="*", default=[],
                    help="more pool manifests (e.g. data/manifests/aigen_rapidata.csv)")
    ap.add_argument("--robustness-report", default=str(REPO / "reports/aigen_robustness.md"))
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    cfg = aigen.merged_config(yaml.safe_load(Path(args.config).read_text()))
    cfg["cache_dir"] = str(REPO / cfg["cache_dir"])
    workers = int(cfg.get("workers", 2))
    if args.robustness_only:
        robustness_only(args, cfg, workers)
        return

    rows = load_rows(Path(args.manifest), Path(args.ffpp_manifest),
                     tuple(Path(m) for m in args.extra_manifest))
    check_roles(rows)  # every later set is a subset of these rows
    pool_rows = rows[rows.role == "pool"]
    pool_ai = pool_rows[pool_rows.label == "ai"]
    pool_real = pool_rows[pool_rows.label == "real"]
    # FF++ talking-head reals are the domain closest to the user's videos:
    # always kept. The cap applies to the AIGVDBench reals, content-paired
    # ones first (an earlier version capped all reals together and dropped
    # every FF++ clip, because 1,501 paired reals came first).
    ff_real = pool_real[pool_real.source == "ffpp_original"]
    av_real = pool_real[pool_real.source != "ffpp_original"]
    paired = av_real.split_group.isin(set(pool_ai.split_group))
    av_real = pd.concat([av_real[paired], av_real[~paired].sample(frac=1, random_state=7)])
    pool_real = pd.concat([ff_real, av_real.head(int(args.max_pool_real))])
    ext_rows = cap_rows(rows[rows.role == "external"], int(args.external_per_group))
    rows = pd.concat([pool_ai, pool_real, ext_rows], ignore_index=True)
    log(f"rows: {len(rows)}\n{rows.groupby(['role', 'source', 'label']).size().to_string()}")

    log("normalising every clip to one format")
    norm = normalize_all(rows.path.tolist(), cfg["normalize"], workers=4)
    rows["feat_path"] = rows.path.map(norm)
    log(f"normalised {rows.feat_path.notna().sum()}/{len(rows)}")
    rows = rows[rows.feat_path.notna()].reset_index(drop=True)

    log("extracting features")
    feats = featurize_all(rows.feat_path.tolist(), cfg, workers)
    errs = feats[feats.error != ""]
    if len(errs):
        log(f"feature failures: {errs.error.value_counts().to_dict()}")
    data = rows.merge(feats[feats.error == ""], on="feat_path")
    data.to_pickle(args.features)
    pool = data[data.role == "pool"].reset_index(drop=True)
    ext = data[data.role == "external"].reset_index(drop=True)
    log(f"pool {len(pool)} ({(pool.label == 'ai').sum()} ai), external {len(ext)}")

    # ---- selection by macro LOGO AUC on the pool ----
    search = []
    for fs in cfg["feature_set_candidates"]:
        cols = columns_for(fs, pool)
        for C in cfg["C_candidates"]:
            lg = logo_eval(pool, cols, float(C))
            search.append({"feature_set": fs, "C": float(C), "logo_macro_auc": lg.auc.mean(),
                           "logo_min_auc": lg.auc.min(), "logo": lg})
            log(f"  LOGO {fs:9s} C={C:<5} macro AUC {lg.auc.mean():.4f} (min {lg.auc.min():.3f})")
    best = max(search, key=lambda r: r["logo_macro_auc"])
    fs, C = best["feature_set"], best["C"]
    cols = columns_for(fs, pool)
    log(f"selected {fs} C={C}")

    # ---- threshold from out-of-fold pool real scores ----
    oof = oof_scores(pool, cols, C)
    y_pool = (pool.label == "ai").to_numpy().astype(int)
    thr = float(np.quantile(oof[y_pool == 0], 1.0 - float(cfg["target_fpr"])))
    pool_auc = safe_auc(y_pool, oof)

    final = make_model(C).fit(pool[cols].to_numpy(), pool.label.to_numpy())
    bundle_cfg = {k: v for k, v in cfg.items() if k not in ("cache_dir",)}
    # The shipped path must be the evaluated path: inference scores exactly
    # the segments training and evaluation used (docs/11 review #4).
    bundle_cfg["max_segments"] = int(cfg.get("train_segments", 1))
    bundle = {
        "model": final, "feature_set": fs, "feature_columns": cols, "threshold": thr,
        # Out-of-fold scores of the pool's real clips: the app reports where a
        # clip falls among them ("scores higher than X% of real clips"), which
        # is consistent with the threshold by construction (the threshold is
        # their 1 - target_fpr quantile). Replaces an uncalibrated sigmoid.
        "real_reference_scores": np.sort(oof[y_pool == 0]).astype(np.float32),
        "config": bundle_cfg,
        "config_hash": cfg_hash(cfg),
        "trained_on": " + ".join(
            f"{src} ({n} clips)" for src, n in pool.source.value_counts().sort_index().items()),
        "selection": "macro leave-one-generator-out AUC on the pool",
        "created": time.strftime("%Y-%m-%d %H:%M"),
        "sklearn_version": sklearn.__version__,  # checked on load (trubeat.bundles, docs/13 D-05)
    }
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model_out)

    # ---- external evaluation (once) ----
    ext_s = ai_score(final, ext[cols].to_numpy())
    ext = ext.assign(score=ext_s, flagged=ext_s >= thr)
    lines = external_tables(ext)

    variants = [] if args.skip_robustness else run_variants(
        ext, variant_subset(ext, args.variant_limit, args.variant_real_limit),
        cfg, bundle, workers)

    write_report(args.report, bundle, best, search, pool, pool_auc, thr, lines, variants, ext)
    log(f"model -> {args.model_out}\nreport -> {args.report}")


# (row name, config key of the first transform or None, then the app's own
# normalisation?). The app always re-encodes its input to the training format
# (aigen.normalize_for_inference), so a variant measures the app path only
# when that second stage is applied; "as-is" is the app without ffmpeg.
# The two resolution rows are counterfactuals for the round-2 audit's
# metadata finding (pool AI clips ~480p, pool reals ~1080p): if the detector
# had learnt "soft = AI", shrinking real clips would raise false alarms and
# enlarging AI clips would hide them.
VARIANTS = (("harsh re-encode (240p, CRF 35) + app normalisation", "robustness_normalize", True),
            ("tab-capture-like (VP9 WebM, 720p, 2.5 Mb/s) + app normalisation",
             "tabcapture_normalize", True),
            ("real-world downscale to 480p + app normalisation", "downscale_480", True),
            ("upscale to 1080p + app normalisation", "upscale_1080", True),
            ("as-is (no re-encode: the app without ffmpeg)", None, False))


def variant_subset(ext: pd.DataFrame, limit: int, real_limit: int) -> pd.DataFrame:
    """External clips for the robustness rows: at most `limit` AI clips per
    (source, generator) and `real_limit` real clips per source. Reals get
    their own, larger cap: the false-positive rate is what the extension's
    tab capture most needs measured, and 20 reals bound it only to <16%."""
    fakes = cap_rows(ext[ext.label == "ai"], int(limit), seed=7)
    reals = cap_rows(ext[ext.label == "real"], int(real_limit), seed=7)
    return pd.concat([fakes, reals], ignore_index=True)


def run_variants(ext: pd.DataFrame, sub: pd.DataFrame, cfg: dict, bundle: dict,
                 workers: int) -> list[tuple[str, dict]]:
    """Score `sub` under each input variant with the bundle's fixed threshold.
    `ext` must carry `score`/`flagged` under the training normalisation; the
    first row is that same subset, for a like-for-like comparison."""
    final, cols, thr = bundle["model"], bundle["feature_columns"], float(bundle["threshold"])
    out = [("training normalisation (360p, CRF 28)",
            compact_eval(ext[ext.path.isin(set(sub.path))]))]
    keep = ["path", "label", "generator", "source", "family", "split_group", "role", "feat_path"]
    for name, key, then_app in VARIANTS:
        log(f"variant: {name} on {len(sub)} external clips")
        if key is None:
            vx = sub.assign(feat_path=sub.path)
        else:
            m = normalize_all(sub.path.tolist(), cfg[key], workers=4)
            if then_app:
                m2 = normalize_all(list(m.values()), cfg["normalize"], workers=4)
                m = {k: m2.get(v) for k, v in m.items()}
            vx = sub.assign(feat_path=sub.path.map(m)).dropna(subset=["feat_path"])
        vf = featurize_all(vx.feat_path.tolist(), cfg, workers)
        vx = vx[keep].merge(vf[vf.error == ""], on="feat_path")
        vs = ai_score(final, vx[cols].to_numpy())
        out.append((name, compact_eval(vx.assign(score=vs, flagged=vs >= thr))))
    return out


def robustness_only(args, cfg: dict, workers: int) -> None:
    """Re-run only the robustness rows from saved features and the saved
    bundle (no retraining), into their own report."""
    data = pd.read_pickle(args.features)
    check_roles(data)
    bundle = joblib.load(args.model_out)
    cols, thr = bundle["feature_columns"], float(bundle["threshold"])
    ext = data[data.role == "external"].reset_index(drop=True)
    s = ai_score(bundle["model"], ext[cols].to_numpy())
    ext = ext.assign(score=s, flagged=s >= thr)
    sub = variant_subset(ext, args.variant_limit, args.variant_real_limit)
    robust = run_variants(ext, sub, cfg, bundle, workers)
    sizes = sub.groupby(["source", "label"]).size()
    L = [f"# AI-video detector: input robustness — {time.strftime('%Y-%m-%d %H:%M')}", "",
         (f"Bundle `{args.model_out}` (created {bundle.get('created')}, config "
          f"`{bundle.get('config_hash')}`), threshold {thr:+.4f} fixed. External clips only "
          "(never used for training, selection or the threshold)."), "",
         "Subset: " + ", ".join(f"{s} {lab} {n}" for (s, lab), n in sizes.items()) + ".", "",
         *robustness_table(robust),
         "",
         ("AUC per tier is against the GenBuster reals in the subset; recall and FPR are at "
          "the fixed threshold. `n_*` columns are clip counts.")]
    Path(args.robustness_report).write_text("\n".join(L) + "\n")
    log(f"robustness report -> {args.robustness_report}")


def robustness_table(robust: list[tuple[str, dict]]) -> list[str]:
    keys = sorted({k for _, d in robust for k in d})

    def fmt(v):
        return str(v) if isinstance(v, int) else f"{v:.3f}"

    # one column per input variant, one row per measure (reads better than
    # 17 measure columns)
    return (["| measure | " + " | ".join(name for name, _ in robust) + " |",
             "|---" * (len(robust) + 1) + "|"]
            + [f"| {k} | " + " | ".join(fmt(d.get(k, float("nan"))) for _, d in robust) + " |"
               for k in keys])


def cap_rows(df: pd.DataFrame, per_group: int, seed: int = 7,
             by: tuple[str, ...] = ("source", "generator", "label")) -> pd.DataFrame:
    """At most `per_group` rows per (source, generator, label), seeded."""
    if per_group <= 0:
        return df
    # shuffle, then the first `per_group` rows of each group. groupby().apply
    # drops the grouping columns under pandas 3 (it returned only `path`);
    # head() keeps every column on every pandas version.
    shuffled = df.sample(frac=1.0, random_state=seed)
    return shuffled.groupby(list(by), sort=False, dropna=False).head(per_group).reset_index(drop=True)


def compact_eval(df: pd.DataFrame) -> dict:
    """AUC per GenBuster AI tier vs GenBuster reals, FPRs, recall."""
    gb_real = df[(df.label == "real") & df.source.str.startswith("genbuster")]
    ff_real = df[df.source == "ffpp_original"]
    out = {"fpr_genbuster_real": float(gb_real.flagged.mean()) if len(gb_real) else float("nan"),
           "fpr_ffpp_real": float(ff_real.flagged.mean()) if len(ff_real) else float("nan"),
           "n_genbuster_real": len(gb_real), "n_ffpp_real": len(ff_real),
           "median_score_real": float(df[df.label == "real"].score.median()),
           "median_score_ai": float(df[df.label == "ai"].score.median())}
    for tier in ("genbuster_id", "genbuster_ood", "genbuster_wild"):
        fakes = df[(df.source == tier) & (df.label == "ai")]
        if fakes.empty or gb_real.empty:
            continue
        both = pd.concat([gb_real, fakes])
        y = (both.label == "ai").to_numpy().astype(int)
        out[f"auc_{tier.split('_')[1]}"] = safe_auc(y, both.score.to_numpy())
        out[f"recall_{tier.split('_')[1]}"] = float(fakes.flagged.mean())
        out[f"n_ai_{tier.split('_')[1]}"] = len(fakes)
    return out


def external_tables(ext: pd.DataFrame) -> list[str]:
    out = []
    gb_real = ext[(ext.label == "real") & ext.source.str.startswith("genbuster")]
    ff_real = ext[ext.source == "ffpp_original"]
    for name, reals in (("GenBuster real", gb_real), ("FF++ c23 real (talking heads)", ff_real)):
        k, n = int(reals.flagged.sum()), len(reals)
        lo, hi = wilson(k, n)
        out.append(f"| false-positive rate, {name} | {k}/{n} = {k / max(n, 1):.3f} | [{lo:.3f}, {hi:.3f}] |")
    out.append("")
    out.append("| GenBuster tier | n AI | AUC vs GenBuster reals | 95% CI | recall at threshold | 95% CI |")
    out.append("|---|---|---|---|---|---|")
    for tier in ("genbuster_id", "genbuster_ood", "genbuster_wild"):
        fakes = ext[(ext.source == tier) & (ext.label == "ai")]
        if fakes.empty:
            continue
        both = pd.concat([gb_real, fakes])
        y = (both.label == "ai").to_numpy().astype(int)
        auc = safe_auc(y, both.score.to_numpy())
        lo, hi = bootstrap_auc(y, both.score.to_numpy())
        k = int(fakes.flagged.sum())
        rl, rh = wilson(k, len(fakes))
        out.append(f"| {tier.split('_')[1]} | {len(fakes)} | {auc:.3f} | [{lo:.3f}, {hi:.3f}] | "
                   f"{k / len(fakes):.3f} | [{rl:.3f}, {rh:.3f}] |")
    out.append("")
    out.append("| generator (external) | tier | n | AUC vs GenBuster reals | recall at threshold |")
    out.append("|---|---|---|---|---|")
    for (tier, gen), fakes in ext[ext.label == "ai"].groupby(["source", "generator"]):
        both = pd.concat([gb_real, fakes])
        y = (both.label == "ai").to_numpy().astype(int)
        out.append(f"| {gen} | {tier.split('_')[1]} | {len(fakes)} | "
                   f"{safe_auc(y, both.score.to_numpy()):.3f} | {fakes.flagged.mean():.3f} |")
    return out


def write_report(path, bundle, best, search, pool, pool_auc, thr, ext_lines, robust, ext) -> None:
    lg = best["logo"].sort_values("auc")
    L = [
        f"# Whole-frame AI-video detector — {bundle['created']}",
        "",
        (
            "Protocol: docs/11 D4/D5 and `scripts/train_aigen.py` docstring. Every clip "
            f"re-encoded to one format before features ({bundle['config']['normalize']}). "
            f"Threshold from out-of-fold pool reals at FPR {bundle['config']['target_fpr']}; "
            "external data never used for selection or the threshold. Config hash "
            f"`{bundle['config_hash']}`."
        ),
        "",
        "## Data",
        "",
        (
            f"- pool: {len(pool)} clips ({int((pool.label == 'ai').sum())} AI from "
            f"{pool[pool.label == 'ai'].generator.nunique()} generators; "
            f"{int((pool.label == 'real').sum())} real)"
        ),
        f"- external: {len(ext)} clips (GenBuster-Bench all tiers + FF++ c23 original test split)",
        "",
        "## Model selection (macro leave-one-generator-family-out AUC on the pool)",
        "",
        "| feature set | C | macro LOGO AUC | worst generator AUC |",
        "|---|---|---|---|",
    ]
    for r in search:
        L.append(f"| {r['feature_set']} | {r['C']} | {r['logo_macro_auc']:.4f} | {r['logo_min_auc']:.3f} |")
    L += [
        "",
        (
            f"Selected: **{bundle['feature_set']}, C={best['C']}**. Pooled out-of-fold AUC "
            f"(grouped 5-fold, in-distribution): {pool_auc:.4f}. Threshold: {thr:+.4f}."
        ),
        "",
        "### Leave-one-generator-family-out, selected model (each family unseen in training)",
        "",
        "| generator family | n AI | n real | AUC |",
        "|---|---|---|---|",
    ]
    L += [f"| {r.generator} | {r.n_fake} | {r.n_real} | {r.auc:.3f} |" for r in lg.itertuples()]
    L += ["", "## External evaluation (never seen in training, selection or threshold)", "",
          "| metric | value | 95% CI (Wilson) |", "|---|---|---|"]
    L += ext_lines
    if len(robust) > 1:
        L += ["", "## Robustness and input handling (same external subset, threshold fixed)", "",
              *robustness_table(robust)]
    L += ["", "## Limits", "",
          ("- Real clips come from OpenVid-HD (AIGVDBench, GenBuster) and FF++ YouTube; other "
           "real content (phone video, screen recordings, animation) is not covered."),
          ("- AUC and recall are per-clip on curated sets; the user's own labelled set "
           "(docs/11 F-1) is the test that matters."),
          "- Generators released after these datasets are untested by construction."]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
