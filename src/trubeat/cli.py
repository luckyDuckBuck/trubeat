"""CLI: trubeat hr|train|predict|eval. (SRS §3)"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import detector, face, features, ingest, roi, signal
from . import eval as evalmod
from . import hr as hrmod
from . import model as modelmod
from .errors import TrueBeatError


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if isinstance(cfg.get("band"), list):
        cfg["band"] = tuple(cfg["band"])
    return cfg


def _jsonable(d):
    """NaN → None so output is always valid JSON."""
    return {
        k: (None if isinstance(v, float) and math.isnan(v) else v)
        for k, v in d.items()
    }


def _feat_for_video(path, config):
    """Shared wiring: video → windowed feature DataFrame."""
    method = config.get("method", "pos")
    stream = ingest.open_video(path)
    cut = ingest.truncation_warning(stream)
    if cut:   # training on part of a clip is allowed, never silent (AGENTS.md §5)
        print(f"{Path(path).name}: {cut}", file=sys.stderr)
    landmarks, valid = face.track_landmarks(stream.frames, stream.fps)
    traces = roi.extract_roi_means(stream.frames, landmarks, valid, roi.ROI_DEFS_P1)
    ppg = signal.extract_ppg(traces, stream.fps, method=method)
    return features.window_features(
        ppg, stream.fps,
        win_s=config.get("window_s", 10.0),
        overlap=config.get("overlap", 0.5),
    )


def cmd_hr(args):
    config = load_config(args.config) if args.config else {}
    band = tuple(config.get("band", (0.7, 3.0)))
    method = config.get("method", "pos")
    stream = ingest.open_video(args.video)
    landmarks, valid = face.track_landmarks(stream.frames, stream.fps)
    traces = roi.extract_roi_means(stream.frames, landmarks, valid, roi.ROI_DEFS_P1)
    ppg = signal.extract_ppg(traces, stream.fps, method=method)
    res = hrmod.estimate_hr(np.nanmean(ppg, axis=1), stream.fps, band=band)
    cut = ingest.truncation_warning(stream)
    print(json.dumps(_jsonable({
        "bpm": round(float(res.bpm), 1),
        "confidence": round(float(res.confidence), 3),
        "fps": float(stream.fps),
        "duration_s": round(float(stream.duration_s), 1),
        "face_valid_frac": round(float(np.mean(valid)), 3),
        "warnings": [cut] if cut else [],
    })))


def cmd_train(args):
    import joblib
    import sklearn

    config = load_config(args.config)
    manifest = pd.read_csv(args.manifest)
    evalmod.validate_manifest(manifest)
    train_rows = manifest[manifest["split"] == "train"]
    if train_rows.empty:
        sys.exit("no rows with split=='train' in manifest")
    X_parts, y, groups = [], [], []
    for _, row in train_rows.iterrows():
        feat = _feat_for_video(row["path"], config)
        if len(feat) == 0:
            print(f"warning: no windows for {row['path']}", file=sys.stderr)
            continue
        X_parts.append(feat)
        y.extend([row["label"]] * len(feat))
        groups.extend([row["split_group"]] * len(feat))
    X = pd.concat(X_parts, ignore_index=True)
    # Train on the detector feature set only: the positional window columns
    # (win_start_s/win_end_s) say where a window sits, not what the pulse
    # does. predict_video hands the model exactly its fitted columns
    # (model.feature_names_in_), so train and predict stay in step, and the
    # bundle has the same shape as the G2 detector (docs/11 A6).
    cols = list(detector.DETECTOR_COLUMNS)
    model = modelmod.train_classifier(X[cols], np.array(y), np.array(groups))
    out = args.output or "runs/trubeat_model.joblib"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "config": config,
            "feature_columns": cols,
            "method": config.get("method", "pos"),
            "window_s": float(config.get("window_s", 10.0)),
            "sklearn_version": sklearn.__version__,  # checked on load (trubeat.bundles)
        },
        out,
    )
    print(json.dumps({"saved": out, "n_windows": len(X),
                      "n_videos": len(train_rows)}))


def _load_bundle(path):
    """(model, saved config, load warnings such as W_SKLEARN_VERSION)."""
    from .bundles import load_checked

    bundle, load_warnings = load_checked(path)
    for w in load_warnings:
        print(f"warning: {w}", file=sys.stderr)
    if isinstance(bundle, dict):
        return bundle["model"], bundle.get("config", {}), load_warnings
    return bundle, {}, load_warnings


def cmd_predict(args):
    model, saved_config, load_warnings = _load_bundle(args.model)
    config = load_config(args.config) if args.config else saved_config
    pred = modelmod.predict_video(args.video, model, config)
    pred.warnings[:0] = load_warnings
    print(json.dumps(_jsonable(pred.to_json())))


def cmd_eval(args):
    config = load_config(args.config)
    ds = args.dataset
    if Path(ds).exists():
        manifest_path = ds
    elif Path(ds).is_absolute():
        sys.exit(f"manifest not found: {ds}")
    else:
        manifest_path = f"data/manifests/{ds}.csv"
    manifest = pd.read_csv(manifest_path)
    if args.task == "hr":
        report = evalmod.evaluate_hr(manifest, config)
        summary = {"mae": report.mae, "rmse": report.rmse,
                   "pearson_r": report.pearson_r}
    else:
        if not args.model:
            sys.exit("eval --task detect requires --model")
        model, _, _ = _load_bundle(args.model)  # load warnings go to stderr
        report = evalmod.evaluate_detector(manifest, model, config)
        summary = {"accuracy": report.accuracy, "auc": report.auc,
                   "fpr_on_real": report.fpr_on_real}
    out = args.report or "reports/eval_report.md"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    evalmod.write_report(report, config, out)
    print(json.dumps(_jsonable({"report": out, **summary})))


def cmd_aigen(args):
    from . import aigen, provenance

    prov = provenance.read_provenance(args.video)
    bundle = aigen.find_bundle(args.model)
    out = {"provenance": json.loads(prov.to_json())}
    if bundle is None:
        out["aigen"] = {"label": "unknown", "warnings": ["W_NO_AIGEN_MODEL: no trained bundle found"]}
    else:
        res = aigen.analyze(args.video, bundle)
        out["aigen"] = json.loads(res.to_json())
    print(json.dumps(out))


def cmd_fetch_models(args):
    """Download the backbone weights the installed bundles need (the only
    network step): the AI-video detector's and, when a face-swap bundle is
    installed, its backbone too. ``--backbone`` fetches just that one."""
    from . import aigen, faceswap
    from .errors import ModelMissing

    wanted = [args.backbone] if args.backbone else []
    fsb = None if args.backbone else faceswap.load_bundle()
    if not wanted:
        ai = aigen.find_bundle()
        wanted.append(ai["config"]["backbone"] if ai else aigen.DEFAULT_CONFIG["backbone"])
        if fsb is not None and fsb.get("kind") != "cnn":
            wanted.append(fsb["config"]["backbone"])
    out, failed = {}, {}
    for b in dict.fromkeys(wanted):
        out[b] = aigen.fetch_weights(b)
    if fsb is not None and fsb.get("kind") == "cnn":
        try:
            out[fsb["cnn"]["file"]] = str(faceswap.fetch_cnn_weights(fsb))
        except ModelMissing as exc:
            failed[fsb["cnn"]["file"]] = str(exc)
    print(json.dumps({"fetched": out, **({"failed": failed} if failed else {})}))
    if failed:
        raise SystemExit(1)


def main() -> None:
    p = argparse.ArgumentParser(prog="trubeat")
    sub = p.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("hr", help="estimate heart rate from a video")
    h.add_argument("video")
    h.add_argument("--config", default=None)

    t = sub.add_parser("train", help="train the detector on a manifest")
    t.add_argument("--manifest", required=True)
    t.add_argument("--config", required=True)
    t.add_argument("--output", default=None)

    pr = sub.add_parser("predict", help="classify one video (offline)")
    pr.add_argument("video")
    pr.add_argument("--model", required=True)
    pr.add_argument("--config", default=None)

    e = sub.add_parser("eval", help="batch-score a dataset manifest")
    e.add_argument("--dataset", required=True,
                   help="manifest CSV path or data/manifests/<name>")
    e.add_argument("--config", required=True)
    e.add_argument("--task", default="detect", choices=["detect", "hr"])
    e.add_argument("--model", default=None)
    e.add_argument("--report", default=None)

    ag = sub.add_parser("aigen", help="check one video for AI generation (offline)")
    ag.add_argument("video")
    ag.add_argument("--model", default=None,
                    help="AI-video bundle (default: models/, data/models/, then the packaged copy)")

    fm = sub.add_parser("fetch-models",
                        help="download the detectors' backbone weights once (the only network step)")
    fm.add_argument("--backbone", default=None,
                    help="fetch only this backbone (default: every backbone the installed bundles use)")

    s = sub.add_parser("serve", help="run the TrueBeat web app (Gradio)")
    s.add_argument("--port", type=int, default=7860)
    s.add_argument("--host", default="127.0.0.1",
                   help="address to listen on (default: this machine only)")
    s.add_argument("--share", action="store_true",
                   help="temporary public Gradio link (uploads leave this machine)")
    s.add_argument("--config", default=None)
    s.add_argument("--model", default=None,
                   help="detector bundle (default: search models/, data/models/, "
                        "then the packaged copy)")

    args = p.parse_args()
    try:
        if args.cmd == "serve":
            from . import app as appmod

            argv = ["--port", str(args.port), "--host", args.host]
            if args.share:
                argv.append("--share")
            if args.config:
                argv += ["--config", args.config]
            if args.model:
                argv += ["--model", args.model]
            raise SystemExit(appmod.main(argv))
        {"hr": cmd_hr, "train": cmd_train, "predict": cmd_predict,
         "eval": cmd_eval, "aigen": cmd_aigen,
         "fetch-models": cmd_fetch_models}[args.cmd](args)
    except TrueBeatError as exc:
        print(json.dumps({"error": exc.code, "message": str(exc)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
