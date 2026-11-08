"""TrueBeat web app — one-click analysis over the existing pipeline.

This module has two clearly separated halves:

* **Analysis** (:func:`analyze_video`, :func:`verdict_for`) — pure Python,
  no UI dependency. It runs exactly the pipeline the CLI runs
  (ingest → face → ROI → signal → features → predict) and turns the result
  into a three-state verdict with its evidence. The browser extension calls
  :func:`analyze_video` over HTTP (Gradio's built-in API exposes it)
  without touching the UI code.

* **Interface** (:func:`build_demo`, :func:`main`) — Gradio wiring, imported
  lazily so that importing this module (and running its tests) does not
  require Gradio to be installed.

The verdict logic and every threshold it uses live in
:class:`VerdictThresholds` — one place, unit-tested.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import claims, face, features, hr, ingest, roi, signal
from .errors import TrueBeatError

# Where the trained G2 detector is looked up (see scripts/train_g2.py).
# Search order:
#   1. repo checkout paths — lets a locally retrained bundle override the
#      pinned one by dropping `models/detector.joblib` at the repo root;
#   2. the packaged copy: TrueBeat is local-first (the hosted demo is
#      paused), so a pip-installed tool runs the same detector without
#      any checkout beside it.
MODEL_SEARCH_PATHS = [
    Path("models/detector.joblib"),
    Path("data/models/detector.joblib"),
    Path(__file__).parent / "models" / "detector.joblib",
]
# None means "search MODEL_SEARCH_PATHS in order".
DEFAULT_MODEL_PATH = None

# Always-visible honesty caveat: the measured operating envelope of each
# layer (reports/aigen_detector_v2.md, reports/g2_detector.md, docs/11).
# The app renders it from the claims of the models it loads (claims.caveat,
# docs/13 A-07); this constant is that rendering for the packaged models,
# kept as the fallback text and checked against claims.caveat by a test.
# Numbers: reports/aigen_detector_v2.md (2026-11-01 retrain adopted under
# docs/11 §9.7; external GenBuster-Bench and FF++ c23 test clips, never used
# for training, selection or the threshold).
CAVEAT = (
    "Checks for fully AI-generated video with a whole-frame detector trained "
    "on 21 generator families. On held-out test sets it caught about 4 in 5 clips from "
    "2024-era tools (Sora, Kling, Runway, Luma, Pika), about half from the "
    "newest (Veo 3.1, Sora 2, Kling 2.6) and from some open-source ones, and "
    "wrongly flagged about 5% of real test videos (reports/aigen_detector_v2.md). "
    "Clips under 2 s are not checked. It also reads C2PA Content Credentials "
    "when present. Face-swaps: a weak pulse check can decide (it catches about a "
    "quarter of Deepfakes-style swaps and is blind to reenactment fakes such as "
    "Face2Face and NeuralTextures); a face-swap detector trained on FaceForensics++ "
    "is shown as evidence only, because it wrongly flagged 38% of real faces from "
    "other sources (docs/12). 'No sign of AI generation' is not proof a video is real."
)

# Analysis is capped so a long upload cannot run for an hour.
MAX_ANALYSIS_SECONDS = 120.0

# Progress bar weights (UI only; they change no number the analysis
# produces). Measured with `trubeat serve` on a Ryzen 5 4600H, 49 s 1080p
# clip, 2026-11-06: the AI-video layer ~5 s, decoding for the pulse ~11 ms
# and face tracking ~18 ms per frame, the pulse maths ~2 s.
PROGRESS_COST_AI_S = 5.0
PROGRESS_COST_DECODE_S_PER_FRAME = 0.011
PROGRESS_COST_FACE_S_PER_FRAME = 0.018
PROGRESS_COST_PULSE_S = 2.0
# face-swap detector: 16 frames found + embedded, plus grabbing the frames between them
PROGRESS_COST_FSD_S = 3.0
PROGRESS_COST_FSD_S_PER_FRAME = 0.002

ProgressFn = Callable[[float, str], None]


class _ProgressPlan:
    """One 0-1 progress bar spread over the stages that will run, each in
    proportion to its expected cost (the clip's frame count drives decoding
    and face tracking). A no-op when no callback is given."""

    def __init__(self, fn: ProgressFn | None, path: Path, layers: tuple[str, ...],
                 max_seconds: float) -> None:
        self.fn = fn
        self.spans: dict[str, tuple[float, float]] = {}
        if fn is None:
            return
        n_frames, fps = _frame_count(path, max_seconds)
        pulse = "pulse" in layers and fps > 0 and n_frames / fps >= ingest.MIN_DURATION_S
        n_tracked = min(n_frames, int(max_seconds * fps)) if pulse else 0
        costs = {
            "provenance": 0.1,
            "ai_video": PROGRESS_COST_AI_S if "ai_video" in layers else 0.0,
            # the face-swap detector reads frames only as far as its last sample (<= max_seconds)
            "face_swap": (PROGRESS_COST_FSD_S
                          + (min(n_frames, int(max_seconds * fps)) if fps > 0 else n_frames)
                          * PROGRESS_COST_FSD_S_PER_FRAME
                          if "face_swap" in layers else 0.0),
            "decode": n_frames * PROGRESS_COST_DECODE_S_PER_FRAME if pulse else 0.0,
            "face": n_tracked * PROGRESS_COST_FACE_S_PER_FRAME,
            "pulse": PROGRESS_COST_PULSE_S if pulse else 0.0,
        }
        total = sum(costs.values()) or 1.0
        start = 0.0
        for stage, cost in costs.items():
            self.spans[stage] = (start, start + cost / total)
            start += cost / total

    def __call__(self, stage: str, frac: float, desc: str) -> None:
        if self.fn is None or stage not in self.spans:
            return
        a, b = self.spans[stage]
        self.fn(min(0.99, a + (b - a) * max(0.0, min(1.0, frac))), desc)


def _frame_count(path: Path, max_seconds: float = MAX_ANALYSIS_SECONDS) -> tuple[int, float]:
    """(frames, fps) for the progress plan: container metadata when it is
    sane, else counted from frame timestamps (the extension's WebM reports
    -2**63 frames at "1000 fps"; docs/13 A-08). (0, 0.0) when unreadable."""
    pr = ingest.probe(path, max_seconds=max_seconds)
    return max(0, int(pr.n_frames)), float(pr.fps)


@dataclass(frozen=True)
class VerdictThresholds:
    """Every threshold the fused verdict uses, in one place (docs/11 E1).

    * The AI-video detector's threshold travels inside its bundle: set on
      out-of-fold real clips for a 5% false-positive rate, never on test
      data (scripts/train_aigen.py).
    * ``faceswap_threshold``: G2 rPPG-consistency score at or below which a
      face-swap is flagged. Strict on purpose: the validated G2 threshold
      flagged 24% of real test videos. This one was set on FF++ c23 val
      reals (n=30) for 5% and measured once on FF++ c23 test: 3 of 135 real
      videos flagged (2.2%); recall Deepfakes 24%, FaceSwap 7%, Face2Face
      3%, NeuralTextures 0%.
    * ``min_duration_s`` / ``min_face_valid_frac``: when the face-swap score
      is computed at all (docs/04 §8: 10 s, 70% tracking).
    * ``min_pulse_snr_db``: when the pulse counts as measurable
      (``hr.pulse_snr_db``; noise and drift stay at or below 0 dB, docs/11
      B2). Evidence only: on web video real faces read below it too.
    """

    min_duration_s: float = 10.0
    min_face_valid_frac: float = 0.70
    min_pulse_snr_db: float = 0.0
    faceswap_threshold: float = -0.605


DEFAULT_THRESHOLDS = VerdictThresholds()


@lru_cache(maxsize=1)
def packaged_claims() -> dict:
    """Claims of the packaged models: what verdict_for quotes for an analysis
    built by hand (tests, callers of verdict_for) rather than by analyze_video."""
    from . import aigen, faceswap

    def one(path: Path, load) -> dict | None:
        try:
            b = load(path) if path.is_file() else None
        except Exception:  # noqa: BLE001 — no packaged model: nothing to quote
            return None
        if b is not None:
            b.setdefault("_path", str(path))
        return claims.claims_for(b)

    models = Path(__file__).parent / "models"
    return {"ai_video": one(models / "aigen.joblib", aigen.load_bundle),
            "face_swap": one(models / "faceswap.joblib", faceswap.load_bundle),
            "pulse": one(models / "detector.joblib", lambda q: __import__("joblib").load(q))}


@dataclass
class PulseAnalysis:
    """Everything the verdict and the UI need, from one video.

    Three independent layers fill it (docs/11 §4): provenance (C2PA), the
    whole-frame AI-video detector, and the pulse / face-swap layer.
    ``layers`` records what each one did, in words the UI shows.
    """

    verdict: str  # "real" | "fake" | "inconclusive"
    headline: str  # one-line, human-readable verdict
    detail: str  # why
    per_roi_hr: list[float] = field(default_factory=list)
    roi_names: list[str] = field(default_factory=list)
    agreement_bpm: float | None = None
    face_valid_frac: float | None = None  # None: the pulse layer did not run
    pulse_snr_db: float | None = None  # mean over regions (hr.pulse_snr_db)
    pulse_measurable: bool | None = None
    duration_s: float = 0.0
    n_frames: int = 0
    detector_score: float | None = None  # G2 face-swap score (positive = real)
    detector_available: bool = False
    degraded: bool = False  # no G2 model -> pulse/HR analyzer only
    ai_label: str = "unknown"  # "ai" | "real" | "unknown"
    # share of the detector's reference real clips scoring below this clip
    ai_real_percentile: float | None = None
    ai_score: float | None = None  # decision value (positive = AI)
    ai_threshold: float | None = None
    ai_available: bool = False
    provenance: dict = field(default_factory=dict)
    layers: dict = field(default_factory=dict)  # layer -> what happened
    decided_by: str = ""  # "provenance" | "ai_video" | "face_swap" | ""
    faceswap_threshold: float | None = None  # the threshold actually applied
    # Face-swap detector (docs/12): face crops -> frozen backbone -> linear model
    fsd_available: bool = False
    fsd_mode: str = ""  # "decisive" | "evidence" (from the bundle, per docs/12 §6)
    fsd_label: str = "unknown"  # "swap" | "no_swap" | "no_face" | "unknown"
    fsd_score: float | None = None  # positive = face-swap
    fsd_threshold: float | None = None
    fsd_target_fpr: float | None = None  # share of real faces the threshold lets through
    fsd_outside_fpr: float | None = None  # docs/12 gate 4: real non-FaceForensics faces it flagged
    fsd_real_percentile: float | None = None
    fsd_n_faces: int = 0
    fsd_n_sampled: int = 0
    fsd_top_time_s: float | None = None
    fsd_top_crop: np.ndarray | None = field(default=None, repr=False, compare=False)
    fsd_top_frame: np.ndarray | None = field(default=None, repr=False, compare=False)
    fsd_top_box: tuple[int, int, int, int] | None = None  # x0, y0, side, side in that frame
    warnings: list[str] = field(default_factory=list)
    # The measured numbers each layer's model carries (claims.claims_for; docs/13
    # A-07) and the caveat rendered from them; empty -> the packaged models'.
    claims: dict = field(default_factory=dict, repr=False, compare=False)
    caveat: str = ""
    roi_waveforms: list[list[float]] = field(default_factory=list)  # 3 x T
    waveform_names: list[str] = field(default_factory=list)
    # The band the plotted waveforms were filtered to (Hz); the plot title
    # reads it, so the title cannot claim a filter that was not applied.
    waveform_band_hz: list[float] = field(default_factory=lambda: [0.7, 3.0])

    def to_json(self) -> str:
        d = asdict(self)
        d.pop("fsd_top_crop", None)  # images: the UI shows them, the JSON API does not carry them
        d.pop("fsd_top_frame", None)
        return json.dumps(d, default=float)


def _roi_names(config: dict) -> list[str]:
    names = config.get("rois") or list(roi.ROI_DEFS_P1)
    return [str(n) for n in names]


def _pick_bundle(explicit, candidates, load, what: str):
    """The first loadable bundle: ``explicit`` if given, else the search paths
    in order (repo overrides first, the packaged copy last). A bundle found by
    search that would override the packaged one without measured numbers
    (claims) is refused, so no text can quote numbers that belong to another
    model (docs/13 A-07). ``load(path)`` returns the bundle or raises."""
    cands = [Path(explicit)] if explicit else list(candidates)
    packaged = Path(__file__).parent / "models"
    refused: list[str] = []
    for path in cands:
        if not path.is_file():
            continue
        try:
            bundle = load(path)
        except Exception as exc:  # noqa: BLE001 — a broken bundle must not break the app
            print(f"warning: could not load {what} {path}: {exc}")
            continue
        if bundle is None:
            continue
        bundle.setdefault("_path", str(path))
        overriding = explicit is None and path.resolve().parent != packaged.resolve()
        if overriding and claims.claims_for(bundle) is None:
            note = (f"W_OVERRIDE_REFUSED: {what} {path} records no measured numbers (claims), so the "
                    "packaged model was used (docs/13 A-07)")
            print(f"warning: {note}")
            refused.append(note)
            continue
        if refused:   # shown with the result, not only on the server's console
            from .bundles import WARNINGS_KEY

            bundle[WARNINGS_KEY] = list(bundle.get(WARNINGS_KEY) or []) + refused
        return bundle
    return None


def load_detector(model_path: Path | str | None = DEFAULT_MODEL_PATH):
    """Load the trained G2 detector bundle, or None if it is not available.

    The app degrades gracefully: without a detector it still measures the
    pulse (heart rate, inter-region agreement, QC) and says so.
    """
    def load(path: Path) -> dict:
        from .bundles import WARNINGS_KEY, load_checked

        bundle, load_warnings = load_checked(path)   # W_SKLEARN_VERSION (docs/13 D-05)
        if not (isinstance(bundle, dict) and "model" in bundle):
            bundle = {"model": bundle, "config": {}, "threshold": 0.0}
        bundle[WARNINGS_KEY] = load_warnings
        return bundle

    return _pick_bundle(model_path, MODEL_SEARCH_PATHS, load, "pulse detector")


# One loaded AI-video bundle + backbone per process: loading DINOv2 takes
# seconds, analysing a clip should not pay that every time.
_AIGEN_CACHE: dict[str, tuple] = {}


def load_aigen(path: Path | str | None = None):
    """(bundle, Embedder) for the AI-video layer, or (None, None)."""
    from . import aigen

    key = str(path) if path else "<search>"
    if key not in _AIGEN_CACHE:
        try:
            bundle = _pick_bundle(path, aigen.BUNDLE_SEARCH_PATHS, aigen.load_bundle, "AI-video bundle")
            emb = aigen.Embedder(bundle["config"]) if bundle else None
        except Exception as exc:  # noqa: BLE001 — a broken bundle must not break the app (docs/13 A-14)
            print(f"warning: could not load AI-video bundle: {exc}")
            bundle, emb = None, None
        _AIGEN_CACHE[key] = (bundle, emb)
    return _AIGEN_CACHE[key]


_FS_CACHE: dict = {}


def load_faceswap(path: Path | str | None = None):
    """(bundle, FaceEmbedder, FaceFinder) for the face-swap layer, or Nones."""
    from . import faceswap

    key = str(path) if path else "<search>"
    if key not in _FS_CACHE:
        try:
            bundle = _pick_bundle(path, faceswap.BUNDLE_SEARCH_PATHS, faceswap.load_bundle, "face-swap bundle")
            cfg = faceswap.merged_config(bundle["config"]) if bundle is not None else None
        except Exception as exc:  # noqa: BLE001 — a broken bundle must not break the app
            print(f"warning: could not load face-swap bundle: {exc}")
            bundle = None
        if bundle is None:
            return None, None, None  # not cached: a bundle installed later is picked up
        try:
            if bundle.get("kind") == "cnn":
                try:
                    emb = faceswap.CNNScorer.from_bundle(bundle)
                except Exception as exc:  # noqa: BLE001 — weights not fetched: the layer says so
                    print(f"warning: face-swap CNN not usable: {exc}")
                    emb = None
            else:
                emb = faceswap.FaceEmbedder(cfg["backbone"], cfg["feature"], int(cfg["torch_threads"]),
                                            img_size=int(cfg["crop_size"]))
            finder = faceswap.FaceFinder(cfg)
        except Exception as exc:  # noqa: BLE001 — a malformed bundle must not break the app (docs/13 A-14)
            print(f"warning: could not load face-swap bundle: {exc}")
            return None, None, None
        if emb is None:
            # not cached: `trubeat fetch-models` may supply the weights later; no
            # finder either, so faceswap.analyze opens and closes its own
            finder.close()
            return bundle, None, None
        _FS_CACHE[key] = (bundle, emb, finder)
    return _FS_CACHE[key]


def analyze_video(
    video_path: str | Path,
    config: dict | None = None,
    model_path: Path | str | None = DEFAULT_MODEL_PATH,
    thresholds: VerdictThresholds | None = None,
    max_seconds: float = MAX_ANALYSIS_SECONDS,
    aigen_path: Path | str | None = None,
    layers: tuple[str, ...] = ("provenance", "ai_video", "face_swap", "pulse"),
    progress: ProgressFn | None = None,
    faceswap_path: Path | str | None = None,
    region: tuple[float, float, float, float] | None = None,
) -> PulseAnalysis:
    """Run every analysis layer on one video and fuse them (docs/11 E1).

    Layers are independent: a clip too short for the pulse layer (< 10 s)
    is still checked for provenance and by the AI-video detector (>= 2 s).
    Never raises for bad input: what each layer could not do is recorded in
    ``layers`` and, when nothing could run, the verdict is inconclusive.
    ``progress(fraction, stage)``, when given, is called as each stage
    advances (the web app's progress bar); it never reaches 1.0 here.
    ``region`` = (x0, y0, x1, y1) in source pixels limits every layer to that
    part of the frame (the browser extension's video element, docs/13 §3.4).
    """
    config = dict(config or {})
    path = Path(video_path)
    a = PulseAnalysis(verdict="inconclusive", headline="", detail="")
    if not path.is_file():
        _quote(a, model_claims(model_path, aigen_path, faceswap_path, layers))   # bundle files only
        a.warnings.append("E_UNREADABLE")
        a.layers["file"] = f"not a readable file: {path.name}"
        return verdict_for(a, thresholds or DEFAULT_THRESHOLDS)
    bar = _ProgressPlan(progress, path, layers, max_seconds)
    # Each layer returns the bundle it scored with; the texts quote the claims
    # of exactly those objects (docs/13 A-07), loaded once per analysis.
    used: dict = {}
    if "provenance" in layers:
        bar("provenance", 0.0, "Reading Content Credentials")
        _provenance_layer(a, path)
    if "ai_video" in layers:
        bar("ai_video", 0.0, "AI-video detector: loading the model")
        used["ai_video"] = _ai_video_layer(a, path, aigen_path, bar, region)
    if "face_swap" in layers:
        bar("face_swap", 0.0, "Face-swap detector: loading the model")
        used["face_swap"] = _face_swap_layer(a, path, faceswap_path, bar, region)
    thr = thresholds or DEFAULT_THRESHOLDS
    if "pulse" in layers:
        det = load_detector(model_path)          # its claims set the strict threshold, its model scores
        used["pulse"] = det
        strict = (claims.claims_for(det) or {}).get("strict_threshold")
        if thresholds is None and strict is not None:
            thr = replace(DEFAULT_THRESHOLDS, faceswap_threshold=float(strict))
        bar("decode", 0.0, "Pulse: decoding the video")
        _pulse_layer(a, path, config, model_path, thr, max_seconds, bar, region, detector=det)
    _quote(a, claims_of(used))
    return verdict_for(a, thr)


def claims_of(bundles: dict) -> dict:
    """Claims per layer of the given bundles ({layer: bundle or None}), plus
    the face-swap bundle's mode under ``fsd_mode`` (docs/13 A-07)."""
    out: dict = {"fsd_mode": "evidence"}
    for layer, b in bundles.items():
        out[layer] = claims.claims_for(b)
    if bundles.get("face_swap") is not None:
        out["fsd_mode"] = str(bundles["face_swap"].get("mode", "evidence"))
    return out


def _quote(a: PulseAnalysis, cl: dict) -> None:
    """Attach the claims and the caveat rendered from them; a text never
    breaks an analysis (docs/13 A-14): on any error the caveat quotes no numbers."""
    a.claims = cl
    try:
        a.caveat = claims.caveat(cl.get("ai_video"), cl.get("pulse"), cl.get("face_swap"),
                                 fsd_mode=cl.get("fsd_mode", "evidence"))
    except Exception:  # noqa: BLE001
        a.caveat = claims.caveat(None, None, None, fsd_mode=cl.get("fsd_mode", "evidence"))


def model_claims(model_path=DEFAULT_MODEL_PATH, aigen_path=None, faceswap_path=None,
                 layers: tuple[str, ...] = ("ai_video", "face_swap", "pulse")) -> dict:
    """The claims of the models an analysis would use, per layer, plus the
    face-swap bundle's mode under ``fsd_mode`` (docs/13 A-07), from the small
    bundle files alone (no backbone): the UI's caveat before any analysis.
    An analysis quotes the bundles it actually scored with (``claims_of``)."""
    from . import aigen, faceswap

    b: dict = {}
    if "ai_video" in layers:
        b["ai_video"] = _pick_bundle(aigen_path, aigen.BUNDLE_SEARCH_PATHS, aigen.load_bundle, "AI-video bundle")
    if "face_swap" in layers:
        b["face_swap"] = _pick_bundle(faceswap_path, faceswap.BUNDLE_SEARCH_PATHS, faceswap.load_bundle,
                                      "face-swap bundle")
    if "pulse" in layers:
        b["pulse"] = load_detector(model_path)
    return claims_of(b)


def current_caveat() -> str:
    """The caveat for the models a default analysis would load (bundle files
    only; no backbone is loaded)."""
    c = model_claims()
    return claims.caveat(c.get("ai_video"), c.get("pulse"), c.get("face_swap"), fsd_mode=c["fsd_mode"])


def _provenance_layer(a: PulseAnalysis, path: Path) -> None:
    from . import provenance

    r = provenance.read_provenance(path)
    a.provenance = asdict(r)
    a.warnings += [w for w in r.warnings if w not in a.warnings]
    if r.ai_declared:
        a.layers["provenance"] = "Content Credentials declare AI generation"
    elif r.has_c2pa:
        a.layers["provenance"] = "Content Credentials present, no AI declaration"
    else:
        a.layers["provenance"] = "no Content Credentials (re-encoding usually strips them)"


def _ai_video_layer(a: PulseAnalysis, path: Path, aigen_path,
                    bar: _ProgressPlan | None = None, region=None) -> dict | None:
    """The whole-frame AI-video detector; returns the bundle it used (or None)."""
    from . import aigen
    from .bundles import WARNINGS_KEY
    from .errors import ModelMissing

    bundle, embedder = load_aigen(aigen_path)
    if bundle is None:
        a.layers["ai_video"] = "not available: no trained AI-video model installed"
        a.warnings.append("W_NO_AIGEN_MODEL")
        return None
    a.warnings += [w for w in bundle.get(WARNINGS_KEY) or [] if w not in a.warnings]
    try:
        res = aigen.analyze(
            path, bundle, embedder=embedder,
            progress=(lambda f, s: bar("ai_video", f, f"AI-video detector: {s}")) if bar else None,
            region=region,
        )
    except ModelMissing as exc:
        a.layers["ai_video"] = f"not available: {exc}"
        a.warnings.append(exc.code)
        return bundle
    except Exception as exc:  # noqa: BLE001 — the app must not crash
        a.layers["ai_video"] = f"failed: {type(exc).__name__}: {exc}"
        a.warnings.append("W_AIGEN_FAILED")
        return bundle
    a.ai_available = True
    a.ai_label = res.label
    a.ai_real_percentile = res.real_percentile
    a.ai_score = res.score
    a.ai_threshold = res.threshold
    a.warnings += [w for w in res.warnings if w not in a.warnings]
    if res.label == "unknown":
        a.layers["ai_video"] = "could not analyse: " + (res.warnings[-1] if res.warnings else "")
    else:
        a.layers["ai_video"] = (
            f"{'flags AI generation' if res.label == 'ai' else 'no AI-generation signs'} "
            f"(score {res.score:+.2f}, threshold {res.threshold:+.2f}, "
            f"{res.n_segments} segment{'s' if res.n_segments != 1 else ''})"
        )
    if not a.duration_s and res.duration_s:
        a.duration_s = float(res.duration_s)
    return bundle


def _face_swap_layer(a: PulseAnalysis, path: Path, faceswap_path,
                     bar: _ProgressPlan | None = None, region=None) -> dict | None:
    """The face-swap detector (docs/12): runs on any clip with a visible face;
    returns the bundle it used (or None)."""
    from . import faceswap
    from .bundles import WARNINGS_KEY
    from .errors import ModelMissing

    bundle, emb, finder = load_faceswap(faceswap_path)
    if bundle is None:
        a.layers["face_swap"] = "not available: no trained face-swap model installed"
        return None
    a.warnings += [w for w in bundle.get(WARNINGS_KEY) or [] if w not in a.warnings]
    try:
        res = faceswap.analyze(
            path, bundle, embedder=emb, finder=finder,
            progress=(lambda f, s: bar("face_swap", f, f"Face-swap detector: {s}")) if bar else None,
            region=region,
        )
        mode = str(bundle.get("mode", "evidence"))
        target_fpr = float(faceswap.merged_config(bundle["config"])["target_fpr"])
        gate4 = bundle.get("gate4") or {}
        outside = float(gate4["rate"]) if gate4.get("rate") is not None else None
    except ModelMissing as exc:
        a.layers["face_swap"] = f"not available: {exc}"
        a.warnings.append(exc.code)
        return bundle
    except Exception as exc:  # noqa: BLE001 — the app must not crash, even on a malformed bundle
        a.layers["face_swap"] = f"failed: {type(exc).__name__}: {exc}"
        a.warnings.append("W_FACESWAP_DETECTOR_FAILED")
        return bundle
    a.fsd_mode, a.fsd_target_fpr, a.fsd_outside_fpr = mode, target_fpr, outside
    a.fsd_label, a.fsd_score, a.fsd_threshold = res.label, res.score, res.threshold
    a.fsd_real_percentile, a.fsd_n_faces, a.fsd_n_sampled = res.real_percentile, res.n_faces, res.n_sampled
    a.fsd_top_time_s, a.fsd_top_crop = res.top_time_s, res.top_crop
    a.fsd_top_frame, a.fsd_top_box = res.top_frame, res.top_box
    a.fsd_available = res.label in ("swap", "no_swap")
    a.warnings += [w for w in res.warnings if w not in a.warnings]
    if a.fsd_available:
        a.layers["face_swap"] = (
            f"{'flags a face-swap' if res.label == 'swap' else 'no face-swap signs'} "
            f"(score {res.score:+.2f}, threshold {res.threshold:+.2f}, face in {res.n_faces} of "
            f"{res.n_sampled} sampled frames{'' if a.fsd_mode == 'decisive' else ', evidence only'})"
        )
    else:
        a.layers["face_swap"] = "could not run: " + (res.warnings[-1] if res.warnings else res.label)
    if not a.duration_s and res.duration_s:
        a.duration_s = float(res.duration_s)
    return bundle


_LOAD = object()   # _pulse_layer's default: load the detector itself


def _pulse_layer(
    a: PulseAnalysis, path: Path, config: dict, model_path, thr: VerdictThresholds,
    max_seconds: float,
    bar: _ProgressPlan | None = None,
    region=None,
    detector=_LOAD,
) -> None:
    """Face tracking, per-region pulse, and the G2 face-swap score.
    ``detector`` = the bundle analyze_video already loaded (None: none)."""
    bar = bar or _ProgressPlan(None, path, (), max_seconds)
    try:
        stream = ingest.open_video(
            str(path),
            frame_height=config.get("frame_height", ingest.DEFAULT_FRAME_HEIGHT),
            progress=lambda i, n: bar("decode", i / n, f"Pulse: decoding frame {i} of {n}"),
            max_seconds=max_seconds,     # docs/13 A-02: never decode past what is analysed
            region=region,
        )
    except TrueBeatError as exc:
        if exc.code == "E_SHORT_VIDEO":
            a.layers["pulse"] = "skipped: clip shorter than the 10 s the pulse layer needs"
        else:
            a.layers["pulse"] = f"skipped: {exc.code}: {exc}"
            if exc.code == "E_UNREADABLE":
                a.warnings.append("E_UNREADABLE")
        a.warnings.append(f"W_PULSE_SKIPPED: {exc.code}")
        return
    except Exception as exc:  # noqa: BLE001 — the app must not crash
        a.layers["pulse"] = f"skipped: unreadable video ({exc})"
        a.warnings.append("E_UNREADABLE")
        return

    duration = float(stream.duration_s)
    fps = stream.fps
    frames = stream.frames
    if stream.truncated:
        whole = f" of a {stream.full_duration_s:.0f}s clip" if stream.full_duration_s else ""
        a.warnings.append(f"W_TRUNCATED: the pulse layer analysed the first {duration:.0f}s{whole}")

    # The G2 bundle decides the rPPG method and window (it was trained with
    # them; using the config's would feed the model the wrong features).
    bundle = load_detector(model_path) if detector is _LOAD else detector
    if bundle is not None:
        from .bundles import WARNINGS_KEY

        a.warnings += [w for w in bundle.get(WARNINGS_KEY, []) if w not in a.warnings]
    try:
        if bundle is not None:
            method = str(bundle.get("method") or config.get("method", "green"))
            win_s = float(bundle.get("window_s", config.get("window_s", 10.0)))
        else:
            method = str(config.get("method", "green"))
            win_s = float(config.get("window_s", 10.0))
    except Exception as exc:  # noqa: BLE001 — a malformed bundle must not break the app (docs/13 A-14)
        a.layers["pulse"] = f"failed: the pulse model bundle is malformed ({type(exc).__name__})"
        a.warnings.append("W_PULSE_FAILED")
        return

    try:
        landmarks, valid = face.track_landmarks(
            frames, fps,
            progress=lambda i, n: bar("face", i / n, f"Pulse: tracking the face, frame {i} of {n}"),
        )
        bar("pulse", 0.0, "Pulse: measuring the pulse and the face-swap score")
        face_valid_frac = float(np.mean(valid)) if len(valid) else 0.0
        traces = roi.extract_roi_means(
            frames, landmarks, valid, config.get("roi_defs", roi.ROI_DEFS_P1)
        )
        ppg = signal.extract_ppg(traces, fps, method=method)
    except Exception as exc:  # noqa: BLE001 — one layer failing must not lose the others
        a.layers["pulse"] = f"failed: {type(exc).__name__}: {exc}"
        a.warnings.append("W_PULSE_FAILED")
        a.duration_s = a.duration_s or duration
        return
    names = _roi_names(config)

    # Heart rate and pulse SNR on the band-passed signal (docs/11 B1a/B2):
    # the raw green channel locks the spectral peak onto slow drift.
    hr_band = tuple(config.get("band", (0.7, 3.0)))
    per_roi_hr: list[float] = []
    snrs: list[float] = []
    for r in range(ppg.shape[1]):
        try:
            cond = signal.condition(ppg[:, r], fps, band=hr_band)
        except ValueError:
            cond = ppg[:, r]
        bpm = hr.estimate_hr(cond, fps, band=hr_band).bpm
        per_roi_hr.append(float(bpm) if np.isfinite(bpm) else float("nan"))
        snr = hr.pulse_snr_db(cond, fps, bpm=bpm if np.isfinite(bpm) else None, band=hr_band)
        if np.isfinite(snr):
            snrs.append(float(snr))
    finite = [b for b in per_roi_hr if np.isfinite(b)]

    # G2 face-swap score: only on a tracked face (an untracked face gives
    # zero-filled traces, docs/11 F10).
    detector_score: float | None = None
    swap_note = ""
    face_ok = face_valid_frac >= thr.min_face_valid_frac
    if not face_ok:
        a.warnings.append(f"W_UNSTABLE_TRACK: face tracked in {face_valid_frac:.0%} of frames "
                          f"(needs {thr.min_face_valid_frac:.0%})")   # docs/04 §8; docs/13 A-10
    if bundle is not None and face_ok:
        try:
            feat = features.window_features(
                ppg, fps, win_s=win_s, overlap=float(config.get("overlap", 0.5))
            )
            cols = bundle.get("feature_columns") or list(features.FEATURE_COLUMNS)
            if len(feat) and all(c in feat.columns for c in cols):
                dec = np.asarray(bundle["model"].decision_function(feat[cols]), dtype=float)
                detector_score = float(np.mean(dec)) if dec.size else None
            else:
                swap_note = "face-swap check skipped (no complete analysis window)"
        except Exception as exc:  # noqa: BLE001 — never crash the app
            a.warnings.append(f"W_FACESWAP_FAILED: {type(exc).__name__}")
            swap_note = f"face-swap check failed ({type(exc).__name__})"

    a.per_roi_hr = per_roi_hr
    a.roi_names = names[: len(per_roi_hr)]
    a.agreement_bpm = float(np.std(finite)) if len(finite) > 1 else None
    a.face_valid_frac = face_valid_frac
    a.pulse_snr_db = float(np.mean(snrs)) if snrs else None
    a.pulse_measurable = (
        a.pulse_snr_db is not None and a.pulse_snr_db >= thr.min_pulse_snr_db
    )
    a.duration_s = duration
    a.n_frames = int(frames.shape[0])
    a.detector_score = detector_score
    a.faceswap_threshold = thr.faceswap_threshold
    a.detector_available = bundle is not None
    a.degraded = bundle is None
    a.roi_waveforms = _waveforms(ppg, fps, band=hr_band)
    a.waveform_names = names[: ppg.shape[1]]
    a.waveform_band_hz = [float(hr_band[0]), float(hr_band[1])]

    parts = [f"face tracked in {face_valid_frac:.0%} of frames"]
    if a.pulse_snr_db is not None:
        parts.append(
            f"pulse {'measurable' if a.pulse_measurable else 'not measurable'} "
            f"({a.pulse_snr_db:+.1f} dB)"
        )
    if detector_score is not None:
        parts.append(f"face-swap score {detector_score:+.2f}")
    elif swap_note:
        parts.append(swap_note)
    elif not face_ok:
        parts.append("face-swap check skipped (face not tracked reliably)")
    elif bundle is None:
        parts.append("face-swap check unavailable (no model)")
    a.layers["pulse"] = "; ".join(parts)


def _waveforms(
    ppg: np.ndarray,
    fps: float | None = None,
    band: tuple[float, float] = (0.7, 3.0),
    max_points: int = 1500,
) -> list[list[float]]:
    """Down-sampled ROI waveforms for plotting (NaNs replaced by 0).

    With ``fps`` given, each ROI signal is detrended and band-passed to
    ``band`` first (``signal.condition``), i.e. the signal the heart-rate
    step analyses. The raw green channel sits at 100-250 intensity units
    with slow drift and shows no pulse by eye (docs/11 F11).
    """
    out = []
    for r in range(ppg.shape[1]):
        col = np.asarray(ppg[:, r], dtype=float)
        if fps is not None:
            try:
                col = signal.condition(col, fps, band=band)
            except ValueError:
                pass  # band not representable at this fps: plot unfiltered
        col = np.nan_to_num(col, nan=0.0, posinf=0.0, neginf=0.0)
        if col.size > max_points:
            idx = np.linspace(0, col.size - 1, max_points).astype(int)
            col = col[idx]
        out.append([float(v) for v in col])
    return out


def verdict_for(analysis: PulseAnalysis, thresholds: VerdictThresholds | None = None) -> PulseAnalysis:
    """Fuse the layers into the three-state verdict, in place (docs/11 E1).

    Order, strongest evidence first (docs/12 §5 B):
      1. Content Credentials declaring AI generation -> ``fake``.
      2. The whole-frame AI-video detector over its threshold -> ``fake``.
      3. The face-swap detector over its threshold, when its bundle is
         ``decisive`` (docs/12 §6) -> ``fake`` ("possible face-swap").
      4. The pulse check's strict face-swap score -> ``fake``, only when the
         face-swap detector could not decide (no bundle, no face, or
         evidence-only mode).
      5. The AI-video detector ran and nothing fired -> ``real``
         ("no sign of AI generation found", never "authentic"); an
         evidence-only face-swap flag is stated in the detail.
      6. Nothing could decide -> ``inconclusive``, listing why per layer.

    The pulse rate itself does not decide: on web-quality video it is not
    measurable for real faces either (docs/11, Phase B result).
    """
    thr = thresholds or DEFAULT_THRESHOLDS
    a = analysis
    prov = a.provenance or {}
    cl = a.claims or packaged_claims()
    ai_cl, pulse_cl = cl.get("ai_video"), cl.get("pulse")

    # Content Credentials decide only when their signature checks out
    # (docs/13 A-19): an unsigned or broken manifest declaring AI generation
    # is a claim anyone can paste into a real video, so it is evidence only.
    prov_note = ""
    if prov.get("ai_declared"):
        gen = prov.get("generator")
        state = prov.get("validation_state")
        if state in ("Valid", "Trusted"):
            a.verdict, a.decided_by = "fake", "provenance"
            a.headline = "AI-generated: declared by its Content Credentials"
            a.detail = (
                "The file carries a C2PA manifest whose history declares generation "
                f"by a trained model{f' ({gen})' if gen else ''}; its signature checks out ({state})."
            )
            return a
        prov_note = (f" Its Content Credentials claim AI generation{f' ({gen})' if gen else ''}, but their "
                     f"signature could not be verified ({state or 'no validation result'}), so that claim "
                     "was not used to decide.")

    if a.ai_label == "ai" and a.ai_score is not None:
        a.verdict, a.decided_by = "fake", "ai_video"
        a.headline = "Likely AI-generated"
        a.detail = (
            f"The whole-frame detector scores this clip {a.ai_score:+.2f}, above its "
            f"threshold of {a.ai_threshold:+.2f}"
            + (f" (set so that {claims.about(float(ai_cl['target_fpr']))} of real clips cross it)."
               if ai_cl and ai_cl.get("target_fpr") is not None else ".")
        )
        if a.ai_real_percentile is not None:
            a.detail += (
                f" That is higher than {a.ai_real_percentile:.0%} of the real "
                "validation clips."
            )
        a.detail += prov_note
        return a

    fsd_decides = a.fsd_mode == "decisive" and a.fsd_available
    if fsd_decides and a.fsd_label == "swap":
        a.verdict, a.decided_by = "fake", "face_swap"
        a.headline = "Possible face-swap"
        a.detail = (
            f"The face-swap detector scores the face {a.fsd_score:+.2f}, at or above its "
            f"threshold of {a.fsd_threshold:+.2f} (set so that about "
            f"{a.fsd_target_fpr or 0.03:.0%} of real faces cross it)."
        )
        if a.fsd_real_percentile is not None:
            a.detail += f" That is higher than {a.fsd_real_percentile:.0%} of its real training faces."
        if a.fsd_top_time_s is not None:
            a.detail += f" The most suspicious face is at {a.fsd_top_time_s:.1f} s (shown below)."
        if a.ai_label == "real":
            a.detail += " The whole-frame AI-video detector did not flag the clip."
        a.detail += prov_note
        return a

    # The pulse check decides only when the face-swap detector could not
    # look at a face (no bundle, no face found, or evidence-only mode).
    if (not fsd_decides and a.detector_score is not None
            and a.detector_score <= thr.faceswap_threshold):
        a.verdict, a.decided_by = "fake", "pulse"
        a.headline = "Possible face-swap"
        a.detail = (
            f"The face-region consistency score ({a.detector_score:+.2f}) is at or "
            f"below the strict face-swap threshold ({thr.faceswap_threshold:+.2f}). "
            + (f"This check is weak: it catches {claims.about(float(pulse_cl['recall']['Deepfakes']))} of "
               f"Deepfakes-style swaps and flags {claims.about(float(pulse_cl['real_fpr']['rate']))} of real "
               "face videos." if pulse_cl else "This check is weak; its model records no measured accuracy.")
        )
        if a.fsd_available:  # evidence only here (fsd_decides is False on this path)
            a.detail += (f" The face-swap detector (evidence only) "
                         f"{'also flags' if a.fsd_label == 'swap' else 'does not flag'} the face "
                         f"({a.fsd_score:+.2f}, threshold {a.fsd_threshold:+.2f}).")
        if a.ai_label == "real":
            a.detail += " The whole-frame AI-video detector did not flag the clip."
        a.detail += prov_note
        return a

    if a.ai_label == "real" and a.ai_score is not None:
        a.verdict, a.decided_by = "real", "ai_video"
        a.headline = "No sign of AI generation found"
        if fsd_decides:
            swap = f"The face-swap detector also found no sign of a face-swap ({a.fsd_score:+.2f})."
        elif a.fsd_available and a.fsd_label == "swap":
            outside = (f": on real videos from outside its training data it flagged "
                       f"{a.fsd_outside_fpr:.0%} of faces, so this flag alone means little"
                       if a.fsd_outside_fpr is not None
                       else ": it is not yet accurate enough to decide on its own")
            swap = (f"The face-swap detector does flag the face ({a.fsd_score:+.2f}, threshold "
                    f"{a.fsd_threshold:+.2f}), but it is shown as evidence only{outside}.")
        elif a.fsd_available:
            swap = (f"The face-swap detector (evidence only) found no sign of a face-swap "
                    f"({a.fsd_score:+.2f}).")
        elif a.detector_score is not None:
            swap = "The pulse face-swap check also passed."
        else:
            swap = (f"No face-swap check could run ({a.layers.get('face_swap', 'not run')}; "
                    f"pulse: {a.layers.get('pulse', 'not run')}).")
        a.detail = (
            f"The whole-frame detector scores this clip {a.ai_score:+.2f}, below its "
            f"threshold of {a.ai_threshold:+.2f}. {swap} This is not proof the video "
            "is authentic: see the caveat."
        )
        a.detail += prov_note
        return a

    a.verdict, a.decided_by = "inconclusive", ""
    unreadable = "E_UNREADABLE" in a.warnings
    a.headline = (
        "Inconclusive: could not read the video"
        if unreadable
        else "Inconclusive: no check could reach a decision"
    )
    a.detail = "; ".join(f"{k}: {v}" for k, v in a.layers.items()) + "." + prov_note
    return a


# ---------------------------------------------------------------------------
# Interface (Gradio) — imported lazily so the analysis half stays usable
# without Gradio installed.
# ---------------------------------------------------------------------------


# Web look: TrueBeat website v4 (trubeat.vercel.app). Black ink page, dark
# panels, paper text, safety orange accent, sharp corners, hard offset
# shadows. Fonts are a local stack only: the UI must make zero network
# requests, so no webfont links ever ship with the app.
UI_CSS = """
.gradio-container{background:#0D0D0D !important;color:#F4F0E8;
  font-family:'Archivo','Arial Black',system-ui,sans-serif !important}
.gradio-container .prose, .gradio-container .prose p{color:#F4F0E8}
.gradio-container h1, .gradio-container h2, .gradio-container h3{color:#F4F0E8}
#tb-head h1{text-transform:uppercase;letter-spacing:.01em;font-weight:900}
[data-testid="block-label"]{background:#FF4D00 !important;color:#0D0D0D !important;
  font-weight:800;text-transform:uppercase;letter-spacing:.07em;font-size:11px;
  border-radius:0 !important}
#tb-video{border:2px solid rgba(244,240,232,.85) !important;border-radius:0 !important;
  background:#141414 !important}
#tb-video .upload-container{background:#141414 !important;border-radius:0 !important;
  transition:transform .15s, box-shadow .15s}
#tb-video .upload-container:hover{box-shadow:4px 4px 0 rgba(255,77,0,.5)}
#tb-video [data-testid="upload-text"]{color:#F4F0E8}
#tb-analyse{background:#FF4D00 !important;color:#0D0D0D !important;font-weight:800;
  text-transform:uppercase;letter-spacing:.04em;border:2px solid #FF4D00 !important;
  border-radius:0 !important;box-shadow:5px 5px 0 rgba(244,240,232,.85) !important;
  transition:transform .15s, box-shadow .15s}
#tb-analyse:hover{transform:translate(-2px,-2px);
  box-shadow:7px 7px 0 rgba(244,240,232,.85) !important}
#tb-evidence table{border-collapse:collapse;width:100%}
#tb-evidence th, #tb-evidence td{border:1px solid rgba(244,240,232,.32);
  padding:6px 10px;text-align:left;color:#F4F0E8}
#tb-evidence th{background:#141414;text-transform:uppercase;font-weight:800;
  letter-spacing:.05em;font-size:.85em}
#tb-evidence tr:nth-child(even) td{background:rgba(244,240,232,.03)}
#tb-caveat{background:#141414;border:2px solid #FF4D00;border-radius:0;
  padding:12px 14px;color:#F4F0E8}
#tb-caveat strong{color:#FF4D00}
#tb-verdict{font-size:1.05em}
a{color:#F4F0E8 !important}
.progress-bar{background-color:#FF4D00 !important}
"""


def _ui_theme():
    """The v4 web look as token overrides on the bare Base theme."""
    import gradio as gr

    fonts = gr.themes.utils.fonts
    return gr.themes.Base(
        # radius_none: sharp corners everywhere (no Soft theme, no rounding)
        radius_size=gr.themes.sizes.radius_none,
        # Font wrapper = bare font-family name, no webfont, zero requests.
        # (Plain strings work at the constructor but break blocks.launch's
        # theme comparison, which expects font objects.)
        font=[
            fonts.Font("Archivo"),
            fonts.Font("Arial Black"),
            fonts.Font("system-ui"),
            fonts.Font("sans-serif"),
        ],
        font_mono=[
            fonts.Font("ui-monospace"),
            fonts.Font("Consolas"),
            fonts.Font("monospace"),
        ],
    ).set(
        body_background_fill="#0D0D0D",
        body_background_fill_dark="#0D0D0D",
        background_fill_primary="#0D0D0D",
        background_fill_secondary="#141414",
        panel_background_fill="#141414",
        panel_background_fill_dark="#141414",
        block_background_fill="#141414",
        block_background_fill_dark="#141414",
        input_background_fill="#1A1A1A",
        input_background_fill_dark="#1A1A1A",
        body_text_color="#F4F0E8",
        body_text_color_dark="#F4F0E8",
        body_text_color_subdued="#A89F8F",
        block_title_text_color="#F4F0E8",
        block_label_background_fill="#FF4D00",
        block_label_text_color="#0D0D0D",
        block_border_color="rgba(244,240,232,.32)",
        input_border_color="rgba(244,240,232,.32)",
        border_color_accent="#FF4D00",
        border_color_accent_subdued="rgba(255,77,0,.5)",
        button_primary_background_fill="#FF4D00",
        button_primary_background_fill_dark="#FF4D00",
        button_primary_background_fill_hover="#FF4D00",
        button_primary_text_color="#0D0D0D",
        button_primary_border_color="#FF4D00",
        button_secondary_background_fill="#141414",
        button_secondary_text_color="#F4F0E8",
        link_text_color="#F4F0E8",
        loader_color="#FF4D00",
        error_background_fill="#1A1A1A",
        error_text_color="#F4F0E8",
        # none of gradio's soft shadows: shadows here are hard offsets (CSS)
        shadow_inset="none",
        shadow_drop="none",
        shadow_drop_lg="none",
        shadow_spread="0",
    )


def suspicious_frame(a: PulseAnalysis, max_width: int = 960) -> np.ndarray | None:
    """The frame the face-swap detector found most suspicious, shrunk to at
    most ``max_width`` pixels wide, with that face outlined in the UI accent
    (RGB). None when the detector scored no face."""
    import cv2

    frame, box = a.fsd_top_frame, a.fsd_top_box
    if frame is None or box is None:
        return None
    h, w = frame.shape[:2]
    k = min(1.0, max_width / max(w, 1))
    img = cv2.resize(frame, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA) if k < 1 else frame.copy()
    x0, y0, side, _ = box
    p0 = (max(0, round(x0 * k)), max(0, round(y0 * k)))
    p1 = (min(img.shape[1] - 1, round((x0 + side) * k)), min(img.shape[0] - 1, round((y0 + side) * k)))
    cv2.rectangle(img, p0, p1, (255, 77, 0), max(2, round(img.shape[1] / 240)))
    return img


# Moves the uploaded video's preview to the most suspicious frame (seconds),
# paused, once the analysis is shown. Runs in the browser only.
_SEEK_JS = """(t) => {
  const v = document.querySelector('#tb-video video');
  if (v && t !== null && t !== undefined && !Number.isNaN(Number(t))) {
    v.pause();
    v.currentTime = Number(t);
  }
}"""


def api_payload(a: PulseAnalysis) -> dict:
    """The JSON the browser extension reads (one place, unit-tested; docs/13 D-08).
    Images never travel here."""
    finite_hr = [b for b in a.per_roi_hr if np.isfinite(b)]
    return {
        "verdict": a.verdict,
        "headline": a.headline,
        "detail": a.detail,
        "score": a.detector_score,
        "hr_bpm": float(np.mean(finite_hr)) if finite_hr else None,
        "region_agreement_bpm": a.agreement_bpm,
        "snr": a.pulse_snr_db,
        "pulse_measurable": a.pulse_measurable,
        "face_valid_frac": a.face_valid_frac,
        "ai_label": a.ai_label,
        "ai_real_percentile": a.ai_real_percentile,
        "ai_score": a.ai_score,
        "ai_threshold": a.ai_threshold,
        "decided_by": a.decided_by,
        # The face-swap detector (docs/12): its own score and threshold, so a
        # verdict it decides carries its numbers.
        **{k: getattr(a, k) for k in (
            "fsd_mode", "fsd_label", "fsd_score", "fsd_threshold", "fsd_target_fpr", "fsd_outside_fpr",
            "fsd_real_percentile", "fsd_n_faces", "fsd_n_sampled", "fsd_top_time_s")},
        "layers": a.layers,
        "provenance": a.provenance,
        "duration_s": a.duration_s,
        "warnings": list(a.warnings),
        "per_roi_hr": {n: (float(b) if np.isfinite(b) else None) for n, b in zip(a.roi_names, a.per_roi_hr)},
        "qc": {
            "ok": a.verdict != "inconclusive",
            "face_valid_frac": a.face_valid_frac,
            "snr": a.pulse_snr_db,
            "duration_s": a.duration_s,
            "degraded": a.degraded,
            "warnings": a.warnings,
        },
        "caveat": a.caveat or CAVEAT,
    }


API_PAYLOAD_KEYS = (
    "verdict", "headline", "detail", "score", "hr_bpm", "region_agreement_bpm", "snr", "pulse_measurable",
    "face_valid_frac", "ai_label", "ai_real_percentile", "ai_score", "ai_threshold", "decided_by", "fsd_mode",
    "fsd_label", "fsd_score", "fsd_threshold", "fsd_target_fpr", "fsd_outside_fpr", "fsd_real_percentile",
    "fsd_n_faces", "fsd_n_sampled", "fsd_top_time_s", "layers", "provenance", "duration_s", "warnings",
    "per_roi_hr", "qc", "caveat",
)


def parse_region(text: str | None) -> tuple[float, float, float, float] | None:
    """The extension's region: JSON {"x0":..,"y0":..,"x1":..,"y1":..} in the
    recording's pixels, or empty. Anything else is ignored (whole frame)."""
    if not text:
        return None
    try:
        r = json.loads(text) if isinstance(text, str) else dict(text)
        x0, y0, x1, y1 = (float(r[k]) for k in ("x0", "y0", "x1", "y1"))
    except (ValueError, TypeError, KeyError):
        return None
    if not all(np.isfinite(v) for v in (x0, y0, x1, y1)) or x1 - x0 < 16 or y1 - y0 < 16:
        return None
    return x0, y0, x1, y1


# Uploads (and the rendered face image) live in Gradio's cache; files older
# than this are deleted every few minutes (docs/13 A-04). The extension's
# captures are deleted as soon as their analysis returns.
UPLOAD_MAX_AGE_S = 3600
UPLOAD_SWEEP_S = 300


def build_demo(
    config: dict | None = None,
    model_path: Path | str | None = DEFAULT_MODEL_PATH,
    thresholds: VerdictThresholds | None = None,
):
    """Build the Gradio interface. Requires `gradio` to be installed."""
    import gradio as gr

    cfg = dict(config or {})

    def run(video_path, progress=None):
        if progress is None:
            progress = gr.Progress()
        if not video_path:
            return ("Upload a video first.", "", "", gr.update(value=None, visible=False), None, None)
        progress(0.0, desc="Starting")
        analysis = analyze_video(
            video_path, cfg, model_path=model_path, thresholds=thresholds,
            progress=lambda frac, stage: progress(frac, desc=stage),
        )
        progress(1.0, desc="Done")
        evidence = _evidence_text(analysis)
        img = suspicious_frame(analysis)
        if img is None:  # the panel appears only when the detector scored a face
            face, seek = gr.update(value=None, visible=False), None
        else:
            caption = (f"Most suspicious frame, at {analysis.fsd_top_time_s:.1f} s: face-swap score "
                       f"{analysis.fsd_score:+.2f}"
                       f"{'' if analysis.fsd_mode == 'decisive' else ', evidence only'}")
            face, seek = gr.update(value=img, label=caption, visible=True), analysis.fsd_top_time_s
        return (analysis.headline, analysis.detail, evidence, face, _plot(analysis), seek)

    # Note (gradio 6+): theme/CSS moved from the Blocks constructor to
    # demo.launch(); they are applied in main() from _ui_theme()/UI_CSS.
    # No analytics beacon, whoever launches the app (docs/13 A-18), and an
    # hourly sweep of uploaded videos (A-04).
    with gr.Blocks(title="TrueBeat: AI-video and deepfake check", analytics_enabled=False,
                   delete_cache=(UPLOAD_SWEEP_S, UPLOAD_MAX_AGE_S)) as demo:
        gr.Markdown(
            "# TrueBeat: AI-video and deepfake check\n"
            "Upload a video (2–120 s). Every check runs on this machine. "
            "Expect roughly **10 s plus 20 s per minute of video** on a CPU.",
            elem_id="tb-head",
        )
        with gr.Row():
            with gr.Column():
                video = gr.Video(
                    label="Video (2–120 s)",
                    sources=["upload"],
                    elem_id="tb-video",
                )
                gr.Markdown(
                    "The preview may not play for HEVC / iPhone video; the analysis still runs.",
                    elem_id="tb-hevc",
                )
                btn = gr.Button("Analyse", variant="primary", elem_id="tb-analyse")
            with gr.Column():
                verdict = gr.Markdown(elem_id="tb-verdict")
                detail = gr.Markdown(elem_id="tb-detail")
                evidence = gr.Markdown(elem_id="tb-evidence")
                # Hidden until an analysis has scored a face; the whole frame,
                # scaled to fit (no inner scrolling), with the face outlined.
                face_img = gr.Image(label="Most suspicious frame", elem_id="tb-face", type="numpy",
                                    interactive=False, visible=False)
                seek = gr.Number(visible=False, elem_id="tb-seek")
        plot = gr.Plot(label="Per-region pulse waveforms", elem_id="tb-plot")
        gr.Markdown(f"**Caveat: always applies:** {current_caveat()}", elem_id="tb-caveat")
        # One loader, one place: the Plot panel is the output the user
        # watches. Without this Gradio renders the progress overlay on every
        # output (a duplicate sat over the caveat). Outputs stay unchanged.
        # The caveat is static (outputs no longer overwrite it, so its heading
        # stays; docs/13 C-15). Every analysis shares one queue, so the web
        # app and the extension never run two at once on a 7 GB laptop (A-05).
        btn.click(
            run,
            inputs=[video],
            outputs=[verdict, detail, evidence, face_img, plot, seek],
            show_progress_on=[plot],
            concurrency_id="analysis",
        ).then(None, inputs=[seek], outputs=None, js=_SEEK_JS)
        # a new upload hides the previous clip's frame until it is analysed
        video.change(lambda _v: gr.update(value=None, visible=False), inputs=[video],
                     outputs=[face_img], show_progress="hidden")

        # Thin, machine-readable endpoint for the browser extension
        # (Phase 4b). It only *packages* what analyze_video already
        # computed — no new pipeline, no new thresholds, no recalibration.
        # Wired through a hidden button so the input type is the same
        # FileData the upload flow sends.
        def analyze_json(video_path, region_text=None):
            """The extension's call: analyse the uploaded capture (optionally only
            the video element's region), then delete the capture (docs/13 A-04)."""
            try:
                a = analyze_video(video_path, cfg, model_path=model_path, thresholds=thresholds,
                                  region=parse_region(region_text))
                return json.dumps(api_payload(a))
            finally:
                try:
                    if video_path:
                        Path(video_path).unlink(missing_ok=True)
                except OSError:
                    pass

        api_btn = gr.Button(visible=False)
        api_out = gr.Textbox(label="result", visible=False)
        api_region = gr.Textbox(label="region", visible=False)
        api_btn.click(
            lambda v: analyze_json(v, None),
            inputs=[video],
            outputs=[api_out],
            api_name="analyze_video",
            api_description=(
                "Analyse a video of a face; returns JSON with the verdict, "
                "the evidence numbers and the standing caveat."
            ),
            concurrency_id="analysis",
        )
        api_btn2 = gr.Button(visible=False)
        api_btn2.click(
            analyze_json,
            inputs=[video, api_region],
            outputs=[api_out],
            api_name="analyze_video_region",
            api_description=(
                "As analyze_video, limited to a region of the frame given as JSON "
                '{"x0":..,"y0":..,"x1":..,"y1":..} in the video\'s pixels (the extension sends the '
                "video element's box)."
            ),
            concurrency_id="analysis",
        )
    return demo


def _evidence_text(a: PulseAnalysis) -> str:
    """Markdown evidence table: one row per layer, then the pulse details."""

    def fmt(v, spec):
        return "n/a" if v is None else format(v, spec)

    if a.ai_score is not None:
        ai_text = f"score {a.ai_score:+.2f} (flags at >= {a.ai_threshold:+.2f})"
        if a.ai_real_percentile is not None:
            ai_text += f"; higher than {a.ai_real_percentile:.0%} of real validation clips"
    else:
        ai_text = a.layers.get("ai_video", "not run")
    prov = a.provenance or {}
    prov_text = a.layers.get("provenance", "not run")
    if prov.get("generator"):
        prov_text += f" ({prov['generator']})"
    pulse_layer = a.layers.get("pulse", "")
    if a.detector_score is not None:
        swap_thr = (a.faceswap_threshold if a.faceswap_threshold is not None
                    else DEFAULT_THRESHOLDS.faceswap_threshold)
        swap_text = f"score {a.detector_score:+.2f} (flags at <= {swap_thr:+.2f})"
    elif pulse_layer.startswith(("skipped", "failed")):
        swap_text = "n/a (" + pulse_layer + ")"
    elif "face-swap check" in pulse_layer:
        swap_text = "n/a (" + pulse_layer.split("; ")[-1] + ")"
    elif a.detector_available:
        swap_text = "n/a (not computed: face not tracked reliably)"
    else:
        swap_text = "n/a (no model: degraded mode)"
    if a.fsd_available:
        fsd_text = (f"score {a.fsd_score:+.2f} (flags at >= {a.fsd_threshold:+.2f})"
                    + (f"; higher than {a.fsd_real_percentile:.0%} of its real training faces"
                       if a.fsd_real_percentile is not None else "")
                    + f"; face in {a.fsd_n_faces} of {a.fsd_n_sampled} frames"
                    + ("" if a.fsd_mode == "decisive" else "; evidence only"
                       + (f" (flagged {a.fsd_outside_fpr:.0%} of real faces outside its training data)"
                          if a.fsd_outside_fpr is not None else "")))
    else:
        fsd_text = a.layers.get("face_swap", "not run")
    hr_text = ", ".join(
        f"{n}: {b:.1f} bpm" for n, b in zip(a.roi_names, a.per_roi_hr) if np.isfinite(b)
    ) or "n/a"
    if a.pulse_snr_db is not None and not a.pulse_measurable:
        hr_text += " (not reliable: pulse not measurable)"
    if a.pulse_snr_db is not None:
        pulse_text = (
            f"{'measurable' if a.pulse_measurable else 'not measurable'} "
            f"(SNR {a.pulse_snr_db:+.1f} dB)"
        )
    else:
        pulse_text = "n/a"
    rows = [
        "| evidence | value |",
        "|---------|-------|",
        f"| AI-video detector | {ai_text} |",
        f"| Content Credentials | {prov_text} |",
        f"| face-swap detector | {fsd_text} |",
        f"| face-swap check (pulse) | {swap_text} |",
        f"| pulse | {pulse_text} |",
        f"| per-region heart rate | {hr_text} |",
        f"| region agreement | {fmt(a.agreement_bpm, '.1f')} bpm spread |",
        (f"| face tracked | {a.face_valid_frac:.0%} of frames |"
         if a.face_valid_frac is not None else "| face tracked | not measured |"),
        f"| clip length | {a.duration_s:.1f} s |",
        f"| decided by | {a.decided_by or 'nothing (no layer could decide)'} |",
    ]
    if a.warnings:
        rows.append(f"| warnings | {'; '.join(a.warnings)} |")
    return "\n".join(rows)


def _plot(a: PulseAnalysis):
    """Small matplotlib plot of the per-ROI pulse waveforms."""
    if not a.roi_waveforms:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 2.6), dpi=110)
    fig.patch.set_facecolor("#0D0D0D")
    ax.set_facecolor("#0D0D0D")
    # website palette across the three ROIs: orange, paper, muted warm grey
    line_colors = ("#FF4D00", "#F4F0E8", "#A89F8F")
    for i, (name, wave) in enumerate(zip(a.waveform_names, a.roi_waveforms)):
        ax.plot(wave, linewidth=0.7, label=name, color=line_colors[i % 3])
    lo, hi = (a.waveform_band_hz + [0.7, 3.0])[:2]
    ink = "#F4F0E8"  # paper text on the black panel
    ax.set_xlabel("frame", color=ink)
    ax.set_ylabel("band-passed signal", color=ink)
    ax.set_title(
        f"Per-region pulse waveforms (band-passed {lo:g}-{hi:g} Hz)", color=ink
    )
    ax.tick_params(colors=ink, labelcolor=ink)
    for spine in ax.spines.values():
        spine.set_color("#A89F8F")
    legend = ax.legend(loc="upper right", fontsize=8, frameon=False)
    for text in legend.get_texts():
        text.set_color("#F4F0E8")
    fig.tight_layout()
    return fig


def main(argv: list[str] | None = None) -> int:
    """`trubeat serve` entry point."""
    import argparse

    p = argparse.ArgumentParser(prog="trubeat serve")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", default="127.0.0.1",
                   help="address to listen on (default: this machine only; 0.0.0.0 lets every device on "
                        "the network upload videos to it)")
    p.add_argument("--share", action="store_true",
                   help="create a temporary public Gradio link (uploads leave "
                        "this machine: use only for demos)")
    p.add_argument("--config", default=None)
    p.add_argument("--model", default=None,
                   help="detector bundle path (default: search models/, "
                        "data/models/, then the packaged copy)")
    args = p.parse_args(argv)

    # Offline by architecture: no analytics beacon unless the user asked
    # for one (an explicitly set value wins; this only sets the default).
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "0")

    try:
        import yaml
    except ImportError:  # pragma: no cover — pyyaml is a hard dependency
        yaml = None

    config: dict = {}
    if args.config and yaml is not None:
        with open(args.config) as f:
            config = yaml.safe_load(f) or {}
        if isinstance(config.get("band"), list):
            config["band"] = tuple(config["band"])

    # availability probe (gradio itself is only imported inside launch paths)
    import importlib.util

    if importlib.util.find_spec("gradio") is None:
        print(
            "Gradio is not installed. Install the optional extra:\n"
            "    uv pip install 'trubeat[serving]'\n"
            "or:  uv pip install gradio"
        )
        return 1

    demo = build_demo(config, model_path=args.model)
    if args.share:
        print("WARNING: --share exposes this app publicly. Uploaded videos "
              "leave this machine. Use it for demos only.")
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"WARNING: listening on {args.host}: any device that can reach this machine can upload "
              "videos to it. Use only on a network you trust.")
    demo.launch(
        server_name=args.host,          # docs/13 A-03: this machine only, unless asked
        server_port=args.port,
        share=args.share,
        max_file_size="2gb",            # A-04: an upload cannot fill the disk (or a RAM-backed /tmp)
        show_error=True,                # C-01: the extension sees the real error, not {"error": null}
        # the v4 web look (theme + CSS moved from Blocks to launch in gradio 6)
        theme=_ui_theme(),
        css=UI_CSS,
    )
    return 0
