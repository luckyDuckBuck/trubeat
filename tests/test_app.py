"""Tests for trubeat.app — the web app's analysis + verdict logic.

The verdict logic is pure (no Gradio), so it is tested directly. The
integration test runs the real pipeline on a synthetic video; it skips
when the MediaPipe face model is not available (CI).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synth

from trubeat import app as app_mod
from trubeat.app import PulseAnalysis, VerdictThresholds, verdict_for

THR = VerdictThresholds(
    min_duration_s=10.0,
    min_face_valid_frac=0.70,
    min_pulse_snr_db=0.0,
    faceswap_threshold=-0.6,
)


def _analysis(**kw) -> PulseAnalysis:
    """An analysis where every layer ran and nothing fired; override per test."""
    base = {
        "verdict": "inconclusive",
        "headline": "",
        "detail": "",
        "per_roi_hr": [70.0, 70.2, 69.8],
        "roi_names": ["forehead", "cheek_l", "cheek_r"],
        "agreement_bpm": 0.2,
        "face_valid_frac": 1.0,
        "pulse_snr_db": -3.0,
        "pulse_measurable": False,
        "duration_s": 30.0,
        "n_frames": 900,
        "detector_score": 0.5,
        "detector_available": True,
        "degraded": False,
        "ai_label": "real",
        "ai_real_percentile": 0.2,
        "ai_score": -1.0,
        "ai_threshold": 0.4,
        "ai_available": True,
        "provenance": {"has_c2pa": False, "ai_declared": False},
        "layers": {"provenance": "no Content Credentials", "ai_video": "ran", "pulse": "ran"},
    }
    base.update(kw)
    return PulseAnalysis(**base)


# ---------------------------------------------------------------- verdict ---
# docs/11 E1: provenance > AI-video detector > strict face-swap > "no sign".


def test_nothing_fires_means_no_sign_of_ai_never_authentic():
    a = verdict_for(_analysis(), THR)
    assert a.verdict == "real"
    assert a.decided_by == "ai_video"
    assert a.headline == "No sign of AI generation found"
    assert "authentic" not in a.headline.lower()
    assert "not proof" in a.detail


def test_ai_detector_over_threshold_is_fake():
    a = verdict_for(_analysis(ai_label="ai", ai_score=1.2, ai_real_percentile=0.99), THR)
    assert a.verdict == "fake" and a.decided_by == "ai_video"
    assert a.headline == "Likely AI-generated"
    assert "+1.20" in a.detail and "+0.40" in a.detail
    assert "higher than 99% of the real validation clips" in a.detail


def test_provenance_declaration_outranks_the_detectors():
    a = verdict_for(
        _analysis(provenance={"has_c2pa": True, "ai_declared": True, "generator": "GenModel 2",
                              "validation_state": "Valid"}),
        THR,
    )
    assert a.verdict == "fake" and a.decided_by == "provenance"
    assert "Content Credentials" in a.headline
    assert "GenModel 2" in a.detail


def test_unverified_provenance_is_evidence_not_a_verdict():
    # docs/13 A-19: a manifest whose signature does not check out is a claim anyone
    # can paste into a real video; it is stated, and the other layers decide
    for state in ("Invalid", None):
        a = verdict_for(
            _analysis(provenance={"has_c2pa": True, "ai_declared": True, "generator": "GenModel 2",
                                  "validation_state": state}),
            THR,
        )
        assert a.decided_by != "provenance" and a.verdict == "real"
        assert "could not be verified" in a.detail and "not used to decide" in a.detail
        assert "GenModel 2" in a.detail


def test_faceswap_boundary_is_inclusive():
    at = verdict_for(_analysis(detector_score=THR.faceswap_threshold), THR)
    above = verdict_for(_analysis(detector_score=THR.faceswap_threshold + 1e-9), THR)
    assert at.verdict == "fake" and at.decided_by == "pulse"  # no face-swap detector ran
    assert at.headline == "Possible face-swap"
    assert "weak" in at.detail
    assert above.verdict == "real"


def test_ai_detector_outranks_the_faceswap_check():
    a = verdict_for(_analysis(ai_label="ai", ai_score=1.0, ai_real_percentile=0.97,
                              detector_score=-5.0), THR)
    assert a.decided_by == "ai_video"


def test_short_clip_still_gets_a_verdict_from_the_ai_detector():
    """The 10 s pulse minimum no longer ends the analysis (docs/11 F3)."""
    a = verdict_for(
        _analysis(duration_s=6.0, detector_score=None, detector_available=True,
                  layers={"ai_video": "ran", "pulse": "skipped: clip shorter than the 10 s"}),
        THR,
    )
    assert a.verdict == "real"
    assert "No face-swap check could run" in a.detail


def test_without_the_ai_detector_a_passing_faceswap_check_is_not_a_real_verdict():
    """The face-swap check alone is too weak to clear a video."""
    a = verdict_for(
        _analysis(ai_label="unknown", ai_score=None, ai_available=False,
                  layers={"ai_video": "not available: no trained AI-video model installed",
                          "pulse": "face tracked in 100% of frames"}),
        THR,
    )
    assert a.verdict == "inconclusive"
    assert "ai_video: not available" in a.detail


def test_nothing_ran_is_inconclusive_with_a_reason_per_layer():
    a = verdict_for(
        _analysis(ai_label="unknown", ai_score=None, detector_score=None,
                  layers={"ai_video": "could not analyse: E_SHORT_VIDEO",
                          "pulse": "skipped: clip shorter than the 10 s"}),
        THR,
    )
    assert a.verdict == "inconclusive"
    assert a.decided_by == ""
    assert "ai_video: could not analyse" in a.detail and "pulse: skipped" in a.detail


def test_unreadable_file_says_so():
    a = app_mod.analyze_video("/definitely/not/here.mp4")
    assert a.verdict == "inconclusive"
    assert a.headline == "Inconclusive: could not read the video"


def test_the_pulse_alone_never_decides():
    """A measurable or unmeasurable pulse changes no verdict (docs/11 B2)."""
    for snr, ok in ((-6.0, False), (+4.0, True)):
        a = verdict_for(_analysis(pulse_snr_db=snr, pulse_measurable=ok), THR)
        assert a.verdict == "real" and a.decided_by == "ai_video"


def test_caveat_states_the_measured_envelope():
    c = app_mod.CAVEAT
    assert "21 generator families" in c and "reports/aigen_detector_v2.md" in c
    assert "5%" in c and "under 2 s" in c  # measured false-alarm rate, scope
    assert "Face2Face" in c and "NeuralTextures" in c
    assert "evidence only" in c and "38%" in c  # the face-swap detector (docs/12 §9)
    assert "not proof" in c


def test_thresholds_live_in_one_place():
    fields = set(VerdictThresholds.__dataclass_fields__)
    assert {"min_duration_s", "min_face_valid_frac", "min_pulse_snr_db",
            "faceswap_threshold"} <= fields


def test_load_detector_returns_none_when_missing(tmp_path):
    assert app_mod.load_detector(tmp_path / "nope.joblib") is None


def test_load_detector_reads_a_bundle(tmp_path):
    import joblib

    p = tmp_path / "m.joblib"
    joblib.dump({"model": object(), "threshold": -0.5}, p)
    bundle = app_mod.load_detector(p)
    assert bundle is not None
    assert bundle["threshold"] == -0.5


def test_load_detector_finds_the_packaged_copy_from_any_cwd(tmp_path, monkeypatch):
    """A pip-installed tool must find the detector with no checkout beside it.

    The G2 bundle is committed inside the package (src/trubeat/models/) and
    MODEL_SEARCH_PATHS ends with a package-relative entry. From a foreign
    working directory — exactly the situation in a Hugging Face Space — the
    search must still resolve it.
    """
    monkeypatch.chdir(tmp_path)  # no repo-root models/, no data/models/
    bundle = app_mod.load_detector(None)
    assert bundle is not None, "packaged detector not found via search"
    assert "model" in bundle
    assert Path(bundle["_path"]).parts[-3:] == ("trubeat", "models", "detector.joblib")


# ------------------------------------------------------------ integration ---


def _has_face_model() -> bool:
    from trubeat import face as face_mod

    try:
        return face_mod.resolve_model_path().is_file()
    except RuntimeError:
        return False


needs_model = pytest.mark.skipif(
    not _has_face_model(), reason="face_landmarker.task not downloaded"
)


@pytest.fixture
def no_aigen(monkeypatch):
    """Integration tests must not depend on a trained AI-video bundle."""
    monkeypatch.setattr(app_mod, "load_aigen", lambda path=None: (None, None))


@needs_model
def test_integration_synthetic_video_runs_the_pipeline(tmp_path, no_aigen):
    """A synthetic pulse video has no face: every layer runs or says why,
    and with no AI-video model the verdict is inconclusive, not a crash."""
    p = synth.write_pulse_video(tmp_path / "v.mp4", fps=30.0, duration_s=12.0)
    a = app_mod.analyze_video(p, config={"method": "green"})
    assert a.verdict == "inconclusive"
    assert "ai_video: not available" in a.detail
    assert "face tracked in 0% of frames" in a.layers["pulse"]
    assert a.duration_s == pytest.approx(12.0, abs=0.5)
    assert a.face_valid_frac == 0.0
    assert a.detector_available is True  # the model exists
    # ...but no score is computed from a faceless (zero-filled) trace
    # (docs/11 F10: this clip type used to show a meaningless "+4.925").
    assert a.detector_score is None
    assert len(a.roi_waveforms) == 3  # the plot data was produced anyway
    assert "no Content Credentials" in a.layers["provenance"]


@needs_model
def test_integration_short_video_skips_only_the_pulse_layer(tmp_path, no_aigen):
    p = synth.write_pulse_video(tmp_path / "short.mp4", fps=30.0, duration_s=4.0)
    a = app_mod.analyze_video(p, config={"method": "green"})
    assert a.layers["pulse"].startswith("skipped: clip shorter")
    assert "provenance" in a.layers  # the other layers still ran
    assert a.verdict == "inconclusive"  # no AI-video model in this test


def test_integration_short_video_gets_an_ai_verdict(tmp_path, monkeypatch):
    """With an AI-video bundle, a 4 s clip is scored (docs/11 F3)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_aigen import _StubEmbedder, _tiny_bundle

    bundle = _tiny_bundle()
    monkeypatch.setattr(app_mod, "load_aigen", lambda path=None: (bundle, _StubEmbedder()))
    p = synth.write_pulse_video(tmp_path / "short.mp4", fps=30.0, duration_s=4.0)
    a = app_mod.analyze_video(p, config={"method": "green"}, layers=("provenance", "ai_video"))
    assert a.ai_available and a.ai_label in ("ai", "real")
    assert a.verdict in ("fake", "real")
    assert a.decided_by == "ai_video"


def test_malformed_face_swap_bundle_fails_the_layer_not_the_app(tmp_path, monkeypatch):
    # review 2026-11-07: bundle fields read after scoring must not escape the layer's guard
    from trubeat import faceswap

    res = faceswap.FaceSwapResult(score=2.0, threshold=1.0, label="swap", n_sampled=16, n_faces=16)
    monkeypatch.setattr(faceswap, "analyze", lambda *a, **k: res)
    bad = {"mode": "evidence", "config": {}, "gate4": "not a dict"}
    monkeypatch.setattr(app_mod, "load_faceswap", lambda path=None: (bad, object(), object()))
    a = _analysis(fsd_available=False)
    app_mod._face_swap_layer(a, tmp_path / "clip.mp4", None)
    assert a.layers["face_swap"].startswith("failed") and not a.fsd_available
    assert "W_FACESWAP_DETECTOR_FAILED" in a.warnings


def test_integration_missing_file_is_inconclusive_not_a_crash(tmp_path):
    a = app_mod.analyze_video(tmp_path / "does_not_exist.mp4")
    assert a.verdict == "inconclusive"
    assert "E_UNREADABLE" in a.warnings


def test_cli_exposes_serve():
    # The subparser exists (its presence is what the user-facing command needs).
    from trubeat import cli

    src = Path(cli.__file__).read_text()
    assert '"serve"' in src
    assert '"--share"' in src


def test_evidence_table_explains_a_missing_score():
    """The face-swap row says *why* there is no score."""
    with_score = app_mod._evidence_text(_analysis(detector_score=-0.5))
    assert "| face-swap check (pulse) | score -0.50 (flags at <=" in with_score
    assert ("| AI-video detector | score -1.00 (flags at >= +0.40); "
            "higher than 20% of real validation clips |") in with_score
    assert "(not reliable: pulse not measurable)" in with_score
    no_face = app_mod._evidence_text(
        _analysis(detector_score=None, detector_available=True, face_valid_frac=0.0)
    )
    assert "not computed: face not tracked reliably" in no_face
    degraded = app_mod._evidence_text(
        _analysis(detector_score=None, detector_available=False, degraded=True)
    )
    assert "no model: degraded mode" in degraded


def test_waveforms_are_band_passed_not_raw():
    """docs/11 F11: the plot showed the raw green channel (DC ~200, drift)
    under a 'conditioned' title. With fps given, the plotted waveform is the
    band-passed signal: no DC offset, drift removed, the pulse kept."""
    import numpy as np

    fps, n = 25.0, 500
    t = np.arange(n) / fps
    pulse = np.sin(2 * np.pi * 1.2 * t)  # 72 bpm
    raw = 200.0 + 0.05 * np.arange(n) + pulse  # DC + slow drift + pulse
    ppg = np.column_stack([raw, raw, raw])
    (w, *_rest) = app_mod._waveforms(ppg, fps, band=(0.7, 3.0))
    w = np.asarray(w)
    assert abs(w.mean()) < 0.05  # DC and drift gone
    core = slice(50, -50)  # ignore filter edge effects
    assert np.corrcoef(w[core], pulse[core])[0, 1] > 0.95  # the pulse survives


def test_plot_title_names_the_applied_band_and_is_legible():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.colors import to_hex

    a = _analysis(
        roi_waveforms=[[0.0, 1.0, 0.0]] * 3,
        waveform_names=["forehead", "cheek_l", "cheek_r"],
        waveform_band_hz=[0.7, 3.0],
    )
    fig = app_mod._plot(a)
    ax = fig.axes[0]
    assert "band-passed 0.7-3 Hz" in ax.get_title()
    bg = to_hex(ax.get_facecolor())
    for text in (ax.title, ax.xaxis.label, ax.yaxis.label):
        assert to_hex(text.get_color()) != bg
        assert to_hex(text.get_color()).lower() == "#f4f0e8"


def test_evidence_uses_the_threshold_actually_applied():
    """Review #9: the table printed DEFAULT_THRESHOLDS, not the one used."""
    txt = app_mod._evidence_text(_analysis(detector_score=-0.7, faceswap_threshold=-0.9))
    assert "(flags at <= -0.90)" in txt


def test_face_tracking_not_measured_is_not_zero():
    """Review #8: a skipped pulse layer must not report 0% face visibility."""
    a = _analysis(face_valid_frac=None, detector_score=None,
                  layers={"ai_video": "ran", "pulse": "skipped: clip shorter than the 10 s"})
    txt = app_mod._evidence_text(verdict_for(a, THR))
    assert "| face tracked | not measured |" in txt
    assert "0% of frames" not in txt


def test_a_failing_pulse_layer_keeps_the_other_layers(tmp_path, monkeypatch):
    """Review #6: an exception inside face tracking used to raise out of
    analyze_video and discard the provenance and AI-video results."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_aigen import _StubEmbedder, _tiny_bundle

    bundle = _tiny_bundle()
    monkeypatch.setattr(app_mod, "load_aigen", lambda path=None: (bundle, _StubEmbedder()))

    def boom(*_a, **_k):
        raise RuntimeError("landmarker model missing")

    monkeypatch.setattr(app_mod.face, "track_landmarks", boom)
    p = synth.write_pulse_video(tmp_path / "v.mp4", fps=30.0, duration_s=11.0)
    a = app_mod.analyze_video(p, config={"method": "green"})
    assert a.layers["pulse"].startswith("failed: RuntimeError")
    assert a.ai_available and a.decided_by == "ai_video"
    assert "W_PULSE_FAILED" in a.warnings


@needs_model
def test_progress_moves_forward_through_every_stage(tmp_path, no_aigen):
    """The web app's bar (docs/11): fractions only grow, stay below 1.0
    until the app says done, and the face-tracking stage reports per frame."""
    p = synth.write_pulse_video(tmp_path / "v.mp4", fps=30.0, duration_s=12.0)
    calls = []
    app_mod.analyze_video(p, config={"method": "green"},
                          progress=lambda f, s: calls.append((f, s)))
    fracs = [f for f, _ in calls]
    assert fracs == sorted(fracs) and 0.0 <= fracs[0] and fracs[-1] < 1.0
    stages = " | ".join(s for _, s in calls)
    for want in ("Content Credentials", "AI-video detector", "decoding frame", "tracking the face, frame",
                 "measuring the pulse"):
        assert want in stages
    assert sum("tracking the face" in s for _, s in calls) >= 10  # per-frame, not one jump
    assert fracs[-1] > 0.8  # the bar ends near the end, not stuck at 10%


def test_progress_plan_gives_short_clips_to_the_ai_layer():
    """Under 10 s the pulse layer never runs, so its share must not hold
    the bar back."""
    import unittest.mock as m
    with m.patch.object(app_mod, "_frame_count", return_value=(120, 24.0)):  # 5 s clip
        plan = app_mod._ProgressPlan(lambda f, s: None, Path("x.mp4"), ("provenance", "ai_video", "pulse"), 120.0)
    assert plan.spans["ai_video"][1] == pytest.approx(1.0)
    assert plan.spans["face"][1] - plan.spans["face"][0] == 0.0
    with m.patch.object(app_mod, "_frame_count", return_value=(1186, 24.0)):  # 49 s clip
        plan = app_mod._ProgressPlan(lambda f, s: None, Path("x.mp4"), ("provenance", "ai_video", "pulse"), 120.0)
    face_share = plan.spans["face"][1] - plan.spans["face"][0]
    assert 0.4 < face_share < 0.7  # face tracking is the long stage


# ------------------------------------------------- face-swap detector (docs/12)

def _fsd(label, score, mode="decisive", **kw):
    return _analysis(fsd_available=label in ("swap", "no_swap"), fsd_label=label, fsd_score=score,
                     fsd_threshold=0.5, fsd_mode=mode, fsd_real_percentile=0.99, fsd_n_faces=16,
                     fsd_n_sampled=16, fsd_top_time_s=3.2, **kw)


def test_decisive_face_swap_detector_flags_a_swap():
    a = verdict_for(_fsd("swap", 1.4), THR)
    assert a.verdict == "fake" and a.decided_by == "face_swap" and a.headline == "Possible face-swap"
    assert "+1.40" in a.detail and "3.2 s" in a.detail and "about 3% of real faces" in a.detail


def test_ai_video_detector_still_comes_first():
    a = verdict_for(_fsd("swap", 1.4, ai_label="ai", ai_score=2.0, ai_threshold=1.1), THR)
    assert a.decided_by == "ai_video"


def test_evidence_only_detector_never_decides():
    a = verdict_for(_fsd("swap", 1.4, mode="evidence"), THR)
    assert a.verdict == "real" and a.decided_by == "ai_video"
    # ... but its flag is stated plainly, never as a check that passed or did not run (review 2026-11-06)
    assert "does flag the face (+1.40" in a.detail and "evidence only" in a.detail
    assert "passed" not in a.detail and "could run" not in a.detail
    b = verdict_for(_fsd("swap", 1.4, mode="evidence", detector_score=0.5), THR)  # pulse passed too
    assert "does flag the face" in b.detail and "pulse" not in b.detail


def test_evidence_only_flag_states_the_measured_outside_false_alarm_rate():
    # docs/12 §9: gate 4 measured 38% on real non-FaceForensics faces; a flag must say so
    a = verdict_for(_fsd("swap", 1.4, mode="evidence", fsd_outside_fpr=0.383), THR)
    assert a.verdict == "real" and "flagged 38% of faces" in a.detail
    assert "means little" in a.detail


def test_shipped_face_swap_bundle_mode_follows_its_recorded_gates():
    # the packaged bundle is tracked in git, so this runs on every checkout and in CI
    from trubeat import faceswap
    b = faceswap.load_bundle(Path(app_mod.__file__).parent / "models" / "faceswap.joblib")
    assert b is not None, "src/trubeat/models/faceswap.joblib is missing"
    g4 = b.get("gate4")
    assert isinstance(g4, dict), "the bundle was not installed by `train_faceswap.py ship`"
    assert g4["n"] >= 200 and 0.0 <= g4["rate"] <= 1.0
    decisive = all(ok is not False for _, ok, _ in b["gates"]) and g4["passed"]
    assert b["mode"] == ("decisive" if decisive else "evidence")  # docs/12 §6


def test_caveats_quote_the_shipped_bundles_false_alarm_rate():
    # CAVEAT and the extension's standing caveat state the gate-4 rate in words;
    # re-shipping a bundle with another rate must update both (docs/12 §9)
    from trubeat import faceswap
    b = faceswap.load_bundle(Path(app_mod.__file__).parent / "models" / "faceswap.joblib")
    rate = f"{b['gate4']['rate']:.0%}"
    overlay = (Path(__file__).parents[1] / "extension" / "overlay.js").read_text()
    if b["mode"] == "evidence":
        assert rate in app_mod.CAVEAT and rate in overlay


def test_pulse_decided_swap_states_what_the_evidence_only_detector_saw():
    a = verdict_for(_fsd("swap", 2.94, mode="evidence", detector_score=-0.77), THR)
    assert a.decided_by == "pulse" and "(evidence only) also flags the face (+2.94" in a.detail
    b = verdict_for(_fsd("no_swap", -0.5, mode="evidence", detector_score=-0.77), THR)
    assert b.decided_by == "pulse" and "does not flag the face (-0.50" in b.detail


def test_face_swap_detail_states_the_bundles_false_alarm_target():
    a = verdict_for(_fsd("swap", 1.4, fsd_target_fpr=0.05), THR)
    assert "about 5% of real faces" in a.detail


def test_decisive_detector_takes_over_from_the_pulse_check():
    """Once the face-swap detector looked at the face, the weak pulse score
    no longer decides (docs/12 §5 B)."""
    a = verdict_for(_fsd("no_swap", -2.0, detector_score=-0.9), THR)
    assert a.verdict == "real" and "face-swap detector also found no sign" in a.detail
    b = verdict_for(_fsd("no_face", None, detector_score=-0.9), THR)  # no face found: pulse may decide
    assert b.decided_by == "pulse"


def test_json_api_leaves_out_the_face_image():
    import json

    import numpy as np

    a = _fsd("swap", 1.4)
    a.fsd_top_crop = np.zeros((8, 8, 3), np.uint8)
    a.fsd_top_frame = np.zeros((90, 160, 3), np.uint8)
    d = json.loads(verdict_for(a, THR).to_json())
    assert "fsd_top_crop" not in d and "fsd_top_frame" not in d
    assert d["fsd_score"] == 1.4 and d["decided_by"] == "face_swap"


def test_suspicious_frame_outlines_the_face_and_fits_the_panel():
    import numpy as np

    a = _fsd("no_swap", -0.5)
    assert app_mod.suspicious_frame(a) is None  # no frame: the panel stays hidden
    a.fsd_top_frame = np.zeros((1080, 1920, 3), np.uint8)
    a.fsd_top_box = (900, 400, 200, 200)
    img = app_mod.suspicious_frame(a)
    assert img.shape == (540, 960, 3)  # shrunk to the panel width, aspect kept
    assert tuple(img[200, 450]) == (255, 77, 0)  # the box edge, in the UI accent
    assert img[250, 500].sum() == 0  # inside the box: the frame itself, untouched
    assert a.fsd_top_frame.sum() == 0  # the analysis' frame is not drawn on
    a.fsd_top_box = (-50, -50, 400, 400)  # a box past the frame edge is clipped, not an error
    assert app_mod.suspicious_frame(a) is not None


# ------------------------------------------------- docs/13 round-4 app fixes

def test_api_payload_carries_exactly_the_documented_keys():
    # docs/13 D-08: one place builds the extension's JSON; the JS harness reads
    # the same key list from tests/js/api_keys.json
    import json

    a = verdict_for(_fsd("swap", 1.4, mode="evidence", fsd_outside_fpr=0.38), THR)
    d = app_mod.api_payload(a)
    assert tuple(d) == app_mod.API_PAYLOAD_KEYS
    assert "pulse_measurable" in d and isinstance(d["warnings"], list)   # docs/13 C-09
    json.dumps(d)                                                         # serialisable as is
    fixture = Path(__file__).parent / "js" / "api_keys.json"
    assert json.loads(fixture.read_text()) == list(app_mod.API_PAYLOAD_KEYS), (
        "tests/js/api_keys.json is stale: regenerate it from app.API_PAYLOAD_KEYS")


def test_parse_region_accepts_only_a_sane_box():
    assert app_mod.parse_region('{"x0": 10, "y0": 20, "x1": 650, "y1": 380}') == (10, 20, 650, 380)
    for bad in (None, "", "nope", '{"x0": 1}', '{"x0": 0, "y0": 0, "x1": 5, "y1": 5}',
                '{"x0": "a", "y0": 0, "x1": 100, "y1": 100}'):
        assert app_mod.parse_region(bad) is None


def test_region_reaches_every_layer(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(app_mod, "_ai_video_layer", lambda a, p, ap, bar, region: seen.update(ai=region))
    monkeypatch.setattr(app_mod, "_face_swap_layer", lambda a, p, fp, bar, region: seen.update(fs=region))
    monkeypatch.setattr(app_mod, "_pulse_layer",
                        lambda a, p, c, m, thr, ms, bar, region: seen.update(pulse=region))
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"x")
    app_mod.analyze_video(p, region=(0, 0, 320, 240), layers=("ai_video", "face_swap", "pulse"))
    assert seen == {"ai": (0, 0, 320, 240), "fs": (0, 0, 320, 240), "pulse": (0, 0, 320, 240)}


def test_untracked_face_is_flagged_as_unstable(tmp_path, no_aigen):
    # docs/04 §8 / docs/13 A-10: below 70% tracking the pulse layer says so in warnings
    p = synth.write_pulse_video(tmp_path / "noface.mp4", fps=30.0, duration_s=11.0)
    a = app_mod.analyze_video(p, config={"method": "green"}, layers=("pulse",))
    assert any(w.startswith("W_UNSTABLE_TRACK") for w in a.warnings)
