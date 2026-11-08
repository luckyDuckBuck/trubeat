"""docs/13 A-07: every number a text quotes comes from the claims of the model that scored."""

from __future__ import annotations

import joblib
import pytest

from trubeat import app, claims
from trubeat.app import PulseAnalysis, verdict_for

MODELS = app.Path(app.__file__).parent / "models"


def _bundle(name: str) -> dict:
    b = joblib.load(MODELS / name)
    b["_path"] = str(MODELS / name)
    return b


def test_caveat_rendered_from_the_packaged_models_matches_the_shipped_text():
    c = claims.caveat(claims.claims_for(_bundle("aigen.joblib")), claims.claims_for(_bundle("detector.joblib")),
                      claims.claims_for(_bundle("faceswap.joblib")), "evidence")
    assert c == app.CAVEAT                    # drift guard: CAVEAT is the packaged models' rendering
    assert app.current_caveat() == app.CAVEAT


@pytest.mark.parametrize("rate,words", [(0.831, "about 4 in 5"), (0.547, "about half"), (0.522, "about half"),
                                        (0.24, "about a quarter"), (0.054, "about 5%"), (0.022, "about 2%"),
                                        (0.05, "about 5%"), (0.0, "none"), (0.61, "about 61%")])
def test_about_says_rates_in_words(rate, words):
    assert claims.about(rate) == words


AI_CLAIMS = {"layer": "ai_video", "families": 3, "min_seconds": 2, "source": "a report", "target_fpr": 0.1,
             "recall": [{"what": "tool A", "rate": 0.5}], "real_fpr": {"rate": 0.05}}
PULSE_CLAIMS = {"layer": "pulse", "recall": {"Deepfakes": 0.2}, "real_fpr": {"rate": 0.02}}


def test_claims_come_from_the_bundle_then_from_the_file_hash(tmp_path):
    own = dict(AI_CLAIMS)
    assert claims.claims_for({"claims": own}) is own
    assert claims.claims_for({"claims": {"layer": "ai_video", "target_fpr": 0.1}}) is None   # malformed: skipped
    packaged = _bundle("detector.joblib")
    assert claims.claims_for(packaged)["strict_threshold"] == -0.605
    copy = tmp_path / "detector.joblib"
    copy.write_bytes((MODELS / "detector.joblib").read_bytes())
    assert claims.claims_for({"_path": str(copy)})["layer"] == "pulse"     # same bytes, same claims
    copy.write_bytes(copy.read_bytes() + b"\0")
    assert claims.claims_for({"_path": str(copy)}) is None                 # a different file: nothing to quote
    assert claims.claims_for(None) is None


def test_an_override_without_claims_is_refused_but_an_explicit_path_is_used(tmp_path, monkeypatch):
    other = tmp_path / "models" / "detector.joblib"
    other.parent.mkdir()
    joblib.dump({"model": "stub", "config": {}, "threshold": 0.0}, other)
    monkeypatch.setattr(app, "MODEL_SEARCH_PATHS", [other, MODELS / "detector.joblib"])
    picked = app.load_detector(None)
    assert app.Path(picked["_path"]) == MODELS / "detector.joblib"       # the silent override is refused
    assert app.load_detector(other)["model"] == "stub"                   # an explicit choice is honoured
    joblib.dump({"model": "stub", "config": {}, "threshold": 0.0, "claims": {"layer": "pulse"}}, other)
    assert app.load_detector(None)["model"] != "stub"                    # malformed claims count as none
    joblib.dump({"model": "stub", "config": {}, "threshold": 0.0, "claims": PULSE_CLAIMS}, other)
    assert app.load_detector(None)["model"] == "stub"                    # an override with claims is fine


def _ai_flagged() -> PulseAnalysis:
    return PulseAnalysis(verdict="inconclusive", headline="", detail="", ai_label="ai", ai_score=2.0,
                         ai_threshold=1.0, ai_available=True)


def test_verdict_texts_quote_the_claims_of_the_analysis():
    a = verdict_for(_ai_flagged())                                     # built by hand: the packaged models' numbers
    assert "set so that about 5% of real clips cross it" in a.detail
    b = _ai_flagged()
    b.claims = {"ai_video": {"target_fpr": 0.01}}
    assert "set so that about 1% of real clips cross it" in verdict_for(b).detail
    c = _ai_flagged()
    c.claims = {"ai_video": None}
    assert "set so that" not in verdict_for(c).detail                  # no claims: no number


def test_pulse_check_text_and_threshold_come_from_its_claims():
    a = PulseAnalysis(verdict="inconclusive", headline="", detail="", detector_score=-0.9)
    a = verdict_for(a)
    assert a.decided_by == "pulse"
    assert "catches about a quarter of Deepfakes-style swaps and flags about 2% of real face videos" in a.detail
    b = PulseAnalysis(verdict="inconclusive", headline="", detail="", detector_score=-0.9)
    b.claims = {"pulse": None}
    assert "records no measured accuracy" in verdict_for(b).detail
    assert app.model_claims(layers=("pulse",))["pulse"]["strict_threshold"] == -0.605


def test_the_api_caveat_belongs_to_the_models_used(tmp_path):
    a = app.analyze_video(tmp_path / "missing.mp4")                     # unreadable: still carries the caveat
    assert a.caveat == app.CAVEAT
    assert app.api_payload(a)["caveat"] == app.CAVEAT
    no_claims = claims.caveat(None, None, None)
    assert "records no measured accuracy" in no_claims and "%" not in no_claims


def test_an_analysis_quotes_the_bundles_it_scored_with(tmp_path, monkeypatch):
    # review 2026-11-08: the claims were re-read from disk while the layers scored
    # with cached bundles, so a bundle replaced under a running server was quoted
    # for a model that did not score
    scored = {"claims": dict(AI_CLAIMS, families=7)}
    monkeypatch.setattr(app, "_ai_video_layer", lambda a, p, ap, bar, region: scored)
    monkeypatch.setattr(app, "_face_swap_layer", lambda a, p, fp, bar, region: None)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    a = app.analyze_video(clip, layers=("ai_video", "face_swap"))
    assert a.claims["ai_video"]["families"] == 7 and "trained on 7 generator families" in a.caveat
    assert a.claims["face_swap"] is None and a.claims["fsd_mode"] == "evidence"


def test_a_refused_override_is_reported_with_the_result(tmp_path, monkeypatch):
    from trubeat import aigen
    from trubeat.bundles import WARNINGS_KEY

    other = tmp_path / "models" / "detector.joblib"
    other.parent.mkdir()
    joblib.dump({"model": "stub", "config": {}, "threshold": 0.0}, other)
    monkeypatch.setattr(app, "MODEL_SEARCH_PATHS", [other, MODELS / "detector.joblib"])
    picked = app.load_detector(None)
    assert app.Path(picked["_path"]) == MODELS / "detector.joblib"
    assert any(w.startswith("W_OVERRIDE_REFUSED") and str(other) in w for w in picked[WARNINGS_KEY])
    # and a layer copies its bundle's notes into the analysis, even when it then fails
    noted = {WARNINGS_KEY: ["W_OVERRIDE_REFUSED: test"], "config": {}}
    monkeypatch.setattr(app, "load_aigen", lambda path=None: (noted, object()))
    monkeypatch.setattr(aigen, "analyze", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop")))
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    a = app.analyze_video(clip, layers=("ai_video",))
    assert "W_OVERRIDE_REFUSED: test" in a.warnings and "W_AIGEN_FAILED" in a.warnings


def test_malformed_claims_never_break_an_analysis(tmp_path, monkeypatch):
    broken = {"claims": {"layer": "ai_video", "recall": "no"}}       # what a damaged bundle could carry
    monkeypatch.setattr(app, "_ai_video_layer", lambda a, p, ap, bar, region: broken)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    a = app.analyze_video(clip, layers=("ai_video",))
    assert a.claims["ai_video"] is None and "records no measured accuracy" in a.caveat
