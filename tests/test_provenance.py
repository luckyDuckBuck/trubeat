"""Tests for trubeat.provenance (C2PA Content Credentials, offline)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synth

from trubeat import provenance

IPTC = "http://cv.iptc.org/newscodes/digitalsourcetype/"


def _store(dst, agent=None, state="Valid", nested=False):
    action = {"action": "c2pa.created", "digitalSourceType": IPTC + dst}
    if agent:
        action["softwareAgent"] = agent
    manifest = {
        "claim_generator_info": [{"name": "SomeEditor", "version": "1.0"}],
        "assertions": [{"label": "c2pa.actions.v2", "data": {"actions": [action]}}],
    }
    if nested:  # the AI action lives in an ingredient's manifest
        outer = {
            "claim_generator_info": [{"name": "Editor"}],
            "assertions": [{"label": "c2pa.actions.v2", "data": {"actions": [
                {"action": "c2pa.edited", "digitalSourceType": IPTC + "humanEdits"}]}}],
        }
        return {"active_manifest": "urn:outer", "validation_state": state,
                "manifests": {"urn:outer": outer, "urn:inner": manifest}}
    return {"active_manifest": "urn:m", "validation_state": state, "manifests": {"urn:m": manifest}}


def test_trained_algorithmic_media_is_ai_declared():
    r = provenance.interpret_manifest_store(
        _store("trainedAlgorithmicMedia", agent={"name": "GenModel", "version": "2"})
    )
    assert r.has_c2pa and r.ai_declared
    assert r.generator == "GenModel 2"
    assert r.warnings == []


def test_composite_with_ai_counts_and_ingredient_history_is_scanned():
    r = provenance.interpret_manifest_store(
        _store("compositeWithTrainedAlgorithmicMedia", nested=True)
    )
    assert r.ai_declared
    assert set(r.source_types) == {"humanEdits", "compositeWithTrainedAlgorithmicMedia"}


def test_camera_capture_is_not_ai():
    r = provenance.interpret_manifest_store(_store("digitalCapture"))
    assert r.has_c2pa and not r.ai_declared
    assert r.generator == "SomeEditor 1.0"  # falls back to the claim generator


def test_unverified_ai_manifest_carries_a_warning():
    r = provenance.interpret_manifest_store(_store("trainedAlgorithmicMedia", state="Invalid"))
    assert r.ai_declared
    assert any("W_C2PA_UNVERIFIED" in w for w in r.warnings)


def test_plain_video_has_no_c2pa(tmp_path):
    # the reader itself must say "none": without the library the same
    # result came back with W_NO_C2PA_LIB and the test proved nothing (D-02)
    pytest.importorskip("c2pa")
    p = synth.write_pulse_video(tmp_path / "v.mp4", fps=30.0, duration_s=2.0)
    r = provenance.read_provenance(p)
    assert not r.has_c2pa and not r.ai_declared
    assert r.warnings == []


def test_missing_file_never_raises(tmp_path):
    r = provenance.read_provenance(tmp_path / "nope.mp4")
    assert not r.has_c2pa
    assert r.warnings and r.warnings[0].startswith("E_UNREADABLE")
