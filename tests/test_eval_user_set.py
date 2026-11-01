"""scripts/eval_user_set.py reads the app's real result object (docs/11 F-1).

The backend is stubbed (no model, no video decode), but the stub returns a
genuine ``PulseAnalysis``, so a renamed field breaks this test instead of
the user's evaluation run.
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import eval_user_set

from trubeat.app import PulseAnalysis


def _fake_analyze(path):
    ai = "ai_" in Path(path).name
    return PulseAnalysis(
        verdict="fake" if ai else "real",
        headline="Likely AI-generated" if ai else "No sign of AI generation found",
        detail="",
        duration_s=8.0,
        ai_label="ai" if ai else "real",
        ai_real_percentile=0.99 if ai else 0.30,
        decided_by="ai_video",
    )


def test_folder_layout_report(tmp_path, monkeypatch):
    for label, names in (("ai", ["ai_1.mp4", "ai_2.mp4"]), ("real", ["r1.mp4", "notes.txt"])):
        (tmp_path / label).mkdir()
        for n in names:
            (tmp_path / label / n).write_bytes(b"")
    out = tmp_path / "report.md"
    monkeypatch.setattr(eval_user_set, "analyze_video", _fake_analyze)
    monkeypatch.setattr(sys, "argv", ["eval_user_set.py", "--dir", str(tmp_path), "--out", str(out)])
    eval_user_set.main()
    text = out.read_text()
    assert "3 clips (2 AI, 1 real)" in text  # notes.txt is not a video
    assert "| AI clips caught | 2/2 = 100%" in text
    assert "| real clips wrongly flagged | 0/1 = 0%" in text
    assert "| ai_1.mp4 | ai | fake | ai_video | 99% |" in text


def test_csv_rejects_bad_labels(tmp_path, monkeypatch):
    csv = tmp_path / "l.csv"
    csv.write_text("path,label\na.mp4,deepfake\n")
    monkeypatch.setattr(sys, "argv", ["eval_user_set.py", "--csv", str(csv)])
    with pytest.raises(SystemExit, match="labels must be ai"):
        eval_user_set.main()
