"""Tests for the TrueBeat browser extension (extension/).

The extension's JavaScript is exercised by tests/js/extension_test.mjs,
which loads the *shipped* files and drives the real code paths (the Gradio
API client against a live backend, the verdict mapping, the black-frame
guard, and a static privacy audit).

This wrapper runs that harness and also checks the manifest and the file
layout. It skips cleanly when Node is unavailable.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "extension"
HARNESS = ROOT / "tests" / "js" / "extension_test.mjs"

node = shutil.which("node")

needs_node = pytest.mark.skipif(node is None, reason="node is not installed")


def test_extension_layout():
    expected = [
        "manifest.json",
        "background.js",
        "offscreen.html",
        "offscreen.js",
        "popup.html",
        "popup.js",
        "options.html",
        "options.js",
        "overlay.js",
        "README.md",
    ]
    for name in expected:
        assert (EXT / name).is_file(), f"missing {name}"
    for size in (16, 32, 48, 128):
        assert (EXT / "icons" / f"icon{size}.png").is_file(), f"missing icon{size}"


def test_manifest_is_mv3_and_local_only():
    m = json.loads((EXT / "manifest.json").read_text())
    assert m["manifest_version"] == 3
    assert re.fullmatch(r"\d+\.\d+\.\d+", m["version"]), m["version"]
    assert m["background"]["service_worker"] == "background.js"
    assert m["action"]["default_popup"] == "popup.html"
    assert m["options_page"] == "options.html"
    # Default permissions stay on localhost; anything else is optional.
    assert all(
        h in ("http://127.0.0.1/*", "http://localhost/*")
        for h in m["host_permissions"]
    )
    assert m["optional_host_permissions"]
    # tabCapture + offscreen are what the capture flow needs; activeTab +
    # scripting inject the panel into the clicked tab.
    for perm in ("tabCapture", "offscreen", "storage", "activeTab", "scripting"):
        assert perm in m["permissions"]


def test_no_content_script_on_every_page():
    """docs/13 C-08: the panel is injected on click under activeTab; a static
    <all_urls> content script (and a web-accessible copy of it) asked for
    "read and change all your data on all websites" and let pages detect
    the extension."""
    m = json.loads((EXT / "manifest.json").read_text())
    assert "content_scripts" not in m
    assert "web_accessible_resources" not in m
    assert "<all_urls>" not in json.dumps(m)
    assert "overlay.js" in (EXT / "background.js").read_text()  # injected by the worker


_SCRIPT_TAG = re.compile(r"<script\b([^>]*)>(.*?)</script\s*>", re.IGNORECASE | re.DOTALL)
_SRC_ATTR = re.compile(r"""\bsrc\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))""", re.IGNORECASE)


def _script_problems(html: str, root: Path) -> tuple[int, list[str]]:
    """(number of <script> elements, problems): inline code, a remote or
    non-file source, a missing file, or a <script> left unclosed."""
    problems = []
    tags = _SCRIPT_TAG.findall(html)
    for attrs, body in tags:
        m = _SRC_ATTR.search(attrs)
        if not m:
            problems.append("inline <script> (MV3's CSP blocks it)")
            continue
        src = next(g for g in m.groups() if g is not None)
        if body.strip():
            problems.append(f"<script src={src!r}> also carries inline code")
        if re.match(r"^(?:[a-z][a-z0-9+.-]*:|//)", src, re.IGNORECASE):
            problems.append(f"remote or non-file script {src}")
        elif not (root / src).is_file():
            problems.append(f"missing script {src}")
    if len(re.findall(r"<script\b", html, re.IGNORECASE)) != len(tags):
        problems.append("a <script> tag that is not closed by </script>")
    return len(tags), problems


def test_no_remote_code_in_html_pages():
    """MV3's CSP blocks inline scripts; every page must load a local file."""
    for name in ("popup.html", "options.html", "offscreen.html"):
        n, problems = _script_problems((EXT / name).read_text(), EXT)
        assert n >= 1, f"{name} loads no script at all"
        assert not problems, f"{name}: {problems}"


@pytest.mark.parametrize(
    ("html", "problem"),
    [
        ("<script>alert(1)</script>", "inline <script>"),
        ('<script src="https://cdn.example.com/x.js"></script>', "remote"),
        ('<script src="//cdn.example.com/x.js"></script>', "remote"),
        ('<script src="data:text/javascript,alert(1)"></script>', "remote"),
        ('<script src="missing.js"></script>', "missing script"),
        ('<script src="ok.js">alert(1)</script>', "inline code"),
        ('<script src="ok.js">', "not closed"),
    ],
)
def test_script_audit_can_fail(tmp_path, html, problem):
    """docs/13 C-20: the audit used to skip every <script src=...> tag
    (it only looked at tags starting with '>'), so it could never fail."""
    (tmp_path / "ok.js").write_text("")
    assert _script_problems('<script src="ok.js"></script>', tmp_path) == (1, [])
    _, problems = _script_problems(html, tmp_path)
    assert any(problem in p for p in problems), problems


@needs_node
def test_extension_javascript_suite():
    """The extension's own test harness (API client, overlay, privacy audit)."""
    proc = subprocess.run(
        [node, str(HARNESS)],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=str(ROOT),
        check=False,
    )
    assert proc.returncode == 0, (
        "extension test harness failed:\n" + proc.stdout[-4000:] + proc.stderr[-2000:]
    )
    assert "0 failed" in proc.stdout, proc.stdout[-2000:]


@needs_node
def test_extension_javascript_parses():
    """Every shipped .js file must be syntactically valid."""
    for js in sorted(EXT.glob("*.js")):
        proc = subprocess.run(
            [node, "--check", str(js)], capture_output=True, text=True, check=False
        )
        assert proc.returncode == 0, f"{js.name}: {proc.stderr}"
