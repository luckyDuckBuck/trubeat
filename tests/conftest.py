"""Shared pytest setup: the skip policy of CI's full job (docs/13 D-02).

``pytest --fail-on-skip`` fails the run when any test skips unless its node
id matches an entry of ALLOWED_SKIPS. The full job installs every extra,
Node, ffmpeg and the model weights, so there a skip means a layer silently
went untested. Without the option (the core job, a laptop) skips only show
up under ``-rs``.
"""

from __future__ import annotations

from fnmatch import fnmatchcase

import pytest

# node id (fnmatch pattern) -> why it may skip even in the full job. Empty:
# with every extra installed, nothing may skip.
ALLOWED_SKIPS: dict[str, str] = {}

_skipped: list[tuple[str, str]] = []


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--fail-on-skip", action="store_true", default=False,
                     help="fail when a test skips and is not in tests/conftest.py ALLOWED_SKIPS")


def _record(report: pytest.CollectReport | pytest.TestReport) -> None:
    if report.skipped and not hasattr(report, "wasxfail"):
        reason = report.longrepr[2] if isinstance(report.longrepr, tuple) else str(report.longrepr)
        _skipped.append((report.nodeid, reason))


def pytest_collectreport(report: pytest.CollectReport) -> None:
    _record(report)  # module-level skips (importorskip, pytestmark)


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    _record(report)


def _unexpected() -> list[tuple[str, str]]:
    return [(n, r) for n, r in _skipped if not any(fnmatchcase(n, pat) for pat in ALLOWED_SKIPS)]


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if session.config.getoption("--fail-on-skip") and _unexpected() and exitstatus == pytest.ExitCode.OK:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_terminal_summary(terminalreporter, exitstatus: int, config: pytest.Config) -> None:
    if not config.getoption("--fail-on-skip"):
        return
    bad = _unexpected()
    if bad:
        terminalreporter.section("skips not in ALLOWED_SKIPS (--fail-on-skip)", red=True)
        for nodeid, reason in bad:
            terminalreporter.line(f"{nodeid}: {reason}")
