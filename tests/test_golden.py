"""Golden-set regression (docs/08 §3). TODO: record 5 videos + expected outputs in Phase 1.

Golden set: 2 real UBFC + 1 real FF++ + 2 fake FF++. Any drift beyond tolerance
fails the build. Re-record deliberately, never by accident.
"""

import pytest

pytestmark = pytest.mark.skip(reason="golden set not recorded yet — Phase 1")


def test_golden_hr_values():
    raise NotImplementedError


def test_golden_detector_votes():
    raise NotImplementedError
