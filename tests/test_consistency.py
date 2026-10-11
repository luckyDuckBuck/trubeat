"""Tests for trubeat.consistency — synthetic-signal property tests."""

import numpy as np
import pytest

import synth
from trubeat.consistency import coherence_features, lag0_correlation

FPS = 30.0
# 30 s windows → ~7 Welch segments at nperseg=256, so the coherence estimate
# can actually discriminate (10 s windows read ~1.0 for everything — see the
# caveat in coherence_features' docstring).
W = 900


def _coherent(n_roi=3, bpm=72.0):
    """Identical pure sines per ROI: the idealized coherent face."""
    return np.stack([synth.pulse_wave(W, FPS, bpm) for _ in range(n_roi)], axis=1)


def _independent():
    """Different-rate sines + independent noise per ROI: incoherent face."""
    return np.stack(
        [
            synth.pulse_wave(W, FPS, b, noise_std=0.3, seed=i)
            for i, b in enumerate((72.0, 95.0, 130.0))
        ],
        axis=1,
    )


def test_coherence_features_keys():
    cf = coherence_features(_coherent(), FPS)
    assert set(cf) == {"coh_mean", "coh_std", "coh_min", "coh_max", "pairs"}
    assert set(cf["pairs"]) == {(0, 1), (0, 2), (1, 2)}


def test_coherent_sines_high_coherence():
    cf = coherence_features(_coherent(), FPS)
    assert cf["coh_mean"] > 0.9
    assert cf["coh_min"] > 0.9


def test_independent_sines_low_coherence():
    cf = coherence_features(_independent(), FPS)
    assert cf["coh_mean"] < 0.3


def test_lag0_identical_is_one():
    lc = lag0_correlation(_coherent())
    assert lc["lag0_mean"] == pytest.approx(1.0)
    assert lc["lag0_min"] == pytest.approx(1.0)


def test_lag0_independent_near_zero():
    lc = lag0_correlation(_independent())
    assert abs(lc["lag0_mean"]) < 0.2


def test_constant_signals_no_nan():
    ppg = np.full((W, 3), 5.0)
    cf = coherence_features(ppg, FPS)
    lc = lag0_correlation(ppg)
    for d in (cf, lc):
        for key, val in d.items():
            if key == "pairs":
                continue
            assert np.isfinite(val), key
    # No measurable coherence / correlation in a flat signal.
    assert cf["coh_mean"] == 0.0
    assert lc["lag0_mean"] == 0.0


def test_degenerate_shapes():
    assert coherence_features(np.zeros((10, 1)), FPS)["pairs"] == {}
    assert coherence_features(np.zeros((10, 1)), FPS)["coh_mean"] == 0.0
    assert lag0_correlation(np.zeros((10, 1)))["lag0_mean"] == 0.0
