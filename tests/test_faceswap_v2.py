"""docs/13 round-4 training helpers: identity grouping, DF40 names, the
leakage check, sample weights, folds, the outcome rules."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_faceswap_v2 as t


def test_df40_names_map_to_ffpp_ids():
    assert t.df40_pair_ids("001_870") == ["001", "870"]
    assert t.df40_pair_ids("970_4mJ1aHFPS1M_50009") == ["970"]
    assert t.df40_pair_ids("id19_id25_0008") == []


def test_union_find_joins_identities():
    uf = t.UnionFind()
    uf.union("dfd:01", "dfd:02")
    uf.union("dfd:02", "dfd:07")
    assert uf.find("dfd:07") == uf.find("dfd:01") and uf.find("dfd:03") == "dfd:03"


def _df(rows):
    return pd.DataFrame(rows, columns=["video_id", "dataset", "family", "label", "split", "group"])


def test_leakage_check_catches_a_group_in_two_splits_and_in_an_external_set():
    ok = _df([("a", "ffpp", "original", "real", "train", "ffpp:000_003"),
              ("b", "df40", "df40:x", "fake", "train", "ffpp:000_003"),
              ("c", "dfd", "DFD", "fake", "external", "dfd:01")])
    t.check_manifest_v2(ok)
    with pytest.raises(ValueError):
        t.check_manifest_v2(pd.concat([ok, _df([("d", "df40", "df40:x", "fake", "test", "ffpp:000_003")])]))
    with pytest.raises(ValueError):
        t.check_manifest_v2(pd.concat([ok, _df([("e", "cdf2", "CDF", "fake", "external", "ffpp:000_003")])]))


def test_balance_weights_equalise_every_label_source_cell():
    st = pd.DataFrame({"label": ["real"] * 6 + ["fake"] * 2 + ["real"] * 2,
                       "source": ["v0"] * 6 + ["v0"] * 2 + ["v1"] * 2})
    w = t.balance_weights(st)
    cells = st.assign(w=w).groupby(["label", "source"]).w.sum()
    assert np.allclose(cells.to_numpy(), cells.iloc[0])


def test_group_folds_keep_groups_whole_and_change_with_the_seed():
    g = pd.Series([f"g{i // 3}" for i in range(60)])
    f1, f2 = t.group_folds(g, 5, 7), t.group_folds(g, 5, 11)
    for idx in g.groupby(g).groups.values():
        assert len(set(f1[list(idx)])) == 1
    assert (f1 != f2).any()


def test_weighted_quantile_weights_each_source_equally():
    v = np.array([0.0] * 90 + [10.0] * 10)
    src = pd.DataFrame({"source": ["a"] * 90 + ["b"] * 10})
    q = t.weighted_quantile(v, t.source_weights(src), 0.6)
    assert q > 5.0          # unweighted, the 60th percentile would be 0


def _g(**ok):
    names = {"g1": "1. FF++ test AUC >= 0.85 on Deepfakes", "g1b": "1b. DF40", "g2": "2. FF++ reals",
             "g2b": "2b. FF++ test reals after v1", "g3": "3. DFD", "g3b": "3b. Celeb-DF", "g4b": "4b. Celeb-DF outside"}
    return [(names[k], v, "") for k, v in ok.items()]


def test_ship_rules():
    good = {"passed": True, "rate": 0.02, "ci": [0.01, 0.04]}
    bad = {"passed": False, "rate": 0.10, "ci": [0.07, 0.14]}
    r3 = {"rate": 0.383, "ci": [0.33, 0.439]}
    gates = _g(g1=True, g1b=None, g2=True, g2b=True, g3=True, g3b=True, g4b=True)
    assert t.ship_decision(gates, good, r3)[0] == "decisive"       # 1b not applicable is fine
    assert t.ship_decision(gates, bad, r3)[0] == "evidence"        # 4a fails, but far below round 3
    close = {"passed": False, "rate": 0.30, "ci": [0.25, 0.35]}
    assert t.ship_decision(gates, close, r3)[0] == "keep-round3"   # overlapping intervals
    assert t.ship_decision(_g(g1=True, g2=True, g2b=False, g3=True), good, r3)[0] == "stage2"
    assert t.ship_decision(_g(g1=None, g2=True, g2b=True, g3=True), good, r3)[0] == "stage2"  # missing core gate
