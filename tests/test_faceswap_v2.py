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


def test_shards_partition_the_videos_and_are_stable():
    ids = [f"ffpp__Deepfakes__{i:03d}_{(i * 7) % 1000:03d}" for i in range(600)]
    shards = [t.shard_of(v, 3) for v in ids]
    assert shards == [t.shard_of(v, 3) for v in ids]              # same answer in every session
    counts = np.bincount(shards, minlength=3)
    assert counts.sum() == len(ids) and counts.min() > 150         # every video in exactly one, roughly balanced
    assert t.parse_shard("2/3") == (2, 3) and t.parse_shard("") is None
    with pytest.raises(SystemExit):
        t.parse_shard("3/3")


def test_dfd_groups_join_both_actors_over_the_whole_manifest():
    stems = ["01__walk", "01_02__talk__A", "02_05__hug__B", "07__sit", "05__run"]
    g = t.dfd_groups(stems)
    assert g["01__walk"] == g["01_02__talk__A"] == g["02_05__hug__B"] == g["05__run"] == "dfd:01"
    assert g["07__sit"] == "dfd:07"
    # two shards grouped separately disagree; regrouping the concatenation restores one group per person
    rows = [{"video_id": f"dfd__x__{s}", "dataset": "dfd", "source_id": s, "group": ""} for s in stems]
    a, b = pd.DataFrame(rows[:2]), pd.DataFrame(rows[2:])
    for part in (a, b):
        part["group"] = part.source_id.map(t.dfd_groups(part.source_id.tolist()))
    assert b.set_index("source_id").group["05__run"] == "dfd:02"   # shard-local answer
    both = t.regroup_dfd(pd.concat([a, b], ignore_index=True))
    assert both.set_index("source_id").group.to_dict() == g


def test_dedupe_removes_only_df40_training_clips_that_nearly_duplicate_a_held_out_clip():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(4, 32))
    near = base[0] + 0.01 * rng.normal(size=32)          # a near-copy of held-out clip 0
    rows = [
        {"video_id": "ffpp__original__900", "dataset": "ffpp", "split": "test", "cls": base[0]},
        {"video_id": "cdf2__Celeb-real__id1_0001", "dataset": "cdf2", "split": "external", "cls": base[1]},
        {"video_id": "df40__simswap__train__001_002", "dataset": "df40", "split": "train", "cls": near},
        {"video_id": "df40__simswap__train__003_004", "dataset": "df40", "split": "train", "cls": base[2]},
        {"video_id": "ffpp__original__001", "dataset": "ffpp", "split": "train", "cls": near},   # not a DF40 clip
    ]
    st = pd.DataFrame(rows).assign(source="v0", judged=True)
    dropped, removed, summary = t.dedupe(st, 0.97)
    assert dropped == {"df40__simswap__train__001_002"}
    assert removed[0]["nearest"] == "ffpp__original__900" and removed[0]["cosine"] > 0.97
    assert summary["candidates"] == 2 and summary["references"] == 2 and summary["removed"] == 1


def test_ship_writes_the_claims_the_app_quotes(tmp_path):
    import argparse
    import hashlib
    import json

    import joblib

    from trubeat import claims

    bundle = {"model": None, "config": {}, "threshold": 1.0, "gates": _g(g1=True, g2=True, g2b=True, g3=False),
              "trained_on": {("ffpp", "original"): 10, ("df40", "df40:simswap"): 5}}
    src = tmp_path / "faceswap_v2.joblib"
    joblib.dump(bundle, src)
    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    frozen = t.REPO / "data" / "manifests" / "faceswap_gate4a.csv"
    g4_result = {"bundle_sha256": sha, "frozen_manifest": "faceswap_gate4a.csv",
                 "frozen_manifest_sha256": hashlib.sha256(frozen.read_bytes()).hexdigest(),
                 "k": 4, "n": 250, "rate": 0.016, "ci": [0.006, 0.04], "passed": True}
    g4 = tmp_path / "g4.json"
    r3 = tmp_path / "r3.json"
    r3.write_text(json.dumps({"rate": 0.383, "ci": [0.33, 0.439]}))
    out = tmp_path / "shipped.joblib"
    args = argparse.Namespace(bundle=str(src), gate4a=str(g4), round3_gate4=str(r3), out=str(out),
                              decision=str(tmp_path / "decision.json"))
    for wrong in ({"frozen_manifest": "some_subset.csv"}, {"frozen_manifest_sha256": "0" * 64}):
        g4.write_text(json.dumps({**g4_result, **wrong}))
        with pytest.raises(SystemExit):                               # only the frozen list itself counts
            t.cmd_ship(args)
    g4.write_text(json.dumps(g4_result))
    t.cmd_ship(args)
    b = joblib.load(out)
    assert b["mode"] == "evidence"                                    # gate 3 failed, 4a far below round 3
    c = claims.claims_for(b)
    assert c["trained_on"] == "FaceForensics++ and DF40"
    assert c["outside_fpr"]["k"] == 4 and c["outside_fpr"]["n"] == 250
    text = claims.caveat(None, None, c, fsd_mode=b["mode"])
    assert "trained on FaceForensics++ and DF40 is shown as evidence only" in text and "2% of real faces" in text


class _RangeNotSupported(Exception):
    pass


_RangeNotSupported.__name__ = "RangeNotSupported"   # the name remotezip's exception carries


def _fake_remotezip(monkeypatch, refuse_after: int):
    """A remotezip stand-in: reads succeed, then every read gets Drive's quota page."""
    import types

    calls = {"reads": 0}

    class RemoteZip:
        def __init__(self, url):
            pass

        def read(self, m):
            calls["reads"] += 1
            if calls["reads"] > refuse_after:
                raise _RangeNotSupported("The server doesn't support range requests")
            return b"clip"

        def infolist(self):
            raise _RangeNotSupported("The server doesn't support range requests")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "remotezip", types.SimpleNamespace(RemoteZip=RemoteZip))
    monkeypatch.setattr(t.time, "sleep", lambda s: pytest.fail("a refused read must not be retried"))
    return calls


def test_a_drive_quota_page_stops_the_fetch_instead_of_retrying_each_clip(tmp_path, monkeypatch):
    calls = _fake_remotezip(monkeypatch, refuse_after=2)
    members = [f"m/{i:03d}.mp4" for i in range(40)]
    with pytest.raises(t.DriveQuota):
        t.fetch_zip_members("https://drive.example/zip", members, tmp_path, threads=1)
    assert sorted(p.name for p in (tmp_path / "m").iterdir()) == ["000.mp4", "001.mp4"]   # kept for the rerun
    assert calls["reads"] == 3          # the first refusal ends it: no read per remaining clip
    with pytest.raises(t.DriveQuota):
        t.zip_index("https://drive.example/zip")


def test_the_laptop_runner_waits_out_a_drive_quota_and_stops_on_other_failures(monkeypatch):
    import run_fsv2_local as r

    logged, slept, ready, codes = [], [], iter([False, True, True]), iter([t.QUOTA_EXIT, 0])
    monkeypatch.setattr(r, "log", logged.append)              # never the real run.log
    monkeypatch.setattr(r.time, "sleep", slept.append)
    monkeypatch.setattr(r.t, "drive_ready", lambda url: next(ready))
    monkeypatch.setattr(r.subprocess, "run", lambda *a, **k: __import__("types").SimpleNamespace(returncode=next(codes)))
    r.fetch(["fetch"], "https://drive.example/zip", "DF40 x")
    assert slept == [r.DRIVE_WAIT_S, r.DRIVE_WAIT_S]          # probe refused, then fetch refused, then done
    assert sum("refuses downloads" in m for m in logged) == 2
    monkeypatch.setattr(r.t, "drive_ready", lambda url: True)
    monkeypatch.setattr(r.subprocess, "run", lambda *a, **k: __import__("types").SimpleNamespace(returncode=1))
    with pytest.raises(SystemExit, match="exit 1"):
        r.fetch(["fetch"], "https://drive.example/zip", "DF40 x")
