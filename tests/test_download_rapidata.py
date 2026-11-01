"""scripts/download_rapidata.py: filename -> (generator, prompt, take).

De-duplication across repos and the prompt grouping (no prompt straddles
a cross-validation fold) both rest on this parse, so every layout seen in
the Rapidata repos is pinned here. No network.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import download_rapidata as dr


@pytest.mark.parametrize(("name", "want"), [
    ("videos/0062_ray2_1.mp4", ("ray2", 62, "1")),                     # veo3 repo layout
    ("Videos/ray2/ray2_0062_1.mp4", ("ray2", 62, "1")),                # seedance / mochi layout
    ("Videos/ray2_0062_1.mp4", ("ray2", 62, "1")),                     # marey layout
    ("videos/veo31_49_0.mp4", ("veo3.1", 49, "0")),
    ("Videos/seedance-1-pro-24-7-25/seedance-1-pro-24-7-25_99_0.mp4", ("seedance-1-pro", 99, "0")),
    ("Videos/mochi-1-24-7-25/mochi-1-24-7-25_0003_0.mp4", ("mochi-1", 3, "0")),
    ("videos/0006_alpha_3945567110.mp4", ("gen3-alpha", 6, "3945567110")),
    ("data/videos/hunyuan/0037_hunyuan_421.mp4", ("hunyuan", 37, "421")),
    ("videos/0000_pika2.2_0.mp4", ("pika2.2", 0, "0")),
])
def test_parse_layouts(name, want):
    assert dr.parse(name) == want


def test_same_clip_in_two_layouts_has_one_key():
    assert dr.parse("videos/0062_ray2_1.mp4") == dr.parse("Videos/ray2/ray2_0062_1.mp4")


@pytest.mark.parametrize("name", ["Videos/temp.txt", "videos/readme.mp4", "Videos/x_y.mp4"])
def test_unparseable_names_are_skipped(name):
    assert dr.parse(name) is None


def test_every_rapidata_generator_has_a_holdout_family():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import train_aigen as ta

    for gen, fam in (("veo3.1", "veo"), ("veo2", "veo"), ("ray2", "luma"), ("gen3-alpha", "runway"),
                     ("pika2.2", "pika"), ("hunyuan", "hunyuanvideo"), ("kling-v2.1-master", "kling"),
                     ("seedance-1-pro", "seedance"), ("mochi-1", "mochi"), ("wan2.1", "wan")):
        assert ta.base_family(gen) == fam
