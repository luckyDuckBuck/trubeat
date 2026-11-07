"""Two-pass face finder (docs/13 §3.1): geometry and pass order, with the
detector and the Landmarker stubbed (real faces are measured by
scripts/check_facefind.py on local data, never committed)."""

import numpy as np
import pytest

from trubeat import facefind as ff
from trubeat import faceswap as fs


def test_scale_detections_maps_back_filters_and_sorts():
    rows = np.zeros((3, 15), np.float32)
    rows[0, :4], rows[0, 14] = (10, 20, 30, 40), 0.9     # side 40 at k=0.5 -> 80 native
    rows[1, :4], rows[1, 14] = (100, 100, 60, 50), 0.95  # the larger one
    rows[2, :4], rows[2, 14] = (5, 5, 90, 90), 0.4       # under the score floor
    d = ff.scale_detections(rows, 0.5, 0.6)
    assert [round(x.side) for x in d] == [120, 80]
    assert (d[0].x, d[0].y) == (200.0, 200.0) and d[1].score == pytest.approx(0.9)
    assert ff.scale_detections(None, 1.0, 0.6) == []


def test_crop_to_native_inverts_cut_square():
    rgb = np.zeros((300, 400, 3), np.uint8)
    rgb[150, 220] = 255                       # one bright pixel
    cx, cy, side, size = 210.0, 140.0, 61.0, 244
    crop = fs.cut_square(rgb, cx, cy, side, size)
    py, px = np.unravel_index(np.argmax(crop[:, :, 0]), crop.shape[:2])
    x, y = ff.crop_to_native(np.array([px + 0.5]), np.array([py + 0.5]), ff.crop_window(cx, cy, side), size)
    assert abs(x[0] - 220.5) < 1.0 and abs(y[0] - 150.5) < 1.0


class _Stub(ff.TwoPassFinder):
    """Detector + Landmarker replaced: ``dets`` are native boxes; ``crop_ok``
    says whether the Landmarker confirms a face on a crop; ``whole`` is what
    the whole-frame Landmarker returns (round-3 fallback)."""

    def __init__(self, dets, crop_ok=True, whole=()):
        super().__init__(fs.merged_config({"finder": "twopass"}))
        self.dets, self.crop_ok, self.whole, self.crops = dets, crop_ok, list(whole), []
        self._lm = object()
        self.detect_calls = 0

    def detect(self, rgb):
        self.detect_calls += 1
        return list(self.dets)

    def _landmark_sets(self, img):
        if img.shape[0] == img.shape[1]:          # a square crop from confirm(); frames are 16:9
            self.crops.append(img.shape[0])
            if not self.crop_ok:
                return []
            c, r = img.shape[0] / 2.0, img.shape[0] / 4.0     # a face filling the crop's middle half
            xs = np.array([c - r, c + r, c], np.float32)
            ys = np.array([c - r, c + r, c + r / 2], np.float32)
            return [(xs, ys)]
        return list(self.whole)


def _rgb():
    return np.zeros((1080, 1920, 3), np.uint8)


def test_small_face_is_confirmed_on_an_upscaled_crop():
    f = _Stub([ff.Detection(900, 500, 50, 60, 0.9)])
    box = f.largest_face(_rgb())
    assert box is not None and f.last_source == "yunet" and f.last_seen == 1
    assert f.crops == [256]                     # 2 x 60 px -> upscaled to the 256 px minimum
    cx, cy, side = box
    assert abs(cx - 925) < 3 and abs(cy - 530) < 3
    assert side == pytest.approx(60 * 2 / 2 * 1.3, rel=0.05)  # landmark box = half the 120 px crop, x1.3


def test_a_face_the_round3_pass_finds_is_returned_unchanged():
    # docs/13 §6.1 rules 2-3: where the whole-frame Landmarker finds the face, nothing changes
    big = (np.array([100, 500], np.float32), np.array([100, 500], np.float32))
    f = _Stub([ff.Detection(900, 500, 60, 60, 0.9)], whole=[big])
    box = f.largest_face(_rgb())
    assert f.last_source == "landmarker" and box == pytest.approx((300, 300, 400 * 1.3))
    assert f.detect_calls == 0                    # YuNet only runs when pass 1 finds nothing


def test_faces_below_min_face_px_are_ignored_in_both_passes():
    tiny = (np.array([100, 120], np.float32), np.array([100, 120], np.float32))  # 20 px
    f = _Stub([ff.Detection(10, 10, 20, 20, 0.9)], whole=[tiny])
    assert f.largest_face(_rgb()) is None and f.detect_calls == 1


def test_unconfirmed_detections_are_not_faces():
    f = _Stub([ff.Detection(900, 500, 80, 80, 0.9)], crop_ok=False, whole=[])
    assert f.largest_face(_rgb()) is None and f.last_source == ""


def test_largest_confirmed_face_wins():
    f = _Stub([ff.Detection(100, 100, 300, 300, 0.9), ff.Detection(900, 500, 60, 60, 0.9)])
    cx, _, _ = f.largest_face(_rgb())
    assert abs(cx - 250) < 3 and f.last_seen == 2


def test_faceswap_finder_is_selected_by_config():
    assert fs.FaceFinder({})._two is None                      # round-3 bundles keep the old finder
    assert isinstance(fs.FaceFinder({"finder": "twopass"})._two, ff.TwoPassFinder)
    with pytest.raises(ValueError):
        fs.merged_config({"finder": "nope"})


def test_config_hash_covers_the_finder_and_the_target():
    base = fs.config_hash({})
    assert fs.config_hash({"finder": "twopass"}) != base
    assert fs.config_hash({"target_fpr": 0.05}) != base
    assert fs.config_hash({"detect_score": 0.7}) != base
    assert fs.config_hash({"torch_threads": 1}) == base


def test_yunet_model_ships_with_the_package():
    p = ff.resolve_yunet_path()
    assert p.is_file() and p.stat().st_size > 100_000
