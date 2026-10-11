"""Smoke tests — must stay green from Phase 0 onward."""


def test_package_imports():
    import trubeat

    assert trubeat.__version__ == "0.1.0"


def test_error_codes_stable():
    from trubeat import errors

    assert errors.UnreadableVideo.code == "E_UNREADABLE"
    assert errors.LowFps.code == "E_LOW_FPS"
    assert errors.ShortVideo.code == "E_SHORT_VIDEO"
    assert errors.NoFace.code == "E_NO_FACE"
