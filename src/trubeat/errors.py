"""Error taxonomy (see docs/04_system_architecture.md §8)."""


class TrueBeatError(Exception):
    """Base error with a stable machine-readable code."""

    code = "E_UNKNOWN"


class UnreadableVideo(TrueBeatError):
    code = "E_UNREADABLE"


class LowFps(TrueBeatError):
    code = "E_LOW_FPS"


class ShortVideo(TrueBeatError):
    code = "E_SHORT_VIDEO"


class NoFace(TrueBeatError):
    code = "E_NO_FACE"
