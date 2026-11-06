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


class GraphicsOnly(TrueBeatError):
    """Every candidate segment is a title card or graphic (docs/12 item A)."""

    code = "E_GRAPHICS_ONLY"


class ModelMissing(TrueBeatError):
    """A model the analysis needs is not in the local cache. Analysis never
    downloads (AGENTS.md §7); `trubeat fetch-models` does, on request."""

    code = "E_MODEL_MISSING"
