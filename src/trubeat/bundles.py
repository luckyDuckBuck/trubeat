"""Loading the trained model bundles (docs/13 D-05).

A bundle is a joblib pickle of scikit-learn models. Read by another
scikit-learn major.minor than the one that saved it, a model can load and
still score differently, so loading records a ``W_SKLEARN_VERSION`` warning
(AGENTS.md §5: never silently) and the analysis using the bundle reports it.

The saved version comes from the bundle's ``sklearn_version`` key (bundles
written since 2026-11-07) and from the versions scikit-learn stores inside
each pickled estimator, which it reports while unpickling when they differ
from the installed one (bundles written before then carry only those).
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any

# Runtime-only key, like "_path": the load warnings an analysis reports.
WARNINGS_KEY = "_warnings"


def _major_minor(version: str) -> tuple[int, int] | None:
    m = re.match(r"(\d+)\.(\d+)", str(version))
    return (int(m.group(1)), int(m.group(2))) if m else None


def sklearn_version_warnings(path: str | Path, saved: set[str], installed: str) -> list[str]:
    """``W_SKLEARN_VERSION`` when a saved scikit-learn version differs from
    the installed one in major.minor (an unparseable one always differs)."""
    have = _major_minor(installed)
    other = sorted(v for v in saved if _major_minor(v) is None or _major_minor(v) != have)
    if not other:
        return []
    return [(f"W_SKLEARN_VERSION: {Path(path).name} was saved with scikit-learn {', '.join(other)} and is "
             f"read with {installed}; its scores may differ from the evaluated ones")]


def load_checked(path: str | Path) -> tuple[Any, list[str]]:
    """``joblib.load(path)`` and the W_* warnings loading it raised.

    scikit-learn's own InconsistentVersionWarning is passed on unchanged;
    it is only read for the version it names.
    """
    import joblib
    import sklearn
    from sklearn.exceptions import InconsistentVersionWarning

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", InconsistentVersionWarning)
        obj = joblib.load(path)
    saved: set[str] = set()
    for w in caught:
        if issubclass(w.category, InconsistentVersionWarning):
            saved.add(str(w.message.original_sklearn_version))
        warnings.warn_explicit(w.message, w.category, w.filename, w.lineno, source=w.source)
    if isinstance(obj, dict) and obj.get("sklearn_version"):
        saved.add(str(obj["sklearn_version"]))
    return obj, sklearn_version_warnings(path, saved, sklearn.__version__)
