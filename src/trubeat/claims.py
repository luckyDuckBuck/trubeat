"""The measured numbers a text may quote, tied to the model file that scores (docs/13 A-07).

Before round 4 the caveat and the verdict texts quoted numbers ("about 5%",
"a quarter", "38%") as code constants, while a locally retrained bundle could
replace the model behind them. Now every number shown comes from the
**claims** of the bundle that was loaded:

* bundles written from round 4 on carry them (`bundle["claims"]`);
* bundles shipped earlier are listed in ``models/claims.json`` by the
  SHA-256 of the file the numbers were measured on.

A bundle with neither gets no numbers quoted, and the app refuses such a
bundle when it would silently override the packaged one (``app._pick_bundle``).
Local only: no network.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

CLAIMS_FILE = Path(__file__).parent / "models" / "claims.json"

# Named fractions for "about ..." wording; a rate within ABOUT_TOLERANCE of one
# is said in words, a rate under ABOUT_PERCENT_BELOW as a percentage.
_NAMED = ((1 / 4, "a quarter"), (1 / 3, "a third"), (1 / 2, "half"), (2 / 3, "two thirds"),
          (3 / 4, "three quarters"), (4 / 5, "4 in 5"), (9 / 10, "9 in 10"))
ABOUT_TOLERANCE = 0.05
ABOUT_PERCENT_BELOW = 0.15


@lru_cache(maxsize=1)
def _table() -> dict:
    try:
        return json.loads(CLAIMS_FILE.read_text())
    except (OSError, ValueError):
        return {}


@lru_cache(maxsize=16)
def _sha256(path: str, mtime_ns: int, size: int) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_sha256(path: str | Path) -> str:
    st = Path(path).stat()
    return _sha256(str(Path(path).resolve()), st.st_mtime_ns, st.st_size)


def claims_for(bundle: dict | None) -> dict | None:
    """The claims of a loaded bundle: its own ``claims``, else the entry for
    its file in ``models/claims.json``, else None."""
    if not isinstance(bundle, dict):
        return None
    own = bundle.get("claims")
    if isinstance(own, dict):
        return own
    path = bundle.get("_path")
    if not path or not Path(path).is_file():
        return None
    entry = _table().get(file_sha256(path))
    return entry if isinstance(entry, dict) else None


def about(rate: float) -> str:
    """A rate in words: 'about half', 'about 4 in 5', 'about 5%'."""
    if rate <= 0:
        return "none"
    if rate < ABOUT_PERCENT_BELOW:
        return f"about {max(1, round(rate * 100))}%"
    frac, name = min(_NAMED, key=lambda x: abs(x[0] - rate))
    if abs(frac - rate) <= ABOUT_TOLERANCE:
        return f"about {name}"
    return f"about {rate:.0%}"


def _recall_sentence(recall: list[dict]) -> str:
    """'about 4 in 5 clips from A, about half from B and from C'."""
    groups: list[tuple[str, list[str]]] = []
    for r in recall:
        phrase = about(float(r["rate"]))
        if groups and groups[-1][0] == phrase:
            groups[-1][1].append(r["what"])
        else:
            groups.append((phrase, [r["what"]]))
    out = []
    for i, (phrase, whats) in enumerate(groups):
        out.append(f"{phrase}{' clips' if i == 0 else ''} " + " and ".join(f"from {w}" for w in whats))
    return ", ".join(out)


def caveat(ai: dict | None, pulse: dict | None, fsd: dict | None, fsd_mode: str = "evidence") -> str:
    """The always-visible caveat, from the claims of the models in use."""
    if ai:
        s = (f"Checks for fully AI-generated video with a whole-frame detector trained on {ai['families']} "
             f"generator families. On held-out test sets it caught {_recall_sentence(ai['recall'])}, and "
             f"wrongly flagged {about(float(ai['real_fpr']['rate']))} of real test videos ({ai['source']}). "
             f"Clips under {float(ai['min_seconds']):g} s are not checked.")
    else:
        s = ("Checks for fully AI-generated video with a whole-frame detector; the installed model records no "
             "measured accuracy, so none is quoted.")
    s += " It also reads C2PA Content Credentials when present."
    if pulse:
        rec = pulse.get("recall", {})
        reenact = max(float(rec.get("Face2Face", 0.0)), float(rec.get("NeuralTextures", 0.0)))
        blind = ("is blind to reenactment fakes such as Face2Face and NeuralTextures" if reenact < 0.05
                 else f"catches {about(reenact)} of reenactment fakes")
        pulse_s = (f"a weak pulse check can decide (it catches {about(float(rec.get('Deepfakes', 0.0)))} of "
                   f"Deepfakes-style swaps and {blind})")
    else:
        pulse_s = "a pulse check can decide (the installed model records no measured accuracy)"
    if fsd:
        out = fsd.get("outside_fpr") or {}
        rate = out.get("rate")
        if fsd_mode == "decisive":
            fsd_s = (f"a face-swap detector trained on {fsd.get('trained_on', 'face-swap datasets')} decides when it "
                     f"finds a face" + (f"; on real faces from other sources it flagged {float(rate):.0%}"
                                        if rate is not None else "") + f" ({fsd['source']})")
        else:
            fsd_s = (f"a face-swap detector trained on {fsd.get('trained_on', 'face-swap datasets')} is shown as "
                     "evidence only" + (f", because it wrongly flagged {float(rate):.0%} of real faces from other "
                                        "sources" if rate is not None else "") + f" ({fsd['source']})")
    else:
        fsd_s = "a face-swap detector, when installed, is shown as evidence only (no measured accuracy recorded)"
    s += f" Face-swaps: {pulse_s}; {fsd_s}. 'No sign of AI generation' is not proof a video is real."
    return s
