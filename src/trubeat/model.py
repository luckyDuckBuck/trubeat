"""Classifier training + single-video prediction. FR-11, FR-12."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Prediction:
    label: str  # "real" | "fake"
    confidence: float
    n_windows: int
    votes: list[str] = field(default_factory=list)
    hr_bpm: float | None = None
    warnings: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "n_windows": self.n_windows,
            "votes": self.votes,
            "hr_bpm": self.hr_bpm,
            "warnings": self.warnings,
        }


def train_classifier(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray):
    """Fit Pipeline(StandardScaler, LogisticRegression|SVC) with GroupKFold CV.

    groups = split_group from the manifest (subject / source video).
    NEVER train without groups — random splits leak identity (R-04).
    Phase 3: swap in the PyTorch CNN on PPG maps (same interface).
    """
    raise NotImplementedError


def predict_video(path: str, model, config: dict) -> Prediction:
    """Full pipeline on one video → label + per-window votes. Majority wins."""
    raise NotImplementedError
