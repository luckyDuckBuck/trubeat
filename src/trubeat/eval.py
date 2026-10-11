"""Batch evaluation + run reports. FR-13, FR-14."""

from dataclasses import dataclass

import pandas as pd


@dataclass
class HRReport:
    mae: float
    rmse: float
    pearson_r: float
    per_subject: pd.DataFrame


@dataclass
class DetectionReport:
    accuracy: float
    auc: float
    precision: float
    recall: float
    fpr_on_real: float  # headline metric — the BBC lesson
    # ... + confusion matrix, ROC data, bootstrap CIs


def validate_manifest(manifest: pd.DataFrame) -> None:
    """Fail fast if any split_group appears on both sides of a split (NFR-08)."""
    raise NotImplementedError


def evaluate_hr(manifest: pd.DataFrame, config: dict) -> HRReport:
    """Score HR estimates vs ground truth. Gate G1: MAE ≤ 5 bpm."""
    raise NotImplementedError


def evaluate_detector(manifest: pd.DataFrame, model, config: dict) -> DetectionReport:
    """Score the detector. Gate G2: AUC ≥ 0.70 person-disjoint."""
    raise NotImplementedError


def write_report(report, config: dict, out_path: str) -> None:
    """Markdown report + plots per docs/08_acceptance_test_plan.md §5."""
    raise NotImplementedError
