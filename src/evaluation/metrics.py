"""Evaluation metrics for biomarker and model assessment."""

from typing import Dict, Optional

import pandas as pd


def compute_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_proba: Optional[pd.Series] = None,
) -> Dict[str, float]:
    """Compute classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities (optional, for AUC).

    Returns:
        Dictionary with metric names and values.
    """
    raise NotImplementedError


def compute_biomarker_metrics(
    df_biomarkers: pd.DataFrame,
    df_diagnoses: pd.DataFrame,
    disease_col: str,
) -> Dict[str, Dict[str, float]]:
    """Compute metrics for biomarker-disease associations.

    Args:
        df_biomarkers: DataFrame with binary biomarker columns.
        df_diagnoses: DataFrame with disease diagnosis labels.
        disease_col: Column name for target disease.

    Returns:
        Dictionary mapping biomarker names to their metrics.
    """
    raise NotImplementedError


def compute_feature_importance_correlation(
    importance_1: pd.Series,
    importance_2: pd.Series,
) -> Dict[str, float]:
    """Compute correlation between two feature importance rankings.

    Args:
        importance_1: First feature importance series.
        importance_2: Second feature importance series.

    Returns:
        Dictionary with correlation metrics (Spearman, Kendall, etc.).
    """
    raise NotImplementedError
