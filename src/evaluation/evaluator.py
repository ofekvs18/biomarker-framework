"""Model and biomarker evaluation orchestration."""

from typing import Any, Dict, List, Optional

import pandas as pd


class Evaluator:
    """Orchestrate model and biomarker evaluation."""

    def __init__(self, metrics: Optional[List[str]] = None):
        """Initialize Evaluator.

        Args:
            metrics: List of metrics to compute.
        """
        self.metrics = metrics or [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "auc_roc",
        ]
        self.results: Dict[str, Any] = {}

    def evaluate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, float]:
        """Evaluate a trained model.

        Args:
            model: Trained model with predict/predict_proba methods.
            X_test: Test features.
            y_test: Test labels.

        Returns:
            Dictionary of metric values.
        """
        raise NotImplementedError

    def evaluate_biomarkers(
        self,
        df_biomarkers: pd.DataFrame,
        df_labels: pd.DataFrame,
        target_col: str,
    ) -> pd.DataFrame:
        """Evaluate biomarker predictive power.

        Args:
            df_biomarkers: DataFrame with binary biomarker columns.
            df_labels: DataFrame with target labels.
            target_col: Column name for target variable.

        Returns:
            DataFrame with metrics per biomarker.
        """
        raise NotImplementedError

    def cross_validate(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
    ) -> Dict[str, List[float]]:
        """Perform cross-validation.

        Args:
            model: Model to evaluate.
            X: Feature matrix.
            y: Target labels.
            cv: Number of folds.

        Returns:
            Dictionary mapping metrics to list of fold scores.
        """
        raise NotImplementedError

    def compare_generators(
        self,
        results_by_generator: Dict[str, Dict],
    ) -> pd.DataFrame:
        """Compare results across different biomarker generators.

        Args:
            results_by_generator: Dictionary mapping generator names to results.

        Returns:
            Comparison DataFrame.
        """
        raise NotImplementedError
