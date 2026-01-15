"""Data-driven threshold biomarker generation."""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold

from .base import BaseBiomarkerGenerator, BiomarkerGenerator


class DataDrivenThresholdGenerator(BaseBiomarkerGenerator):
    """Generate biomarker labels using data-driven thresholds.

    Learns thresholds from the data distribution using methods like:
    - Percentile-based cutoffs
    - Clustering
    - Optimal cutpoint analysis
    """

    def __init__(
        self,
        method: str = "percentile",
        lower_percentile: float = 10.0,
        upper_percentile: float = 90.0,
        feature_config: Optional[Dict] = None,
    ):
        """Initialize data-driven threshold generator.

        Args:
            method: Threshold learning method ('percentile', 'optimal').
            lower_percentile: Lower percentile for abnormal low values.
            upper_percentile: Upper percentile for abnormal high values.
            feature_config: Additional feature configuration.
        """
        super().__init__(feature_config)
        self.method = method
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def fit(self, df: pd.DataFrame) -> "DataDrivenThresholdGenerator":
        """Learn thresholds from data distribution.

        Args:
            df: Training data with CBC features.

        Returns:
            Fitted generator with learned thresholds.
        """
        raise NotImplementedError

    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate binary labels based on learned thresholds.

        Args:
            df: DataFrame with CBC features.

        Returns:
            DataFrame with binary columns for each biomarker.
        """
        raise NotImplementedError

    def get_thresholds(self) -> Dict[str, Dict]:
        """Return learned thresholds.

        Returns:
            Dictionary of feature thresholds learned from data.
        """
        return self.thresholds.copy()


class YoudensIndexGenerator(BiomarkerGenerator):
    """Method 1B: Single feature with data-driven threshold using Youden's Index.

    This generator implements an optimized biomarker discovery method:
    1. Train a logistic regression on all features
    2. Select the feature with the highest absolute coefficient
    3. Optimize the threshold using Youden's Index (maximizes sensitivity + specificity - 1)
    4. Optionally use cross-validation to avoid overfitting

    Youden's Index finds the threshold that maximizes the difference between
    the true positive rate and false positive rate on the ROC curve.
    """

    def __init__(self, config: dict):
        """Initialize the generator.

        Args:
            config: Configuration dictionary with optional parameters:
                - 'random_state': int - Random seed for reproducibility
                - 'max_iter': int - Maximum iterations for logistic regression
                - 'use_cv': bool - Whether to use cross-validation (default: True)
                - 'cv_folds': int - Number of CV folds (default: 5)
        """
        super().__init__(config)
        self.selected_feature_ = None
        self.optimized_threshold_ = None
        self.feature_names_ = None
        self.model_ = None
        self.coefficients_ = None
        self.cv_thresholds_ = None  # Store thresholds from each CV fold

    def generate(
        self, X_train: np.ndarray, y_train: np.ndarray, features: List[str]
    ) -> Dict[str, Any]:
        """Generate biomarker from training data.

        Uses logistic regression to identify the most predictive feature,
        then optimizes threshold using Youden's Index.

        Args:
            X_train: Training feature matrix of shape (n_samples, n_features)
            y_train: Training labels of shape (n_samples,)
            features: List of feature names corresponding to X_train columns

        Returns:
            Dictionary containing:
            - 'formula': str - Selected feature and threshold
            - 'threshold': float - Optimized threshold value
            - 'features_used': list[str] - Single feature used
            - 'metadata': dict - Coefficients, Youden's Index, and training info

        Raises:
            ValueError: If input data is invalid
        """
        # Validate inputs
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                f"X_train and y_train shape mismatch: {X_train.shape[0]} vs {y_train.shape[0]}"
            )
        if X_train.shape[1] != len(features):
            raise ValueError(
                f"Number of features mismatch: {X_train.shape[1]} vs {len(features)}"
            )

        # Check if we have both classes
        unique_classes = np.unique(y_train)
        if len(unique_classes) < 2:
            raise ValueError(
                f"Need at least 2 classes in y_train, found {len(unique_classes)}"
            )

        self.feature_names_ = features

        # Train logistic regression to get feature importance
        random_state = self.config.get("random_state", 42)
        max_iter = self.config.get("max_iter", 1000)

        self.model_ = LogisticRegression(
            random_state=random_state, max_iter=max_iter, penalty=None
        )
        self.model_.fit(X_train, y_train)

        # Get coefficients (importance scores)
        self.coefficients_ = self.model_.coef_[0]

        # Select feature with highest absolute coefficient
        feature_idx = np.argmax(np.abs(self.coefficients_))
        self.selected_feature_ = features[feature_idx]
        selected_coef = self.coefficients_[feature_idx]

        # Optimize threshold using Youden's Index
        use_cv = self.config.get("use_cv", True)

        if use_cv:
            # Use cross-validation to find robust threshold
            threshold, youden_index, cv_info = self._optimize_threshold_cv(
                X_train, y_train, feature_idx
            )
            self.cv_thresholds_ = cv_info["cv_thresholds"]
        else:
            # Use entire training set
            threshold, youden_index = self._optimize_threshold_single(
                X_train[:, feature_idx], y_train
            )
            self.cv_thresholds_ = None

        self.optimized_threshold_ = threshold

        # Determine threshold direction based on coefficient sign
        # Positive coefficient means higher values predict positive class
        threshold_direction = "high" if selected_coef > 0 else "low"

        # Create formula description
        operator = ">=" if threshold_direction == "high" else "<="
        formula = f"{self.selected_feature_} {operator} {threshold:.4f}"

        # Store biomarker info
        metadata = {
            "selected_feature": self.selected_feature_,
            "coefficient": float(selected_coef),
            "threshold_direction": threshold_direction,
            "youden_index": float(youden_index),
            "all_coefficients": {
                feat: float(coef)
                for feat, coef in zip(features, self.coefficients_)
            },
            "training_accuracy": float(self.model_.score(X_train, y_train)),
            "optimization_method": "cv" if use_cv else "single",
        }

        if use_cv:
            metadata["cv_folds"] = cv_info["n_folds"]
            metadata["cv_thresholds"] = [float(t) for t in cv_info["cv_thresholds"]]
            metadata["cv_threshold_std"] = float(cv_info["threshold_std"])

        self.biomarker = {
            "formula": formula,
            "threshold": float(threshold),
            "features_used": [self.selected_feature_],
            "metadata": metadata,
        }

        return self.biomarker

    def apply(self, X_test: np.ndarray) -> np.ndarray:
        """Apply learned biomarker to test data.

        Args:
            X_test: Test feature matrix of shape (n_samples, n_features)

        Returns:
            Binary predictions of shape (n_samples,)

        Raises:
            RuntimeError: If generate() has not been called yet
            ValueError: If X_test has wrong number of features
        """
        if self.biomarker is None or self.selected_feature_ is None:
            raise RuntimeError(
                "Biomarker not generated yet. Call generate() first."
            )

        if X_test.shape[1] != len(self.feature_names_):
            raise ValueError(
                f"X_test has {X_test.shape[1]} features, expected {len(self.feature_names_)}"
            )

        # Get the index of the selected feature
        feature_idx = self.feature_names_.index(self.selected_feature_)

        # Extract the selected feature values
        feature_values = X_test[:, feature_idx]

        # Apply threshold based on direction
        threshold_direction = self.biomarker["metadata"]["threshold_direction"]
        if threshold_direction == "high":
            predictions = (feature_values >= self.optimized_threshold_).astype(int)
        else:
            predictions = (feature_values <= self.optimized_threshold_).astype(int)

        return predictions

    def get_description(self) -> str:
        """Return human-readable description of the learned biomarker.

        Returns:
            String description of the biomarker formula and threshold.
        """
        if self.biomarker is None:
            return "No biomarker generated yet. Call generate() first."

        formula = self.biomarker["formula"]
        feature = self.biomarker["metadata"]["selected_feature"]
        coef = self.biomarker["metadata"]["coefficient"]
        youden = self.biomarker["metadata"]["youden_index"]
        acc = self.biomarker["metadata"]["training_accuracy"]
        method = self.biomarker["metadata"]["optimization_method"]

        desc = (
            f"Data-driven biomarker (Youden's Index): {formula}\n"
            f"Selected feature: {feature} (coefficient: {coef:.3f})\n"
            f"Youden's Index: {youden:.3f}\n"
            f"Training accuracy: {acc:.3f}\n"
            f"Optimization: {method}"
        )

        if method == "cv":
            cv_std = self.biomarker["metadata"]["cv_threshold_std"]
            desc += f" (threshold std: {cv_std:.4f})"

        return desc

    def _optimize_threshold_single(
        self, feature_values: np.ndarray, y_true: np.ndarray
    ) -> tuple[float, float]:
        """Optimize threshold using Youden's Index on a single dataset.

        Args:
            feature_values: Feature values of shape (n_samples,)
            y_true: True labels of shape (n_samples,)

        Returns:
            Tuple of (optimal_threshold, youden_index)
        """
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, feature_values)

        # Youden's Index = Sensitivity + Specificity - 1 = TPR - FPR
        youden_indices = tpr - fpr

        # Filter out infinite thresholds (can occur at edges of ROC curve)
        finite_mask = np.isfinite(thresholds)

        if not np.any(finite_mask):
            # If no finite thresholds, use median of feature values as fallback
            optimal_threshold = np.median(feature_values)
            # Calculate Youden's Index for this threshold
            predictions = (feature_values >= optimal_threshold).astype(int)
            tpr_val = np.mean(predictions[y_true == 1]) if np.any(y_true == 1) else 0
            fpr_val = np.mean(predictions[y_true == 0]) if np.any(y_true == 0) else 0
            max_youden_index = tpr_val - fpr_val
        else:
            # Find threshold that maximizes Youden's Index among finite values
            finite_youden = youden_indices[finite_mask]
            finite_thresholds = thresholds[finite_mask]

            optimal_idx = np.argmax(finite_youden)
            optimal_threshold = finite_thresholds[optimal_idx]
            max_youden_index = finite_youden[optimal_idx]

        return optimal_threshold, max_youden_index

    def _optimize_threshold_cv(
        self, X: np.ndarray, y: np.ndarray, feature_idx: int
    ) -> tuple[float, float, Dict]:
        """Optimize threshold using cross-validation.

        This helps avoid overfitting by averaging thresholds from multiple folds.

        Args:
            X: Full feature matrix of shape (n_samples, n_features)
            y: True labels of shape (n_samples,)
            feature_idx: Index of the selected feature

        Returns:
            Tuple of (mean_threshold, mean_youden_index, cv_info_dict)
        """
        cv_folds = self.config.get("cv_folds", 5)
        random_state = self.config.get("random_state", 42)

        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

        cv_thresholds = []
        cv_youden_indices = []

        for train_idx, val_idx in skf.split(X, y):
            # Get validation fold
            X_val_feature = X[val_idx, feature_idx]
            y_val = y[val_idx]

            # Optimize threshold on validation fold
            threshold, youden_idx = self._optimize_threshold_single(
                X_val_feature, y_val
            )

            cv_thresholds.append(threshold)
            cv_youden_indices.append(youden_idx)

        # Return mean threshold and mean Youden's Index
        mean_threshold = np.mean(cv_thresholds)
        mean_youden_index = np.mean(cv_youden_indices)
        threshold_std = np.std(cv_thresholds)

        cv_info = {
            "n_folds": cv_folds,
            "cv_thresholds": cv_thresholds,
            "cv_youden_indices": cv_youden_indices,
            "threshold_std": threshold_std,
        }

        return mean_threshold, mean_youden_index, cv_info
