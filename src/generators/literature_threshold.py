"""Literature-based threshold biomarker generation."""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .base import BaseBiomarkerGenerator, BiomarkerGenerator


class LiteratureThresholdGenerator(BaseBiomarkerGenerator):
    """Generate biomarker labels using clinically-established thresholds.

    Uses reference ranges from medical literature to classify
    lab values as normal/abnormal.
    """

    # Default CBC reference ranges (can be overridden via config)
    DEFAULT_RANGES = {
        "hemoglobin": {"low": 12.0, "high": 17.5, "unit": "g/dL"},
        "hematocrit": {"low": 36.0, "high": 50.0, "unit": "%"},
        "wbc": {"low": 4.5, "high": 11.0, "unit": "K/uL"},
        "platelets": {"low": 150, "high": 400, "unit": "K/uL"},
        "rbc": {"low": 4.0, "high": 5.5, "unit": "M/uL"},
        "mcv": {"low": 80, "high": 100, "unit": "fL"},
        "mch": {"low": 27, "high": 33, "unit": "pg"},
        "mchc": {"low": 32, "high": 36, "unit": "g/dL"},
        "rdw": {"low": 11.5, "high": 14.5, "unit": "%"},
    }

    def __init__(self, feature_config: Optional[Dict] = None):
        """Initialize with literature-based thresholds.

        Args:
            feature_config: Override default thresholds if provided.
        """
        super().__init__(feature_config)
        self.thresholds = {**self.DEFAULT_RANGES}
        if feature_config:
            self.thresholds.update(feature_config)

    def fit(self, df: pd.DataFrame) -> "LiteratureThresholdGenerator":
        """No fitting needed for literature-based thresholds.

        Args:
            df: Training data (unused).

        Returns:
            Self (thresholds are predefined).
        """
        return self

    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate binary labels based on literature thresholds.

        Args:
            df: DataFrame with CBC features.

        Returns:
            DataFrame with binary columns for each biomarker.
        """
        raise NotImplementedError

    def get_thresholds(self) -> Dict[str, Dict]:
        """Return literature-based thresholds.

        Returns:
            Dictionary of feature thresholds.
        """
        return self.thresholds.copy()


class SingleFeatureLiteratureGenerator(BiomarkerGenerator):
    """Method 1A: Single feature with literature-defined threshold.

    This generator implements a simple biomarker discovery method:
    1. Train a logistic regression on all features
    2. Select the feature with the highest absolute coefficient
    3. Use the literature-defined threshold for that feature

    This serves as a baseline method combining data-driven feature selection
    with domain knowledge (literature thresholds).
    """

    def __init__(self, config: dict, literature_thresholds: dict):
        """Initialize the generator.

        Args:
            config: Configuration dictionary with optional parameters:
                - 'random_state': int - Random seed for reproducibility
                - 'max_iter': int - Maximum iterations for logistic regression
                - 'threshold_direction': str - 'high' or 'low' for abnormal values
            literature_thresholds: Dictionary mapping feature names to thresholds.
                Example: {'hemoglobin': 12.0, 'wbc': 11.0, 'platelets': 150}
                Can also contain structured thresholds with 'low' and 'high' keys.
        """
        super().__init__(config)
        self.literature_thresholds = literature_thresholds
        self.selected_feature_ = None
        self.selected_threshold_ = None
        self.feature_names_ = None
        self.model_ = None
        self.coefficients_ = None

    def generate(
        self, X_train: np.ndarray, y_train: np.ndarray, features: List[str]
    ) -> Dict[str, Any]:
        """Generate biomarker from training data.

        Uses logistic regression to identify the most predictive feature,
        then applies literature threshold for that feature.

        Args:
            X_train: Training feature matrix of shape (n_samples, n_features)
            y_train: Training labels of shape (n_samples,)
            features: List of feature names corresponding to X_train columns

        Returns:
            Dictionary containing:
            - 'formula': str - Selected feature and threshold
            - 'threshold': float - Literature threshold value
            - 'features_used': list[str] - Single feature used
            - 'metadata': dict - Coefficients and training info

        Raises:
            ValueError: If input data is invalid or no threshold available
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

        self.feature_names_ = features

        # Train logistic regression to get feature importance
        random_state = self.config.get("random_state", 42)
        max_iter = self.config.get("max_iter", 1000)

        self.model_ = LogisticRegression(
            random_state=random_state, max_iter=max_iter
        )
        self.model_.fit(X_train, y_train)

        # Get coefficients (importance scores)
        self.coefficients_ = self.model_.coef_[0]

        # Select feature with highest absolute coefficient
        feature_idx = np.argmax(np.abs(self.coefficients_))
        self.selected_feature_ = features[feature_idx]
        selected_coef = self.coefficients_[feature_idx]

        # Get literature threshold for selected feature
        threshold_value = self._get_threshold_for_feature(self.selected_feature_)
        if threshold_value is None:
            raise ValueError(
                f"No literature threshold found for feature: {self.selected_feature_}"
            )

        self.selected_threshold_ = threshold_value

        # Determine threshold direction based on coefficient sign
        # Positive coefficient means higher values predict positive class
        threshold_direction = "high" if selected_coef > 0 else "low"

        # Create formula description
        operator = ">=" if threshold_direction == "high" else "<="
        formula = f"{self.selected_feature_} {operator} {threshold_value}"

        # Store biomarker info
        self.biomarker = {
            "formula": formula,
            "threshold": threshold_value,
            "features_used": [self.selected_feature_],
            "metadata": {
                "selected_feature": self.selected_feature_,
                "coefficient": float(selected_coef),
                "threshold_direction": threshold_direction,
                "all_coefficients": {
                    feat: float(coef)
                    for feat, coef in zip(features, self.coefficients_)
                },
                "training_accuracy": float(self.model_.score(X_train, y_train)),
            },
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
            predictions = (feature_values >= self.selected_threshold_).astype(int)
        else:
            predictions = (feature_values <= self.selected_threshold_).astype(int)

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
        acc = self.biomarker["metadata"]["training_accuracy"]

        return (
            f"Literature-based biomarker: {formula}\n"
            f"Selected feature: {feature} (coefficient: {coef:.3f})\n"
            f"Training accuracy: {acc:.3f}"
        )

    def _get_threshold_for_feature(self, feature_name: str) -> Optional[float]:
        """Extract threshold value for a feature from literature thresholds.

        Handles both simple float thresholds and structured dicts with 'low'/'high'.

        Args:
            feature_name: Name of the feature

        Returns:
            Threshold value or None if not found
        """
        if feature_name not in self.literature_thresholds:
            return None

        threshold_info = self.literature_thresholds[feature_name]

        # If it's a simple float/int, return it directly
        if isinstance(threshold_info, (int, float)):
            return float(threshold_info)

        # If it's a dict, try to get the appropriate threshold
        if isinstance(threshold_info, dict):
            # Prefer direction from config if specified
            direction = self.config.get("threshold_direction", "low")
            if direction in threshold_info:
                return float(threshold_info[direction])
            # Otherwise, try common keys
            if "low" in threshold_info:
                return float(threshold_info["low"])
            if "high" in threshold_info:
                return float(threshold_info["high"])
            if "threshold" in threshold_info:
                return float(threshold_info["threshold"])

        return None
