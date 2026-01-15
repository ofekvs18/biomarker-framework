"""SHAP-based literature threshold biomarker generation."""

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ..models.shap_selector import SHAPFeatureSelector
from .base import BiomarkerGenerator


class SHAPLiteratureThresholdGenerator(BiomarkerGenerator):
    """Method 2A: SHAP-based feature selection with literature-defined threshold.

    This generator implements a SHAP-based biomarker discovery method:
    1. Train a model (default: RandomForest) on all features
    2. Use SHAP values to identify the most important feature
    3. Apply the literature-defined threshold for that feature

    SHAP values provide a unified, theoretically-grounded measure of feature
    importance based on Shapley values from cooperative game theory. Unlike
    simple coefficients, SHAP values account for feature interactions and
    provide consistent attributions across different model types.
    """

    def __init__(self, config: dict, literature_thresholds: dict):
        """Initialize the generator.

        Args:
            config: Configuration dictionary with optional parameters:
                - 'random_state': int - Random seed for reproducibility
                - 'model_type': str - Model type ('RandomForest', 'XGBoost', 'LogisticRegression')
                - 'n_estimators': int - Number of trees (for tree-based models)
                - 'max_depth': int - Maximum tree depth
                - 'min_samples_split': int - Minimum samples to split
                - 'threshold_direction': str - 'high' or 'low' for abnormal values
                - 'shap_explainer': str - SHAP explainer type ('auto', 'tree', 'linear', 'kernel')
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
        self.shap_selector_ = None
        self.shap_importances_ = None

    def _create_model(self):
        """Create model based on configuration.

        Returns:
            Initialized sklearn-compatible model
        """
        model_type = self.config.get("model_type", "RandomForest")
        random_state = self.config.get("random_state", 42)

        if model_type == "RandomForest":
            return RandomForestClassifier(
                n_estimators=self.config.get("n_estimators", 100),
                max_depth=self.config.get("max_depth", 10),
                min_samples_split=self.config.get("min_samples_split", 5),
                random_state=random_state,
                n_jobs=-1
            )
        elif model_type == "XGBoost":
            try:
                import xgboost as xgb
                return xgb.XGBClassifier(
                    n_estimators=self.config.get("n_estimators", 100),
                    max_depth=self.config.get("max_depth", 6),
                    learning_rate=self.config.get("learning_rate", 0.1),
                    random_state=random_state,
                    n_jobs=-1
                )
            except ImportError:
                raise ImportError(
                    "XGBoost not installed. Install with: pip install xgboost"
                )
        elif model_type == "LogisticRegression":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(
                max_iter=self.config.get("max_iter", 1000),
                random_state=random_state
            )
        else:
            raise ValueError(
                f"Unsupported model_type: {model_type}. "
                f"Choose from: RandomForest, XGBoost, LogisticRegression"
            )

    def generate(
        self, X_train: np.ndarray, y_train: np.ndarray, features: List[str]
    ) -> Dict[str, Any]:
        """Generate biomarker from training data using SHAP-based feature selection.

        Uses SHAP values to identify the most important feature,
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
            - 'metadata': dict - SHAP importances and training info

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

        # Train model
        self.model_ = self._create_model()
        self.model_.fit(X_train, y_train)

        # Compute SHAP values
        shap_explainer_type = self.config.get("shap_explainer", "auto")
        self.shap_selector_ = SHAPFeatureSelector(
            model=self.model_,
            model_type=shap_explainer_type,
            random_state=self.config.get("random_state", 42)
        )
        self.shap_selector_.fit(X_train, feature_names=features)

        # Get feature importance rankings
        self.shap_importances_ = self.shap_selector_.get_feature_importances()

        # Select feature with highest mean absolute SHAP value
        self.selected_feature_, shap_importance = self.shap_selector_.get_top_feature()

        # Get literature threshold for selected feature
        threshold_value = self._get_threshold_for_feature(self.selected_feature_)
        if threshold_value is None:
            raise ValueError(
                f"No literature threshold found for feature: {self.selected_feature_}"
            )

        self.selected_threshold_ = threshold_value

        # Determine threshold direction
        # We need to check if higher values of this feature predict positive class
        # by looking at the mean SHAP value (not absolute)
        feature_idx = features.index(self.selected_feature_)
        shap_values = self.shap_selector_.get_shap_values()
        mean_shap = np.mean(shap_values[:, feature_idx])

        # Positive mean SHAP means higher values predict positive class
        threshold_direction = "high" if mean_shap > 0 else "low"

        # Create formula description
        operator = ">=" if threshold_direction == "high" else "<="
        formula = f"{self.selected_feature_} {operator} {threshold_value}"

        # Get top features for comparison
        top_5_features = self.shap_selector_.get_top_k_features(k=5)

        # Store biomarker info
        self.biomarker = {
            "formula": formula,
            "threshold": threshold_value,
            "features_used": [self.selected_feature_],
            "metadata": {
                "selected_feature": self.selected_feature_,
                "shap_importance": float(shap_importance),
                "mean_shap_value": float(mean_shap),
                "threshold_direction": threshold_direction,
                "all_shap_importances": self.shap_importances_,
                "top_5_features": [
                    {"feature": feat, "importance": float(imp)}
                    for feat, imp in top_5_features
                ],
                "model_type": self.config.get("model_type", "RandomForest"),
                "shap_explainer_type": self.shap_selector_.model_type,
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
            String description of the biomarker formula and SHAP importance.
        """
        if self.biomarker is None:
            return "No biomarker generated yet. Call generate() first."

        formula = self.biomarker["formula"]
        feature = self.biomarker["metadata"]["selected_feature"]
        shap_imp = self.biomarker["metadata"]["shap_importance"]
        acc = self.biomarker["metadata"]["training_accuracy"]
        model_type = self.biomarker["metadata"]["model_type"]

        return (
            f"SHAP-based literature biomarker: {formula}\n"
            f"Selected feature: {feature} (SHAP importance: {shap_imp:.3f})\n"
            f"Model: {model_type}\n"
            f"Training accuracy: {acc:.3f}"
        )

    def get_shap_plots(self, max_display: int = 20):
        """Generate SHAP visualization plots.

        Args:
            max_display: Maximum number of features to display

        Returns:
            Dictionary with matplotlib Figure objects:
            - 'summary': SHAP beeswarm summary plot
            - 'bar': SHAP bar plot of mean importances

        Raises:
            RuntimeError: If generate() has not been called yet
        """
        if self.shap_selector_ is None:
            raise RuntimeError(
                "Biomarker not generated yet. Call generate() first."
            )

        return {
            "summary": self.shap_selector_.plot_summary(
                max_display=max_display, show=False
            ),
            "bar": self.shap_selector_.plot_bar(max_display=max_display, show=False),
        }

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
