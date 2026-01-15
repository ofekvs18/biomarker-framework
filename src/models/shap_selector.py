"""SHAP-based feature selection for biomarker identification.

This module provides feature selection using SHAP (SHapley Additive exPlanations) values,
which provide a unified measure of feature importance based on game theory.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.base import BaseEstimator


class SHAPFeatureSelector:
    """Select important features using SHAP values.

    SHAP values provide consistent and locally accurate feature importance
    attributions based on Shapley values from cooperative game theory.

    Attributes:
        model: Trained sklearn-compatible model
        model_type: Type of model ('tree', 'linear', 'kernel')
        explainer: SHAP explainer instance
        shap_values: Computed SHAP values for training data
        base_value: Expected value of model output
        feature_names: Names of input features
        X_background: Background data for explainer
    """

    def __init__(
        self,
        model: BaseEstimator,
        model_type: str = "auto",
        background_samples: int = 100,
        random_state: int = 42
    ):
        """Initialize SHAP feature selector.

        Args:
            model: Trained sklearn-compatible model
            model_type: Type of explainer to use:
                - 'auto': Automatically detect best explainer
                - 'tree': TreeExplainer (fast, for tree-based models)
                - 'linear': LinearExplainer (for linear models)
                - 'kernel': KernelExplainer (slow, model-agnostic)
            background_samples: Number of background samples for KernelExplainer
            random_state: Random seed for reproducibility
        """
        self.model = model
        self.model_type = model_type
        self.background_samples = background_samples
        self.random_state = random_state

        self.explainer: Optional[Any] = None
        self.shap_values: Optional[np.ndarray] = None
        self.base_value: Optional[Union[float, np.ndarray]] = None
        self.feature_names: Optional[List[str]] = None
        self.X_background: Optional[np.ndarray] = None
        self._mean_abs_shap: Optional[np.ndarray] = None

    def _create_explainer(self, X: np.ndarray) -> Any:
        """Create appropriate SHAP explainer based on model type.

        Args:
            X: Training data for background distribution

        Returns:
            SHAP explainer instance

        Raises:
            ValueError: If model_type is not supported
        """
        if self.model_type == "auto":
            # Auto-detect model type
            model_name = type(self.model).__name__.lower()

            if any(tree_name in model_name for tree_name in
                   ['forest', 'tree', 'xgb', 'lgb', 'gbm', 'gradient']):
                self.model_type = "tree"
            elif any(linear_name in model_name for linear_name in
                    ['linear', 'logistic', 'ridge', 'lasso']):
                self.model_type = "linear"
            else:
                self.model_type = "kernel"

        # Create appropriate explainer
        if self.model_type == "tree":
            try:
                explainer = shap.TreeExplainer(self.model)
            except Exception as e:
                print(f"Warning: TreeExplainer failed ({e}), falling back to KernelExplainer")
                self.model_type = "kernel"

        if self.model_type == "linear":
            try:
                explainer = shap.LinearExplainer(self.model, X)
            except Exception as e:
                print(f"Warning: LinearExplainer failed ({e}), falling back to KernelExplainer")
                self.model_type = "kernel"

        if self.model_type == "kernel":
            # Use subset of data as background for efficiency
            np.random.seed(self.random_state)
            n_background = min(self.background_samples, len(X))
            background_indices = np.random.choice(len(X), n_background, replace=False)
            self.X_background = X[background_indices]

            # KernelExplainer needs predict_proba for classifiers
            if hasattr(self.model, 'predict_proba'):
                explainer = shap.KernelExplainer(self.model.predict_proba, self.X_background)
            else:
                explainer = shap.KernelExplainer(self.model.predict, self.X_background)

        return explainer

    def fit(
        self,
        X_train: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> "SHAPFeatureSelector":
        """Compute SHAP values on training data.

        Args:
            X_train: Training data (n_samples, n_features)
            feature_names: Names of features (optional)

        Returns:
            self for method chaining

        Raises:
            ValueError: If model is not trained or X_train is invalid
        """
        if X_train.ndim != 2:
            raise ValueError(f"X_train must be 2D array, got shape {X_train.shape}")

        # Check if model is fitted
        if not hasattr(self.model, 'predict'):
            raise ValueError("Model must be trained before computing SHAP values")

        # Store feature names
        if feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        else:
            if len(feature_names) != X_train.shape[1]:
                raise ValueError(
                    f"Number of feature names ({len(feature_names)}) "
                    f"must match number of features ({X_train.shape[1]})"
                )
            self.feature_names = feature_names

        # Create explainer
        self.explainer = self._create_explainer(X_train)

        # Compute SHAP values
        shap_values_output = self.explainer.shap_values(X_train)

        # Handle different SHAP output formats
        # For binary classification, some explainers return a list [class_0, class_1]
        if isinstance(shap_values_output, list):
            # Use positive class (class 1) SHAP values for binary classification
            self.shap_values = shap_values_output[1]
        else:
            self.shap_values = shap_values_output

        # Get base value (expected model output)
        if hasattr(self.explainer, 'expected_value'):
            base_value = self.explainer.expected_value
            if isinstance(base_value, list):
                self.base_value = base_value[1]  # Positive class
            elif isinstance(base_value, np.ndarray):
                self.base_value = base_value[1] if len(base_value) > 1 else base_value[0]
            else:
                self.base_value = base_value
        else:
            self.base_value = 0.0

        # Compute mean absolute SHAP values for feature importance ranking
        self._mean_abs_shap = np.mean(np.abs(self.shap_values), axis=0)

        return self

    def get_top_feature(self) -> Tuple[str, float]:
        """Get feature with highest mean absolute SHAP value.

        Returns:
            Tuple of (feature_name, importance_score)

        Raises:
            RuntimeError: If fit() has not been called
        """
        if self._mean_abs_shap is None:
            raise RuntimeError("Must call fit() before getting top feature")

        top_idx = np.argmax(self._mean_abs_shap)
        return self.feature_names[top_idx], self._mean_abs_shap[top_idx]

    def get_top_k_features(self, k: int = 5) -> List[Tuple[str, float]]:
        """Get top K features by mean absolute SHAP value.

        Args:
            k: Number of top features to return

        Returns:
            List of (feature_name, importance_score) tuples, sorted by importance

        Raises:
            RuntimeError: If fit() has not been called
        """
        if self._mean_abs_shap is None:
            raise RuntimeError("Must call fit() before getting top features")

        k = min(k, len(self.feature_names))
        top_indices = np.argsort(self._mean_abs_shap)[::-1][:k]

        return [
            (self.feature_names[idx], self._mean_abs_shap[idx])
            for idx in top_indices
        ]

    def get_feature_importances(self) -> Dict[str, float]:
        """Get importance scores for all features.

        Returns:
            Dictionary mapping feature names to mean absolute SHAP values

        Raises:
            RuntimeError: If fit() has not been called
        """
        if self._mean_abs_shap is None:
            raise RuntimeError("Must call fit() before getting importances")

        return {
            name: float(importance)
            for name, importance in zip(self.feature_names, self._mean_abs_shap)
        }

    def plot_summary(
        self,
        max_display: int = 20,
        show: bool = True,
        **kwargs
    ) -> plt.Figure:
        """Create SHAP summary plot (beeswarm plot).

        This plot shows the distribution of SHAP values for each feature,
        revealing both importance and effects (positive/negative).

        Args:
            max_display: Maximum number of features to display
            show: Whether to display the plot
            **kwargs: Additional arguments passed to shap.summary_plot

        Returns:
            Matplotlib figure object

        Raises:
            RuntimeError: If fit() has not been called
        """
        if self.shap_values is None:
            raise RuntimeError("Must call fit() before plotting")

        fig, ax = plt.subplots(figsize=(10, max(6, max_display * 0.3)))

        shap.summary_plot(
            self.shap_values,
            features=None,  # Will use feature names
            feature_names=self.feature_names,
            max_display=max_display,
            show=False,
            **kwargs
        )

        plt.tight_layout()

        if show:
            plt.show()

        return fig

    def plot_bar(
        self,
        max_display: int = 20,
        show: bool = True,
        **kwargs
    ) -> plt.Figure:
        """Create SHAP bar plot showing mean absolute importances.

        This plot shows feature importance ranking in a simple bar chart format.

        Args:
            max_display: Maximum number of features to display
            show: Whether to display the plot
            **kwargs: Additional arguments passed to shap.summary_plot

        Returns:
            Matplotlib figure object

        Raises:
            RuntimeError: If fit() has not been called
        """
        if self.shap_values is None:
            raise RuntimeError("Must call fit() before plotting")

        fig, ax = plt.subplots(figsize=(10, max(6, max_display * 0.3)))

        shap.summary_plot(
            self.shap_values,
            features=None,
            feature_names=self.feature_names,
            plot_type="bar",
            max_display=max_display,
            show=False,
            **kwargs
        )

        plt.tight_layout()

        if show:
            plt.show()

        return fig

    def plot_waterfall(
        self,
        sample_idx: int = 0,
        show: bool = True,
        **kwargs
    ) -> plt.Figure:
        """Create SHAP waterfall plot for a single prediction.

        This plot shows how each feature contributes to pushing the model output
        from the base value to the final prediction for a specific sample.

        Args:
            sample_idx: Index of sample to explain
            show: Whether to display the plot
            **kwargs: Additional arguments passed to shap.waterfall_plot

        Returns:
            Matplotlib figure object

        Raises:
            RuntimeError: If fit() has not been called
        """
        if self.shap_values is None:
            raise RuntimeError("Must call fit() before plotting")

        if sample_idx >= len(self.shap_values):
            raise ValueError(
                f"sample_idx {sample_idx} out of range for {len(self.shap_values)} samples"
            )

        fig, ax = plt.subplots(figsize=(10, 8))

        # Create Explanation object for waterfall plot
        explanation = shap.Explanation(
            values=self.shap_values[sample_idx],
            base_values=self.base_value,
            feature_names=self.feature_names
        )

        shap.waterfall_plot(explanation, show=False, **kwargs)

        plt.tight_layout()

        if show:
            plt.show()

        return fig

    def get_shap_values(self) -> np.ndarray:
        """Get raw SHAP values.

        Returns:
            Array of SHAP values (n_samples, n_features)

        Raises:
            RuntimeError: If fit() has not been called
        """
        if self.shap_values is None:
            raise RuntimeError("Must call fit() before getting SHAP values")

        return self.shap_values

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model and explainer.

        Returns:
            Dictionary with model and explainer information
        """
        return {
            'model_type': type(self.model).__name__,
            'explainer_type': self.model_type,
            'n_features': len(self.feature_names) if self.feature_names else None,
            'base_value': float(self.base_value) if self.base_value is not None else None,
            'background_samples': self.background_samples if self.model_type == 'kernel' else None
        }
