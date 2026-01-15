"""Feature selection methods for biomarker identification."""

from typing import List, Optional, Tuple

import pandas as pd


class FeatureSelector:
    """Select important features for disease prediction."""

    def __init__(self, method: str = "mutual_info"):
        """Initialize FeatureSelector.

        Args:
            method: Selection method ('mutual_info', 'f_classif', 'rfe', 'lasso').
        """
        self.method = method
        self.selected_features: List[str] = []
        self.feature_importances: Optional[pd.Series] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FeatureSelector":
        """Fit the feature selector.

        Args:
            X: Feature matrix.
            y: Target labels.

        Returns:
            Fitted selector instance.
        """
        raise NotImplementedError

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data to selected features only.

        Args:
            X: Feature matrix.

        Returns:
            DataFrame with selected features.
        """
        raise NotImplementedError

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform in one step.

        Args:
            X: Feature matrix.
            y: Target labels.

        Returns:
            DataFrame with selected features.
        """
        self.fit(X, y)
        return self.transform(X)

    def get_feature_ranking(self) -> pd.Series:
        """Get feature importance ranking.

        Returns:
            Series with feature importances sorted descending.
        """
        raise NotImplementedError
