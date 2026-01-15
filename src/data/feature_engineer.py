"""Feature engineering for CBC biomarker analysis."""

from typing import Dict, List, Optional

import pandas as pd


class FeatureEngineer:
    """Engineer features from CBC lab values."""

    def __init__(self, cbc_features: Optional[List[str]] = None):
        """Initialize FeatureEngineer.

        Args:
            cbc_features: List of CBC feature names to process.
        """
        self.cbc_features = cbc_features or []

    def aggregate_patient_labs(
        self, df: pd.DataFrame, agg_functions: List[str] = None
    ) -> pd.DataFrame:
        """Aggregate lab values per patient.

        Args:
            df: Lab events DataFrame.
            agg_functions: Aggregation functions to apply.

        Returns:
            Aggregated features per patient.
        """
        raise NotImplementedError

    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features from CBC values.

        Args:
            df: DataFrame with CBC features.

        Returns:
            DataFrame with additional derived features.
        """
        raise NotImplementedError

    def create_train_test_split(
        self, df: pd.DataFrame, test_size: float = 0.2, stratify_col: str = None
    ) -> Dict[str, pd.DataFrame]:
        """Create train/validation/test splits.

        Args:
            df: Input DataFrame.
            test_size: Proportion for test set.
            stratify_col: Column to stratify by.

        Returns:
            Dictionary with train, val, test DataFrames.
        """
        raise NotImplementedError
