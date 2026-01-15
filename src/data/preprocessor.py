"""Data preprocessing and cleaning utilities."""

from typing import List, Optional

import pandas as pd


class Preprocessor:
    """Preprocess and clean MIMIC-IV data for analysis."""

    def __init__(self):
        """Initialize Preprocessor."""
        pass

    def clean_labevents(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean laboratory events data.

        Args:
            df: Raw labevents DataFrame.

        Returns:
            Cleaned DataFrame with valid values.
        """
        raise NotImplementedError

    def handle_missing_values(
        self, df: pd.DataFrame, strategy: str = "median"
    ) -> pd.DataFrame:
        """Handle missing values in the dataset.

        Args:
            df: Input DataFrame.
            strategy: Imputation strategy ('median', 'mean', 'drop').

        Returns:
            DataFrame with missing values handled.
        """
        raise NotImplementedError

    def remove_outliers(
        self, df: pd.DataFrame, columns: List[str], method: str = "iqr"
    ) -> pd.DataFrame:
        """Remove outliers from specified columns.

        Args:
            df: Input DataFrame.
            columns: Columns to check for outliers.
            method: Outlier detection method ('iqr', 'zscore').

        Returns:
            DataFrame with outliers removed.
        """
        raise NotImplementedError
