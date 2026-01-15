"""Abstract base class for biomarker generation strategies."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import pandas as pd


class BaseBiomarkerGenerator(ABC):
    """Abstract base class for generating biomarker labels.

    Different strategies can be implemented by subclassing:
    - Literature-based thresholds (known clinical ranges)
    - Data-driven thresholds (percentiles, clustering)
    - Hybrid approaches
    """

    def __init__(self, feature_config: Optional[Dict] = None):
        """Initialize the generator.

        Args:
            feature_config: Configuration for feature thresholds.
        """
        self.feature_config = feature_config or {}
        self.thresholds: Dict[str, Dict] = {}

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "BaseBiomarkerGenerator":
        """Fit the generator to learn thresholds from data.

        Args:
            df: Training data with CBC features.

        Returns:
            Fitted generator instance.
        """
        pass

    @abstractmethod
    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate binary biomarker labels based on thresholds.

        Args:
            df: DataFrame with CBC features.

        Returns:
            DataFrame with binary biomarker columns.
        """
        pass

    @abstractmethod
    def get_thresholds(self) -> Dict[str, Dict]:
        """Return the thresholds used for label generation.

        Returns:
            Dictionary mapping feature names to threshold values.
        """
        pass

    def get_feature_names(self) -> List[str]:
        """Get list of features this generator handles.

        Returns:
            List of feature names.
        """
        return list(self.feature_config.keys())
