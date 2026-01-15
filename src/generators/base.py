"""Abstract base class for biomarker generation strategies."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
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


class BiomarkerGenerator(ABC):
    """Abstract base class for biomarker generation methods.

    This class defines the interface for all biomarker generation strategies,
    enabling interchangeable logic for different methods (e.g., statistical,
    machine learning, rule-based).

    The workflow is:
    1. Initialize with configuration
    2. Call generate() with training data to learn a biomarker formula/rule
    3. Call apply() on test data to get predictions
    4. Use get_description() to understand the learned biomarker

    Standard Output Format:
        The generate() method should return a dictionary with:
        - 'formula': str - Mathematical formula or rule description
        - 'threshold': float - Decision threshold for classification
        - 'features_used': list[str] - Features used in the biomarker
        - 'metadata': dict - Method-specific information (e.g., coefficients,
                            feature importances, training metrics)

    Example:
        >>> generator = MyBiomarkerGenerator(config={'param': 0.5})
        >>> result = generator.generate(X_train, y_train, features=['wbc', 'hgb'])
        >>> predictions = generator.apply(X_test)
        >>> print(generator.get_description())
    """

    def __init__(self, config: dict):
        """Initialize the biomarker generator.

        Args:
            config: Configuration dictionary with method-specific parameters.
                   Common parameters might include:
                   - 'regularization': float - Regularization strength
                   - 'max_features': int - Maximum number of features to use
                   - 'random_state': int - Random seed for reproducibility
        """
        self.config = config
        self.biomarker = None  # Generated biomarker formula/rule

    @abstractmethod
    def generate(
        self, X_train: np.ndarray, y_train: np.ndarray, features: List[str]
    ) -> Dict[str, Any]:
        """Generate biomarker from training data.

        This method learns a biomarker formula/rule from the training data
        and stores it internally for later application.

        Args:
            X_train: Training feature matrix of shape (n_samples, n_features)
            y_train: Training labels of shape (n_samples,)
            features: List of feature names corresponding to X_train columns

        Returns:
            Dictionary containing:
            - 'formula': str - Human-readable mathematical formula or rule
            - 'threshold': float - Decision threshold for binary classification
            - 'features_used': list[str] - Subset of features used in biomarker
            - 'metadata': dict - Additional method-specific information such as:
                - 'coefficients': Feature coefficients (for linear methods)
                - 'feature_importances': Feature importance scores
                - 'training_auc': Training set AUC score
                - 'n_iterations': Number of training iterations
                - Any other relevant information for reproducibility

        Raises:
            ValueError: If input data is invalid (e.g., mismatched shapes)
        """
        pass

    @abstractmethod
    def apply(self, X_test: np.ndarray) -> np.ndarray:
        """Apply learned biomarker to test data.

        Uses the biomarker formula/rule learned during generate() to
        produce predictions on new data.

        Args:
            X_test: Test feature matrix of shape (n_samples, n_features).
                   Must have same number of features as training data.

        Returns:
            Binary predictions of shape (n_samples,) where 1 indicates
            positive class (e.g., disease present) and 0 indicates negative
            class (e.g., disease absent).

        Raises:
            RuntimeError: If generate() has not been called yet
            ValueError: If X_test has wrong number of features
        """
        pass

    def get_description(self) -> str:
        """Return human-readable description of the learned biomarker.

        This provides a textual representation of the biomarker formula
        that can be used for interpretation and reporting.

        Returns:
            String description of the biomarker. If no biomarker has been
            generated yet, returns a message indicating this.

        Example:
            "Biomarker = 0.5 * WBC + 0.3 * HGB - 1.2, threshold = 0.0"
        """
        if self.biomarker is None:
            return "No biomarker generated yet. Call generate() first."
        return str(self.biomarker)
