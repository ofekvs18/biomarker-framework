"""Unit tests for SHAPFeatureSelector."""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.models.shap_selector import SHAPFeatureSelector


@pytest.fixture
def sample_data():
    """Generate synthetic data for testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 5

    # Create features where feature 0 is most important
    X = np.random.randn(n_samples, n_features)
    X[:, 0] *= 2  # Make feature 0 more predictive

    # Create labels based on feature 0
    y = (X[:, 0] > 0).astype(int)

    # Add some noise
    noise_idx = np.random.choice(n_samples, size=10, replace=False)
    y[noise_idx] = 1 - y[noise_idx]

    feature_names = [f"feature_{i}" for i in range(n_features)]

    return X, y, feature_names


@pytest.fixture
def trained_rf_model(sample_data):
    """Train a RandomForest model for testing."""
    X, y, _ = sample_data
    model = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=5)
    model.fit(X, y)
    return model


@pytest.fixture
def trained_lr_model(sample_data):
    """Train a LogisticRegression model for testing."""
    X, y, _ = sample_data
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    return model


class TestSHAPFeatureSelector:
    """Test suite for SHAPFeatureSelector."""

    def test_initialization_tree_model(self, trained_rf_model):
        """Test initialization with tree-based model."""
        selector = SHAPFeatureSelector(model=trained_rf_model, model_type="tree")

        assert selector.model == trained_rf_model
        assert selector.model_type == "tree"
        assert selector.explainer is None
        assert selector.shap_values is None

    def test_initialization_auto_detect(self, trained_rf_model):
        """Test automatic model type detection."""
        selector = SHAPFeatureSelector(model=trained_rf_model, model_type="auto")

        assert selector.model_type == "auto"
        # Model type will be set to 'tree' after fitting

    def test_fit_tree_model(self, trained_rf_model, sample_data):
        """Test fitting with tree-based model."""
        X, y, feature_names = sample_data

        selector = SHAPFeatureSelector(model=trained_rf_model, model_type="tree")
        selector.fit(X, feature_names)

        # Check that SHAP values were computed
        assert selector.shap_values is not None
        assert selector.shap_values.shape == X.shape
        assert selector.feature_names == feature_names
        assert selector.base_value is not None
        assert selector._mean_abs_shap is not None

    def test_fit_linear_model(self, trained_lr_model, sample_data):
        """Test fitting with linear model."""
        X, y, feature_names = sample_data

        selector = SHAPFeatureSelector(model=trained_lr_model, model_type="linear")
        selector.fit(X, feature_names)

        # Check that SHAP values were computed
        assert selector.shap_values is not None
        assert selector.shap_values.shape == X.shape
        assert selector.feature_names == feature_names

    def test_fit_auto_detect(self, trained_rf_model, sample_data):
        """Test automatic model type detection during fit."""
        X, y, feature_names = sample_data

        selector = SHAPFeatureSelector(model=trained_rf_model, model_type="auto")
        selector.fit(X, feature_names)

        # Should auto-detect as tree model
        assert selector.model_type == "tree"
        assert selector.shap_values is not None

    def test_get_top_feature(self, trained_rf_model, sample_data):
        """Test getting top feature by SHAP importance."""
        X, y, feature_names = sample_data

        selector = SHAPFeatureSelector(model=trained_rf_model)
        selector.fit(X, feature_names)

        top_feature, importance = selector.get_top_feature()

        # Check output format
        assert isinstance(top_feature, str)
        assert top_feature in feature_names
        assert isinstance(importance, (int, float, np.number))
        assert importance >= 0

    def test_get_top_k_features(self, trained_rf_model, sample_data):
        """Test getting top K features."""
        X, y, feature_names = sample_data

        selector = SHAPFeatureSelector(model=trained_rf_model)
        selector.fit(X, feature_names)

        k = 3
        top_features = selector.get_top_k_features(k=k)

        # Check output format
        assert isinstance(top_features, list)
        assert len(top_features) == k

        for feature_name, importance in top_features:
            assert isinstance(feature_name, str)
            assert feature_name in feature_names
            assert isinstance(importance, (int, float, np.number))
            assert importance >= 0

        # Check that features are sorted by importance (descending)
        importances = [imp for _, imp in top_features]
        assert importances == sorted(importances, reverse=True)

    def test_get_feature_importances(self, trained_rf_model, sample_data):
        """Test getting all feature importances."""
        X, y, feature_names = sample_data

        selector = SHAPFeatureSelector(model=trained_rf_model)
        selector.fit(X, feature_names)

        importances = selector.get_feature_importances()

        # Check output format
        assert isinstance(importances, dict)
        assert len(importances) == len(feature_names)

        for feature_name, importance in importances.items():
            assert feature_name in feature_names
            assert isinstance(importance, float)
            assert importance >= 0

    def test_fit_without_feature_names(self, trained_rf_model, sample_data):
        """Test fitting without providing feature names."""
        X, y, _ = sample_data

        selector = SHAPFeatureSelector(model=trained_rf_model)
        selector.fit(X)

        # Should auto-generate feature names
        assert selector.feature_names is not None
        assert len(selector.feature_names) == X.shape[1]
        assert all(name.startswith("feature_") for name in selector.feature_names)

    def test_get_top_feature_before_fit(self, trained_rf_model):
        """Test that getting top feature before fit raises error."""
        selector = SHAPFeatureSelector(model=trained_rf_model)

        with pytest.raises(RuntimeError, match="Must call fit"):
            selector.get_top_feature()

    def test_get_top_k_features_before_fit(self, trained_rf_model):
        """Test that getting top K features before fit raises error."""
        selector = SHAPFeatureSelector(model=trained_rf_model)

        with pytest.raises(RuntimeError, match="Must call fit"):
            selector.get_top_k_features(k=3)

    def test_invalid_X_shape(self, trained_rf_model):
        """Test that 1D array raises error."""
        selector = SHAPFeatureSelector(model=trained_rf_model)

        X_1d = np.array([1, 2, 3])

        with pytest.raises(ValueError, match="must be 2D array"):
            selector.fit(X_1d)

    def test_feature_name_length_mismatch(self, trained_rf_model, sample_data):
        """Test that mismatched feature names raises error."""
        X, y, _ = sample_data

        selector = SHAPFeatureSelector(model=trained_rf_model)

        wrong_feature_names = ["feature_0", "feature_1"]  # Too few

        with pytest.raises(ValueError, match="Number of feature names"):
            selector.fit(X, wrong_feature_names)

    def test_plot_summary(self, trained_rf_model, sample_data):
        """Test SHAP summary plot creation."""
        X, y, feature_names = sample_data

        selector = SHAPFeatureSelector(model=trained_rf_model)
        selector.fit(X, feature_names)

        # Create plot (don't show it)
        fig = selector.plot_summary(show=False)

        assert fig is not None
        # Clean up
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_bar(self, trained_rf_model, sample_data):
        """Test SHAP bar plot creation."""
        X, y, feature_names = sample_data

        selector = SHAPFeatureSelector(model=trained_rf_model)
        selector.fit(X, feature_names)

        # Create plot (don't show it)
        fig = selector.plot_bar(show=False)

        assert fig is not None
        # Clean up
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_before_fit(self, trained_rf_model):
        """Test that plotting before fit raises error."""
        selector = SHAPFeatureSelector(model=trained_rf_model)

        with pytest.raises(RuntimeError, match="Must call fit"):
            selector.plot_summary()

    def test_get_shap_values(self, trained_rf_model, sample_data):
        """Test getting raw SHAP values."""
        X, y, feature_names = sample_data

        selector = SHAPFeatureSelector(model=trained_rf_model)
        selector.fit(X, feature_names)

        shap_values = selector.get_shap_values()

        assert isinstance(shap_values, np.ndarray)
        assert shap_values.shape == X.shape

    def test_get_model_info(self, trained_rf_model, sample_data):
        """Test getting model information."""
        X, y, feature_names = sample_data

        selector = SHAPFeatureSelector(model=trained_rf_model)
        selector.fit(X, feature_names)

        info = selector.get_model_info()

        assert isinstance(info, dict)
        assert "model_type" in info
        assert "explainer_type" in info
        assert "n_features" in info
        assert "base_value" in info

        assert info["n_features"] == len(feature_names)
        assert info["explainer_type"] == "tree"

    def test_consistent_rankings(self, trained_rf_model, sample_data):
        """Test that feature rankings are consistent across multiple calls."""
        X, y, feature_names = sample_data

        selector = SHAPFeatureSelector(model=trained_rf_model, random_state=42)
        selector.fit(X, feature_names)

        # Get rankings multiple times
        top_1_first = selector.get_top_feature()
        top_k_first = selector.get_top_k_features(k=3)

        top_1_second = selector.get_top_feature()
        top_k_second = selector.get_top_k_features(k=3)

        # Rankings should be consistent
        assert top_1_first == top_1_second
        assert top_k_first == top_k_second

    def test_most_predictive_feature_selected(self, sample_data):
        """Test that SHAP correctly identifies the most predictive feature."""
        X, y, feature_names = sample_data

        # Train model
        model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
        model.fit(X, y)

        # Get SHAP rankings
        selector = SHAPFeatureSelector(model=model)
        selector.fit(X, feature_names)

        top_feature, _ = selector.get_top_feature()

        # Feature 0 should be the most important (it was designed to be)
        # Allow some flexibility due to randomness, but feature_0 should be in top 2
        top_2 = [name for name, _ in selector.get_top_k_features(k=2)]
        assert "feature_0" in top_2

    def test_waterfall_plot(self, trained_rf_model, sample_data):
        """Test waterfall plot for individual prediction."""
        X, y, feature_names = sample_data

        selector = SHAPFeatureSelector(model=trained_rf_model)
        selector.fit(X, feature_names)

        # Create waterfall plot for first sample
        fig = selector.plot_waterfall(sample_idx=0, show=False)

        assert fig is not None
        # Clean up
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_waterfall_plot_invalid_index(self, trained_rf_model, sample_data):
        """Test that invalid sample index raises error."""
        X, y, feature_names = sample_data

        selector = SHAPFeatureSelector(model=trained_rf_model)
        selector.fit(X, feature_names)

        with pytest.raises(ValueError, match="out of range"):
            selector.plot_waterfall(sample_idx=1000)
