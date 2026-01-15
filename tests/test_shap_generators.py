"""Unit tests for SHAP-based biomarker generators."""

import numpy as np
import pytest

from src.generators.shap_datadriven_threshold import SHAPDataDrivenThresholdGenerator
from src.generators.shap_literature_threshold import SHAPLiteratureThresholdGenerator


@pytest.fixture
def sample_literature_thresholds():
    """Sample literature thresholds for testing."""
    return {
        "hemoglobin": {"low": 12.0, "high": 17.5},
        "wbc": {"low": 4.5, "high": 11.0},
        "platelets": {"low": 150, "high": 400},
    }


@pytest.fixture
def simple_thresholds():
    """Simple float thresholds for testing."""
    return {"hemoglobin": 12.0, "wbc": 11.0, "platelets": 150}


@pytest.fixture
def sample_training_data():
    """Generate synthetic training data."""
    np.random.seed(42)
    n_samples = 200  # Larger sample for SHAP

    # Create features where hemoglobin is most predictive
    hemoglobin = np.random.normal(13, 2, n_samples)
    wbc = np.random.normal(8, 1.5, n_samples)
    platelets = np.random.normal(250, 50, n_samples)

    # Create labels based on hemoglobin (anemia predicts positive class)
    y = (hemoglobin < 12.0).astype(int)

    # Add some noise
    noise_idx = np.random.choice(n_samples, size=20, replace=False)
    y[noise_idx] = 1 - y[noise_idx]

    X = np.column_stack([hemoglobin, wbc, platelets])
    features = ["hemoglobin", "wbc", "platelets"]

    return X, y, features


@pytest.fixture
def sample_test_data():
    """Generate synthetic test data."""
    np.random.seed(123)  # Different seed than training
    n_samples = 50

    hemoglobin = np.random.normal(13, 2, n_samples)
    wbc = np.random.normal(8, 1.5, n_samples)
    platelets = np.random.normal(250, 50, n_samples)

    y = (hemoglobin < 12.0).astype(int)

    X = np.column_stack([hemoglobin, wbc, platelets])

    return X, y


class TestSHAPLiteratureThresholdGenerator:
    """Test suite for SHAPLiteratureThresholdGenerator."""

    def test_initialization(self, sample_literature_thresholds):
        """Test generator initialization."""
        config = {"random_state": 42, "model_type": "RandomForest"}
        generator = SHAPLiteratureThresholdGenerator(
            config=config, literature_thresholds=sample_literature_thresholds
        )

        assert generator.config == config
        assert generator.literature_thresholds == sample_literature_thresholds
        assert generator.biomarker is None
        assert generator.selected_feature_ is None

    def test_generate_basic(self, sample_training_data, sample_literature_thresholds):
        """Test basic biomarker generation."""
        X, y, features = sample_training_data
        config = {"random_state": 42, "model_type": "RandomForest", "n_estimators": 10}

        generator = SHAPLiteratureThresholdGenerator(
            config=config, literature_thresholds=sample_literature_thresholds
        )

        result = generator.generate(X, y, features)

        # Check output format
        assert "formula" in result
        assert "threshold" in result
        assert "features_used" in result
        assert "metadata" in result

        # Check that a feature was selected
        assert len(result["features_used"]) == 1
        assert result["features_used"][0] in features

        # Check metadata
        assert "selected_feature" in result["metadata"]
        assert "shap_importance" in result["metadata"]
        assert "mean_shap_value" in result["metadata"]
        assert "threshold_direction" in result["metadata"]
        assert "all_shap_importances" in result["metadata"]
        assert "top_5_features" in result["metadata"]
        assert "model_type" in result["metadata"]
        assert "shap_explainer_type" in result["metadata"]
        assert "training_accuracy" in result["metadata"]

    def test_apply(self, sample_training_data, sample_test_data, sample_literature_thresholds):
        """Test applying biomarker to test data."""
        X_train, y_train, features = sample_training_data
        X_test, y_test = sample_test_data

        config = {"random_state": 42, "model_type": "RandomForest", "n_estimators": 10}
        generator = SHAPLiteratureThresholdGenerator(
            config=config, literature_thresholds=sample_literature_thresholds
        )

        # Generate biomarker
        generator.generate(X_train, y_train, features)

        # Apply to test data
        predictions = generator.apply(X_test)

        # Check predictions
        assert predictions.shape == (len(X_test),)
        assert predictions.dtype == int
        assert set(predictions).issubset({0, 1})

    def test_apply_before_generate(self, sample_test_data, sample_literature_thresholds):
        """Test that apply before generate raises error."""
        X_test, y_test = sample_test_data

        config = {"random_state": 42}
        generator = SHAPLiteratureThresholdGenerator(
            config=config, literature_thresholds=sample_literature_thresholds
        )

        with pytest.raises(RuntimeError, match="not generated yet"):
            generator.apply(X_test)

    def test_get_description(self, sample_training_data, sample_literature_thresholds):
        """Test getting biomarker description."""
        X, y, features = sample_training_data

        config = {"random_state": 42, "model_type": "RandomForest", "n_estimators": 10}
        generator = SHAPLiteratureThresholdGenerator(
            config=config, literature_thresholds=sample_literature_thresholds
        )

        # Before generation
        desc_before = generator.get_description()
        assert "No biomarker generated" in desc_before

        # After generation
        generator.generate(X, y, features)
        desc_after = generator.get_description()

        assert "SHAP-based literature biomarker" in desc_after
        assert "Selected feature" in desc_after
        assert "SHAP importance" in desc_after

    def test_get_shap_plots(self, sample_training_data, sample_literature_thresholds):
        """Test getting SHAP plots."""
        X, y, features = sample_training_data

        config = {"random_state": 42, "model_type": "RandomForest", "n_estimators": 10}
        generator = SHAPLiteratureThresholdGenerator(
            config=config, literature_thresholds=sample_literature_thresholds
        )

        generator.generate(X, y, features)

        # Get SHAP plots
        plots = generator.get_shap_plots(max_display=10)

        assert "summary" in plots
        assert "bar" in plots
        assert plots["summary"] is not None
        assert plots["bar"] is not None

        # Clean up
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_simple_thresholds(self, sample_training_data, simple_thresholds):
        """Test with simple float thresholds."""
        X, y, features = sample_training_data

        config = {"random_state": 42, "model_type": "RandomForest", "n_estimators": 10}
        generator = SHAPLiteratureThresholdGenerator(
            config=config, literature_thresholds=simple_thresholds
        )

        result = generator.generate(X, y, features)

        assert "formula" in result
        assert "threshold" in result
        assert result["threshold"] in simple_thresholds.values()

    def test_missing_threshold(self, sample_training_data):
        """Test behavior when threshold is missing for selected feature."""
        X, y, features = sample_training_data

        # Provide threshold for only one feature
        incomplete_thresholds = {"wbc": 11.0}

        config = {"random_state": 42, "model_type": "RandomForest", "n_estimators": 10}
        generator = SHAPLiteratureThresholdGenerator(
            config=config, literature_thresholds=incomplete_thresholds
        )

        # This should raise error if selected feature has no threshold
        # (which is likely since hemoglobin is most predictive)
        with pytest.raises(ValueError, match="No literature threshold found"):
            generator.generate(X, y, features)

    def test_input_validation(self, sample_literature_thresholds):
        """Test input validation in generate method."""
        config = {"random_state": 42}
        generator = SHAPLiteratureThresholdGenerator(
            config=config, literature_thresholds=sample_literature_thresholds
        )

        # Mismatched X and y shapes
        X = np.random.randn(100, 3)
        y = np.random.randint(0, 2, 50)  # Wrong length
        features = ["f1", "f2", "f3"]

        with pytest.raises(ValueError, match="shape mismatch"):
            generator.generate(X, y, features)

        # Mismatched features length
        X = np.random.randn(100, 3)
        y = np.random.randint(0, 2, 100)
        features = ["f1", "f2"]  # Wrong length

        with pytest.raises(ValueError, match="features mismatch"):
            generator.generate(X, y, features)


class TestSHAPDataDrivenThresholdGenerator:
    """Test suite for SHAPDataDrivenThresholdGenerator."""

    def test_initialization(self):
        """Test generator initialization."""
        config = {"random_state": 42, "model_type": "RandomForest", "use_cv": True}
        generator = SHAPDataDrivenThresholdGenerator(config=config)

        assert generator.config == config
        assert generator.biomarker is None
        assert generator.selected_feature_ is None
        assert generator.optimized_threshold_ is None

    def test_generate_basic(self, sample_training_data):
        """Test basic biomarker generation."""
        X, y, features = sample_training_data

        config = {
            "random_state": 42,
            "model_type": "RandomForest",
            "n_estimators": 10,
            "use_cv": False,  # Faster for testing
        }
        generator = SHAPDataDrivenThresholdGenerator(config=config)

        result = generator.generate(X, y, features)

        # Check output format
        assert "formula" in result
        assert "threshold" in result
        assert "features_used" in result
        assert "metadata" in result

        # Check that a feature was selected
        assert len(result["features_used"]) == 1
        assert result["features_used"][0] in features

        # Check metadata
        assert "selected_feature" in result["metadata"]
        assert "shap_importance" in result["metadata"]
        assert "youden_index" in result["metadata"]
        assert "threshold_direction" in result["metadata"]
        assert "optimization_method" in result["metadata"]

    def test_generate_with_cv(self, sample_training_data):
        """Test biomarker generation with cross-validation."""
        X, y, features = sample_training_data

        config = {
            "random_state": 42,
            "model_type": "RandomForest",
            "n_estimators": 10,
            "use_cv": True,
            "cv_folds": 3,  # Small number for speed
        }
        generator = SHAPDataDrivenThresholdGenerator(config=config)

        result = generator.generate(X, y, features)

        # Check CV-specific metadata
        assert "cv_folds" in result["metadata"]
        assert "cv_thresholds" in result["metadata"]
        assert "cv_threshold_std" in result["metadata"]

        assert result["metadata"]["cv_folds"] == 3
        assert len(result["metadata"]["cv_thresholds"]) == 3

    def test_apply(self, sample_training_data, sample_test_data):
        """Test applying biomarker to test data."""
        X_train, y_train, features = sample_training_data
        X_test, y_test = sample_test_data

        config = {
            "random_state": 42,
            "model_type": "RandomForest",
            "n_estimators": 10,
            "use_cv": False,
        }
        generator = SHAPDataDrivenThresholdGenerator(config=config)

        # Generate biomarker
        generator.generate(X_train, y_train, features)

        # Apply to test data
        predictions = generator.apply(X_test)

        # Check predictions
        assert predictions.shape == (len(X_test),)
        assert predictions.dtype == int
        assert set(predictions).issubset({0, 1})

    def test_apply_before_generate(self, sample_test_data):
        """Test that apply before generate raises error."""
        X_test, y_test = sample_test_data

        config = {"random_state": 42}
        generator = SHAPDataDrivenThresholdGenerator(config=config)

        with pytest.raises(RuntimeError, match="not generated yet"):
            generator.apply(X_test)

    def test_get_description(self, sample_training_data):
        """Test getting biomarker description."""
        X, y, features = sample_training_data

        config = {
            "random_state": 42,
            "model_type": "RandomForest",
            "n_estimators": 10,
            "use_cv": True,
            "cv_folds": 3,
        }
        generator = SHAPDataDrivenThresholdGenerator(config=config)

        # Before generation
        desc_before = generator.get_description()
        assert "No biomarker generated" in desc_before

        # After generation
        generator.generate(X, y, features)
        desc_after = generator.get_description()

        assert "SHAP-based data-driven biomarker" in desc_after
        assert "Youden's Index" in desc_after
        assert "SHAP importance" in desc_after
        assert "threshold std" in desc_after  # CV-specific

    def test_get_shap_plots(self, sample_training_data):
        """Test getting SHAP plots."""
        X, y, features = sample_training_data

        config = {
            "random_state": 42,
            "model_type": "RandomForest",
            "n_estimators": 10,
            "use_cv": False,
        }
        generator = SHAPDataDrivenThresholdGenerator(config=config)

        generator.generate(X, y, features)

        # Get SHAP plots
        plots = generator.get_shap_plots(max_display=10)

        assert "summary" in plots
        assert "bar" in plots
        assert plots["summary"] is not None
        assert plots["bar"] is not None

        # Clean up
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_threshold_optimization(self, sample_training_data):
        """Test that threshold is optimized (different from literature)."""
        X, y, features = sample_training_data

        config = {
            "random_state": 42,
            "model_type": "RandomForest",
            "n_estimators": 10,
            "use_cv": False,
        }
        generator = SHAPDataDrivenThresholdGenerator(config=config)

        result = generator.generate(X, y, features)

        # Threshold should be optimized from data, not a fixed literature value
        threshold = result["threshold"]
        assert isinstance(threshold, (int, float))

        # Youden's Index should be reasonable
        youden = result["metadata"]["youden_index"]
        assert -1 <= youden <= 1  # Valid range for Youden's Index

    def test_input_validation(self):
        """Test input validation in generate method."""
        config = {"random_state": 42}
        generator = SHAPDataDrivenThresholdGenerator(config=config)

        # Mismatched X and y shapes
        X = np.random.randn(100, 3)
        y = np.random.randint(0, 2, 50)  # Wrong length
        features = ["f1", "f2", "f3"]

        with pytest.raises(ValueError, match="shape mismatch"):
            generator.generate(X, y, features)

        # Single class in y
        X = np.random.randn(100, 3)
        y = np.ones(100)  # All same class
        features = ["f1", "f2", "f3"]

        with pytest.raises(ValueError, match="at least 2 classes"):
            generator.generate(X, y, features)

    def test_different_model_types(self, sample_training_data):
        """Test with different model types."""
        X, y, features = sample_training_data

        model_types = ["RandomForest", "LogisticRegression"]

        for model_type in model_types:
            config = {
                "random_state": 42,
                "model_type": model_type,
                "n_estimators": 10 if "Forest" in model_type else None,
                "use_cv": False,
            }
            generator = SHAPDataDrivenThresholdGenerator(config=config)

            result = generator.generate(X, y, features)

            assert "formula" in result
            assert result["metadata"]["model_type"] == model_type

    def test_consistency_across_runs(self, sample_training_data):
        """Test that results are consistent with same random seed."""
        X, y, features = sample_training_data

        config1 = {
            "random_state": 42,
            "model_type": "RandomForest",
            "n_estimators": 10,
            "use_cv": False,
        }
        generator1 = SHAPDataDrivenThresholdGenerator(config=config1)
        result1 = generator1.generate(X, y, features)

        config2 = {
            "random_state": 42,
            "model_type": "RandomForest",
            "n_estimators": 10,
            "use_cv": False,
        }
        generator2 = SHAPDataDrivenThresholdGenerator(config=config2)
        result2 = generator2.generate(X, y, features)

        # Results should be identical with same seed
        assert result1["features_used"] == result2["features_used"]
        assert np.isclose(result1["threshold"], result2["threshold"])
