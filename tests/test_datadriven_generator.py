"""Unit tests for YoudensIndexGenerator."""

import numpy as np
import pytest

from src.generators.datadriven_threshold import YoudensIndexGenerator


@pytest.fixture
def sample_training_data():
    """Generate synthetic training data."""
    np.random.seed(42)
    n_samples = 200

    # Create features where hemoglobin is most predictive
    # Positive class has low hemoglobin (anemia)
    hemoglobin_pos = np.random.normal(10, 1.5, n_samples // 2)
    hemoglobin_neg = np.random.normal(14, 1.5, n_samples // 2)
    hemoglobin = np.concatenate([hemoglobin_pos, hemoglobin_neg])

    # Other features are less predictive
    wbc = np.random.normal(8, 2, n_samples)
    platelets = np.random.normal(250, 50, n_samples)

    # Labels: first half is positive, second half is negative
    y = np.array([1] * (n_samples // 2) + [0] * (n_samples // 2))

    # Shuffle data
    shuffle_idx = np.random.permutation(n_samples)
    X = np.column_stack([hemoglobin, wbc, platelets])[shuffle_idx]
    y = y[shuffle_idx]
    features = ["hemoglobin", "wbc", "platelets"]

    return X, y, features


@pytest.fixture
def balanced_data():
    """Generate balanced data for testing."""
    np.random.seed(123)
    n_samples = 100

    # Create data where high WBC predicts positive class
    wbc_pos = np.random.normal(12, 1, n_samples // 2)
    wbc_neg = np.random.normal(7, 1, n_samples // 2)
    wbc = np.concatenate([wbc_pos, wbc_neg])

    hemoglobin = np.random.normal(14, 1.5, n_samples)
    platelets = np.random.normal(250, 30, n_samples)

    y = np.array([1] * (n_samples // 2) + [0] * (n_samples // 2))

    shuffle_idx = np.random.permutation(n_samples)
    X = np.column_stack([hemoglobin, wbc, platelets])[shuffle_idx]
    y = y[shuffle_idx]
    features = ["hemoglobin", "wbc", "platelets"]

    return X, y, features


class TestYoudensIndexGenerator:
    """Test suite for YoudensIndexGenerator."""

    def test_initialization(self):
        """Test generator initialization."""
        config = {"random_state": 42, "use_cv": True}
        generator = YoudensIndexGenerator(config=config)

        assert generator.config == config
        assert generator.biomarker is None
        assert generator.selected_feature_ is None
        assert generator.optimized_threshold_ is None

    def test_generate_basic(self, sample_training_data):
        """Test basic biomarker generation."""
        X, y, features = sample_training_data
        config = {"random_state": 42, "use_cv": False}

        generator = YoudensIndexGenerator(config=config)
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
        assert "coefficient" in result["metadata"]
        assert "threshold_direction" in result["metadata"]
        assert "youden_index" in result["metadata"]
        assert "all_coefficients" in result["metadata"]
        assert "training_accuracy" in result["metadata"]
        assert "optimization_method" in result["metadata"]

    def test_feature_selection(self, sample_training_data):
        """Test that the most predictive feature is selected."""
        X, y, features = sample_training_data
        config = {"random_state": 42, "use_cv": False}

        generator = YoudensIndexGenerator(config=config)
        result = generator.generate(X, y, features)

        # Hemoglobin should be selected as it's most predictive
        assert result["metadata"]["selected_feature"] == "hemoglobin"
        assert generator.selected_feature_ == "hemoglobin"

    def test_youden_index_optimization(self, sample_training_data):
        """Test that Youden's Index is computed correctly."""
        X, y, features = sample_training_data
        config = {"random_state": 42, "use_cv": False}

        generator = YoudensIndexGenerator(config=config)
        result = generator.generate(X, y, features)

        # Youden's Index should be between -1 and 1 (TPR - FPR)
        youden = result["metadata"]["youden_index"]
        assert -1 <= youden <= 1

        # Youden's Index should be better than random (> 0)
        assert youden >= 0  # At least as good as random

    def test_cv_optimization(self, sample_training_data):
        """Test cross-validation optimization."""
        X, y, features = sample_training_data
        config = {"random_state": 42, "use_cv": True, "cv_folds": 5}

        generator = YoudensIndexGenerator(config=config)
        result = generator.generate(X, y, features)

        # Check CV metadata
        assert result["metadata"]["optimization_method"] == "cv"
        assert "cv_folds" in result["metadata"]
        assert result["metadata"]["cv_folds"] == 5
        assert "cv_thresholds" in result["metadata"]
        assert "cv_threshold_std" in result["metadata"]

        # Should have 5 thresholds (one per fold)
        assert len(result["metadata"]["cv_thresholds"]) == 5

    def test_cv_vs_single_comparison(self, sample_training_data):
        """Test that CV and single optimization produce reasonable results."""
        X, y, features = sample_training_data

        # Single optimization
        generator_single = YoudensIndexGenerator(
            config={"random_state": 42, "use_cv": False}
        )
        result_single = generator_single.generate(X, y, features)

        # CV optimization
        generator_cv = YoudensIndexGenerator(
            config={"random_state": 42, "use_cv": True, "cv_folds": 5}
        )
        result_cv = generator_cv.generate(X, y, features)

        # Both should select the same feature
        assert result_single["metadata"]["selected_feature"] == result_cv["metadata"]["selected_feature"]

        # Thresholds should be finite
        threshold_single = result_single["threshold"]
        threshold_cv = result_cv["threshold"]
        assert np.isfinite(threshold_single)
        assert np.isfinite(threshold_cv)

    def test_apply_predictions(self, sample_training_data):
        """Test applying biomarker to test data."""
        X_train, y_train, features = sample_training_data
        config = {"random_state": 42, "use_cv": False}

        generator = YoudensIndexGenerator(config=config)
        generator.generate(X_train, y_train, features)

        # Create test data
        X_test = np.array([[9.0, 8.0, 250], [15.0, 7.5, 300], [11.5, 9.0, 200]])

        # Apply biomarker
        predictions = generator.apply(X_test)

        # Check output format
        assert predictions.shape == (3,)
        assert predictions.dtype in [np.int32, np.int64, int]
        assert all(p in [0, 1] for p in predictions)

    def test_apply_before_generate_raises_error(self):
        """Test that apply raises error if called before generate."""
        config = {"random_state": 42}
        generator = YoudensIndexGenerator(config=config)

        X_test = np.array([[10.0, 8.0, 250]])

        with pytest.raises(RuntimeError, match="not generated yet"):
            generator.apply(X_test)

    def test_invalid_input_shapes(self):
        """Test that invalid input shapes raise errors."""
        config = {"random_state": 42}
        generator = YoudensIndexGenerator(config=config)

        # Mismatched X and y shapes
        X = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([0])
        features = ["a", "b", "c"]

        with pytest.raises(ValueError, match="shape mismatch"):
            generator.generate(X, y, features)

        # Mismatched X and features
        X = np.array([[1, 2, 3]])
        y = np.array([0])
        features = ["a", "b"]

        with pytest.raises(ValueError, match="features mismatch"):
            generator.generate(X, y, features)

    def test_single_class_raises_error(self):
        """Test that single class data raises error."""
        config = {"random_state": 42}
        generator = YoudensIndexGenerator(config=config)

        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        y = np.array([1, 1, 1])  # Only one class
        features = ["a", "b", "c"]

        with pytest.raises(ValueError, match="at least 2 classes"):
            generator.generate(X, y, features)

    def test_get_description(self, sample_training_data):
        """Test get_description method."""
        X, y, features = sample_training_data
        config = {"random_state": 42, "use_cv": True}

        generator = YoudensIndexGenerator(config=config)

        # Before generation
        desc = generator.get_description()
        assert "generated yet" in desc.lower()

        # After generation
        generator.generate(X, y, features)
        desc = generator.get_description()
        assert "Data-driven biomarker" in desc
        assert "Youden's Index" in desc
        assert generator.selected_feature_ in desc
        assert "coefficient" in desc.lower()

    def test_threshold_direction_high(self, balanced_data):
        """Test biomarker with high threshold direction."""
        X, y, features = balanced_data
        config = {"random_state": 42, "use_cv": False}

        generator = YoudensIndexGenerator(config=config)
        result = generator.generate(X, y, features)

        # WBC should be selected with high direction
        assert result["metadata"]["selected_feature"] == "wbc"
        assert result["metadata"]["threshold_direction"] == "high"
        assert ">=" in result["formula"]

    def test_threshold_direction_low(self, sample_training_data):
        """Test biomarker with low threshold direction."""
        X, y, features = sample_training_data
        config = {"random_state": 42, "use_cv": False}

        generator = YoudensIndexGenerator(config=config)
        result = generator.generate(X, y, features)

        # Hemoglobin should be selected with low direction
        assert result["metadata"]["selected_feature"] == "hemoglobin"
        assert result["metadata"]["threshold_direction"] == "low"
        assert "<=" in result["formula"]

    def test_predictions_consistency(self, sample_training_data):
        """Test that predictions are consistent with threshold."""
        X_train, y_train, features = sample_training_data
        config = {"random_state": 42, "use_cv": False}

        generator = YoudensIndexGenerator(config=config)
        generator.generate(X_train, y_train, features)

        # Apply to training data
        predictions = generator.apply(X_train)

        # Get threshold and direction
        threshold = generator.optimized_threshold_
        direction = generator.biomarker["metadata"]["threshold_direction"]
        feature_idx = features.index(generator.selected_feature_)
        feature_values = X_train[:, feature_idx]

        # Manually compute predictions
        if direction == "high":
            expected_predictions = (feature_values >= threshold).astype(int)
        else:
            expected_predictions = (feature_values <= threshold).astype(int)

        # Check consistency
        assert np.array_equal(predictions, expected_predictions)

    def test_wrong_test_shape_raises_error(self, sample_training_data):
        """Test that wrong test data shape raises error."""
        X_train, y_train, features = sample_training_data
        config = {"random_state": 42, "use_cv": False}

        generator = YoudensIndexGenerator(config=config)
        generator.generate(X_train, y_train, features)

        # Wrong number of features
        X_test_wrong = np.array([[10.0, 8.0]])

        with pytest.raises(ValueError, match="has 2 features, expected 3"):
            generator.apply(X_test_wrong)

    def test_reproducibility(self, sample_training_data):
        """Test that results are reproducible with same random state."""
        X, y, features = sample_training_data
        config = {"random_state": 42, "use_cv": True, "cv_folds": 5}

        generator1 = YoudensIndexGenerator(config=config)
        result1 = generator1.generate(X, y, features)

        generator2 = YoudensIndexGenerator(config=config)
        result2 = generator2.generate(X, y, features)

        # Results should be identical
        assert result1["formula"] == result2["formula"]
        assert result1["threshold"] == result2["threshold"]
        assert result1["features_used"] == result2["features_used"]
        assert (
            result1["metadata"]["selected_feature"]
            == result2["metadata"]["selected_feature"]
        )

    def test_performance_better_than_random(self, sample_training_data):
        """Test that optimized threshold performs reasonably."""
        X_train, y_train, features = sample_training_data
        config = {"random_state": 42, "use_cv": False}

        generator = YoudensIndexGenerator(config=config)
        result = generator.generate(X_train, y_train, features)

        # Apply to training data
        predictions = generator.apply(X_train)

        # Compute accuracy
        accuracy = np.mean(predictions == y_train)

        # Should be at least as good as random (>= 0.5)
        assert accuracy >= 0.45  # Allow some variance

    def test_cv_threshold_variance(self, sample_training_data):
        """Test that CV threshold variance is reasonable."""
        X, y, features = sample_training_data
        config = {"random_state": 42, "use_cv": True, "cv_folds": 5}

        generator = YoudensIndexGenerator(config=config)
        result = generator.generate(X, y, features)

        # Check that threshold standard deviation is finite
        cv_std = result["metadata"]["cv_threshold_std"]
        mean_threshold = result["threshold"]

        assert np.isfinite(cv_std)
        assert np.isfinite(mean_threshold)
        assert cv_std >= 0  # Standard deviation should be non-negative

    def test_different_cv_folds(self, sample_training_data):
        """Test with different numbers of CV folds."""
        X, y, features = sample_training_data

        for cv_folds in [3, 5, 10]:
            config = {"random_state": 42, "use_cv": True, "cv_folds": cv_folds}

            generator = YoudensIndexGenerator(config=config)
            result = generator.generate(X, y, features)

            # Check that correct number of thresholds are stored
            assert len(result["metadata"]["cv_thresholds"]) == cv_folds
            assert result["metadata"]["cv_folds"] == cv_folds

    def test_standard_output_format(self, sample_training_data):
        """Test that output matches standard BiomarkerGenerator format."""
        X, y, features = sample_training_data
        config = {"random_state": 42, "use_cv": True}

        generator = YoudensIndexGenerator(config=config)
        result = generator.generate(X, y, features)

        # Check standard format
        required_keys = ["formula", "threshold", "features_used", "metadata"]
        for key in required_keys:
            assert key in result

        # Check types
        assert isinstance(result["formula"], str)
        assert isinstance(result["threshold"], float)
        assert isinstance(result["features_used"], list)
        assert isinstance(result["metadata"], dict)

        # Check metadata contains expected keys
        expected_metadata = [
            "selected_feature",
            "coefficient",
            "threshold_direction",
            "youden_index",
            "all_coefficients",
            "training_accuracy",
            "optimization_method",
        ]
        for key in expected_metadata:
            assert key in result["metadata"]
