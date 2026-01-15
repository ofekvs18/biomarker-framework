"""Unit tests for SingleFeatureLiteratureGenerator."""

import numpy as np
import pytest

from src.generators.literature_threshold import SingleFeatureLiteratureGenerator


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
    n_samples = 100

    # Create features where hemoglobin is most predictive
    hemoglobin = np.random.normal(13, 2, n_samples)
    wbc = np.random.normal(8, 1.5, n_samples)
    platelets = np.random.normal(250, 50, n_samples)

    # Create labels based on hemoglobin (anemia predicts positive class)
    y = (hemoglobin < 12.0).astype(int)

    # Add some noise
    noise_idx = np.random.choice(n_samples, size=10, replace=False)
    y[noise_idx] = 1 - y[noise_idx]

    X = np.column_stack([hemoglobin, wbc, platelets])
    features = ["hemoglobin", "wbc", "platelets"]

    return X, y, features


class TestSingleFeatureLiteratureGenerator:
    """Test suite for SingleFeatureLiteratureGenerator."""

    def test_initialization(self, sample_literature_thresholds):
        """Test generator initialization."""
        config = {"random_state": 42}
        generator = SingleFeatureLiteratureGenerator(
            config=config, literature_thresholds=sample_literature_thresholds
        )

        assert generator.config == config
        assert generator.literature_thresholds == sample_literature_thresholds
        assert generator.biomarker is None
        assert generator.selected_feature_ is None

    def test_generate_basic(self, sample_training_data, sample_literature_thresholds):
        """Test basic biomarker generation."""
        X, y, features = sample_training_data
        config = {"random_state": 42}

        generator = SingleFeatureLiteratureGenerator(
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
        assert "coefficient" in result["metadata"]
        assert "threshold_direction" in result["metadata"]
        assert "all_coefficients" in result["metadata"]
        assert "training_accuracy" in result["metadata"]

    def test_feature_selection(self, sample_training_data, sample_literature_thresholds):
        """Test that the most predictive feature is selected."""
        X, y, features = sample_training_data
        config = {"random_state": 42}

        generator = SingleFeatureLiteratureGenerator(
            config=config, literature_thresholds=sample_literature_thresholds
        )

        result = generator.generate(X, y, features)

        # Hemoglobin should be selected as it's most predictive
        # (data was generated with hemoglobin < 12 predicting positive class)
        assert result["metadata"]["selected_feature"] == "hemoglobin"
        assert generator.selected_feature_ == "hemoglobin"

    def test_threshold_application(
        self, sample_training_data, sample_literature_thresholds
    ):
        """Test that correct literature threshold is applied."""
        X, y, features = sample_training_data
        config = {"random_state": 42}

        generator = SingleFeatureLiteratureGenerator(
            config=config, literature_thresholds=sample_literature_thresholds
        )

        result = generator.generate(X, y, features)

        # Check that threshold matches literature value
        selected_feature = result["metadata"]["selected_feature"]
        assert result["threshold"] in [
            sample_literature_thresholds[selected_feature]["low"],
            sample_literature_thresholds[selected_feature]["high"],
        ]

    def test_apply_predictions(self, sample_training_data, sample_literature_thresholds):
        """Test applying biomarker to test data."""
        X_train, y_train, features = sample_training_data
        config = {"random_state": 42}

        generator = SingleFeatureLiteratureGenerator(
            config=config, literature_thresholds=sample_literature_thresholds
        )

        # Generate biomarker
        generator.generate(X_train, y_train, features)

        # Create test data
        X_test = np.array([[10.0, 8.0, 250], [15.0, 7.5, 300], [11.5, 9.0, 200]])

        # Apply biomarker
        predictions = generator.apply(X_test)

        # Check output format
        assert predictions.shape == (3,)
        assert predictions.dtype in [np.int32, np.int64, int]
        assert all(p in [0, 1] for p in predictions)

    def test_apply_before_generate_raises_error(self, sample_literature_thresholds):
        """Test that apply raises error if called before generate."""
        config = {"random_state": 42}
        generator = SingleFeatureLiteratureGenerator(
            config=config, literature_thresholds=sample_literature_thresholds
        )

        X_test = np.array([[10.0, 8.0, 250]])

        with pytest.raises(RuntimeError, match="not generated yet"):
            generator.apply(X_test)

    def test_invalid_input_shapes(self, sample_literature_thresholds):
        """Test that invalid input shapes raise errors."""
        config = {"random_state": 42}
        generator = SingleFeatureLiteratureGenerator(
            config=config, literature_thresholds=sample_literature_thresholds
        )

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

    def test_missing_threshold_raises_error(self, sample_training_data):
        """Test that missing threshold raises error."""
        X, y, features = sample_training_data
        config = {"random_state": 42}

        # Thresholds missing hemoglobin
        incomplete_thresholds = {"wbc": 11.0, "platelets": 150}

        generator = SingleFeatureLiteratureGenerator(
            config=config, literature_thresholds=incomplete_thresholds
        )

        with pytest.raises(ValueError, match="No literature threshold found"):
            generator.generate(X, y, features)

    def test_simple_float_thresholds(self, sample_training_data, simple_thresholds):
        """Test that simple float thresholds work correctly."""
        X, y, features = sample_training_data
        config = {"random_state": 42}

        generator = SingleFeatureLiteratureGenerator(
            config=config, literature_thresholds=simple_thresholds
        )

        result = generator.generate(X, y, features)

        # Should work with simple float thresholds
        assert result["threshold"] == simple_thresholds[result["metadata"]["selected_feature"]]

    def test_get_description(self, sample_training_data, sample_literature_thresholds):
        """Test get_description method."""
        X, y, features = sample_training_data
        config = {"random_state": 42}

        generator = SingleFeatureLiteratureGenerator(
            config=config, literature_thresholds=sample_literature_thresholds
        )

        # Before generation
        desc = generator.get_description()
        assert "generated yet" in desc.lower()

        # After generation
        generator.generate(X, y, features)
        desc = generator.get_description()
        assert "Literature-based biomarker" in desc
        assert generator.selected_feature_ in desc
        assert "coefficient" in desc.lower()

    def test_threshold_direction_high(self):
        """Test biomarker with high threshold direction."""
        np.random.seed(123)

        # Create data where high WBC predicts positive class
        wbc = np.random.normal(8, 2, 100)
        hemoglobin = np.random.normal(14, 1, 100)
        platelets = np.random.normal(250, 30, 100)

        y = (wbc > 11.0).astype(int)

        X = np.column_stack([hemoglobin, wbc, platelets])
        features = ["hemoglobin", "wbc", "platelets"]

        thresholds = {"hemoglobin": 12.0, "wbc": 11.0, "platelets": 150}

        generator = SingleFeatureLiteratureGenerator(
            config={"random_state": 42}, literature_thresholds=thresholds
        )

        result = generator.generate(X, y, features)

        # WBC should be selected with high direction
        assert result["metadata"]["selected_feature"] == "wbc"
        assert result["metadata"]["threshold_direction"] == "high"
        assert ">=" in result["formula"]

    def test_threshold_direction_low(self, sample_training_data, sample_literature_thresholds):
        """Test biomarker with low threshold direction."""
        X, y, features = sample_training_data
        config = {"random_state": 42}

        generator = SingleFeatureLiteratureGenerator(
            config=config, literature_thresholds=sample_literature_thresholds
        )

        result = generator.generate(X, y, features)

        # Hemoglobin should be selected with low direction
        # (data was generated with low hemoglobin predicting positive class)
        assert result["metadata"]["selected_feature"] == "hemoglobin"
        assert result["metadata"]["threshold_direction"] == "low"
        assert "<=" in result["formula"]

    def test_predictions_match_threshold(
        self, sample_training_data, sample_literature_thresholds
    ):
        """Test that predictions correctly apply the threshold."""
        X_train, y_train, features = sample_training_data
        config = {"random_state": 42}

        generator = SingleFeatureLiteratureGenerator(
            config=config, literature_thresholds=sample_literature_thresholds
        )

        generator.generate(X_train, y_train, features)

        # Create test data with known values
        # hemoglobin is selected, threshold is 12.0, direction is low (<=)
        X_test = np.array(
            [
                [10.0, 8.0, 250],  # hemoglobin < 12 -> should be 1
                [15.0, 7.5, 300],  # hemoglobin > 12 -> should be 0
                [12.0, 9.0, 200],  # hemoglobin = 12 -> should be 1 (<=)
            ]
        )

        predictions = generator.apply(X_test)

        # Verify predictions match expected threshold logic
        assert predictions[0] == 1  # 10.0 <= 12.0
        assert predictions[1] == 0  # 15.0 > 12.0
        assert predictions[2] == 1  # 12.0 <= 12.0

    def test_wrong_test_shape_raises_error(
        self, sample_training_data, sample_literature_thresholds
    ):
        """Test that wrong test data shape raises error."""
        X_train, y_train, features = sample_training_data
        config = {"random_state": 42}

        generator = SingleFeatureLiteratureGenerator(
            config=config, literature_thresholds=sample_literature_thresholds
        )

        generator.generate(X_train, y_train, features)

        # Wrong number of features
        X_test_wrong = np.array([[10.0, 8.0]])

        with pytest.raises(ValueError, match="has 2 features, expected 3"):
            generator.apply(X_test_wrong)

    def test_reproducibility(self, sample_training_data, sample_literature_thresholds):
        """Test that results are reproducible with same random state."""
        X, y, features = sample_training_data
        config = {"random_state": 42}

        generator1 = SingleFeatureLiteratureGenerator(
            config=config, literature_thresholds=sample_literature_thresholds
        )
        result1 = generator1.generate(X, y, features)

        generator2 = SingleFeatureLiteratureGenerator(
            config=config, literature_thresholds=sample_literature_thresholds
        )
        result2 = generator2.generate(X, y, features)

        # Results should be identical
        assert result1["formula"] == result2["formula"]
        assert result1["threshold"] == result2["threshold"]
        assert result1["features_used"] == result2["features_used"]
        assert (
            result1["metadata"]["selected_feature"]
            == result2["metadata"]["selected_feature"]
        )
