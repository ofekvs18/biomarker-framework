"""Tests for BiomarkerEvaluator class."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import pytest

from src.evaluation.evaluator import BiomarkerEvaluator
from src.generators.base import BaseBiomarkerGenerator


class MockBiomarkerGenerator(BaseBiomarkerGenerator):
    """Mock biomarker generator for testing."""

    def __init__(self, feature_config=None, method="simple"):
        """Initialize mock generator.

        Args:
            feature_config: Feature configuration dictionary.
            method: Method to use for generating labels ('simple' or 'random').
        """
        super().__init__(feature_config)
        self.method = method
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> "MockBiomarkerGenerator":
        """Mock fit method that sets some thresholds.

        Args:
            df: Training dataframe.

        Returns:
            Self for chaining.
        """
        # Set mock thresholds based on data
        for col in df.columns:
            if col in ["hemoglobin", "wbc", "platelets"]:
                self.thresholds[col] = {
                    "low": float(df[col].quantile(0.25)),
                    "high": float(df[col].quantile(0.75)),
                }
        self.is_fitted = True
        return self

    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mock generate_labels method.

        Args:
            df: Dataframe to generate labels for.

        Returns:
            DataFrame with biomarker labels.
        """
        if not self.is_fitted:
            raise ValueError("Generator must be fitted before generating labels")

        if self.method == "simple":
            # Simple method: return mean of normalized features
            result = pd.DataFrame(index=df.index)
            for col in df.columns:
                if col in self.thresholds:
                    # Normalize to 0-1 range based on thresholds
                    low = self.thresholds[col]["low"]
                    high = self.thresholds[col]["high"]
                    normalized = (df[col] - low) / (high - low) if high > low else 0.5
                    result[f"{col}_biomarker"] = normalized.clip(0, 1)
            return result
        elif self.method == "random":
            # Random method for testing
            result = pd.DataFrame(index=df.index)
            for col in df.columns:
                if col in self.thresholds:
                    result[f"{col}_biomarker"] = np.random.rand(len(df))
            return result
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def get_thresholds(self):
        """Return stored thresholds.

        Returns:
            Dictionary of thresholds.
        """
        return self.thresholds.copy()


@pytest.fixture
def sample_data():
    """Create sample training and test data.

    Returns:
        Tuple of (X_train, y_train, X_test, y_test).
    """
    np.random.seed(42)

    # Create training data (100 samples)
    n_train = 100
    X_train = pd.DataFrame(
        {
            "hemoglobin": np.random.normal(14, 2, n_train),
            "wbc": np.random.normal(7, 2, n_train),
            "platelets": np.random.normal(250, 50, n_train),
        }
    )

    # Generate labels with some correlation to features
    y_train = (
        (X_train["hemoglobin"] < 12)
        | (X_train["wbc"] > 10)
        | (X_train["platelets"] < 200)
    ).astype(int)

    # Create test data (50 samples)
    n_test = 50
    X_test = pd.DataFrame(
        {
            "hemoglobin": np.random.normal(14, 2, n_test),
            "wbc": np.random.normal(7, 2, n_test),
            "platelets": np.random.normal(250, 50, n_test),
        }
    )

    y_test = (
        (X_test["hemoglobin"] < 12)
        | (X_test["wbc"] > 10)
        | (X_test["platelets"] < 200)
    ).astype(int)

    return X_train, y_train, X_test, y_test


@pytest.fixture
def mock_generator():
    """Create a mock biomarker generator.

    Returns:
        MockBiomarkerGenerator instance.
    """
    feature_config = {
        "hemoglobin": {"unit": "g/dL"},
        "wbc": {"unit": "K/uL"},
        "platelets": {"unit": "K/uL"},
    }
    return MockBiomarkerGenerator(feature_config=feature_config)


@pytest.fixture
def evaluator():
    """Create a BiomarkerEvaluator instance.

    Returns:
        BiomarkerEvaluator instance.
    """
    # Use a temporary directory for MLflow tracking
    with tempfile.TemporaryDirectory() as tmpdir:
        mlflow_uri = f"file://{tmpdir}/mlruns"
        evaluator = BiomarkerEvaluator(
            mlflow_tracking_uri=mlflow_uri,
            experiment_name="test_experiment",
        )
        return evaluator


class TestBiomarkerEvaluatorInit:
    """Tests for BiomarkerEvaluator initialization."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        evaluator = BiomarkerEvaluator()

        assert evaluator.metrics is not None
        assert isinstance(evaluator.metrics_config, dict)
        assert evaluator.metrics_config["calculate_auc"] is True
        assert evaluator.metrics_config["create_plots"] is True

    def test_init_custom_config(self):
        """Test initialization with custom metrics config."""
        custom_config = {
            "calculate_auc": True,
            "calculate_precision_recall": False,
            "create_plots": False,
        }

        evaluator = BiomarkerEvaluator(metrics_config=custom_config)

        assert evaluator.metrics_config == custom_config
        assert evaluator.metrics_config["calculate_auc"] is True
        assert evaluator.metrics_config["calculate_precision_recall"] is False

    def test_init_with_mlflow_uri(self):
        """Test initialization with MLflow tracking URI."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mlflow_uri = f"file://{tmpdir}/mlruns"

            evaluator = BiomarkerEvaluator(
                mlflow_tracking_uri=mlflow_uri,
                experiment_name="test_exp",
            )

            assert evaluator is not None


class TestEvaluateGenerator:
    """Tests for evaluate_generator method."""

    def test_evaluate_generator_basic(self, sample_data, mock_generator):
        """Test basic evaluation of a generator."""
        X_train, y_train, X_test, y_test = sample_data

        with tempfile.TemporaryDirectory() as tmpdir:
            mlflow_uri = f"file://{tmpdir}/mlruns"
            evaluator = BiomarkerEvaluator(
                mlflow_tracking_uri=mlflow_uri,
                experiment_name="test_exp",
            )

            results = evaluator.evaluate_generator(
                generator=mock_generator,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                generator_name="MockGenerator",
                disease="test_disease",
            )

            # Check that results contain expected keys
            assert "generator_name" in results
            assert results["generator_name"] == "MockGenerator"
            assert "metrics" in results
            assert "thresholds" in results
            assert "predictions" in results
            assert "probabilities" in results

            # Check that metrics were calculated
            assert "auc" in results["metrics"]
            assert "precision" in results["metrics"]
            assert "recall" in results["metrics"]
            assert "f1" in results["metrics"]

            # Check that thresholds were captured
            assert len(results["thresholds"]) > 0

            # Check that predictions have correct shape
            assert len(results["predictions"]) == len(y_test)
            assert len(results["probabilities"]) == len(y_test)

    def test_evaluate_generator_with_numpy_arrays(self, sample_data, mock_generator):
        """Test evaluation with numpy arrays instead of pandas Series."""
        X_train, y_train, X_test, y_test = sample_data

        # Convert to numpy arrays
        y_train_np = y_train.values
        y_test_np = y_test.values

        with tempfile.TemporaryDirectory() as tmpdir:
            mlflow_uri = f"file://{tmpdir}/mlruns"
            evaluator = BiomarkerEvaluator(
                mlflow_tracking_uri=mlflow_uri,
                experiment_name="test_exp",
            )

            results = evaluator.evaluate_generator(
                generator=mock_generator,
                X_train=X_train,
                y_train=y_train_np,
                X_test=X_test,
                y_test=y_test_np,
                generator_name="MockGenerator",
            )

            # Should work fine with numpy arrays
            assert "metrics" in results
            assert "auc" in results["metrics"]

    def test_evaluate_generator_with_additional_params(
        self, sample_data, mock_generator
    ):
        """Test evaluation with additional parameters."""
        X_train, y_train, X_test, y_test = sample_data

        additional_params = {
            "model_version": "v1.0",
            "feature_selection": "manual",
            "preprocessing": "standard",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            mlflow_uri = f"file://{tmpdir}/mlruns"
            evaluator = BiomarkerEvaluator(
                mlflow_tracking_uri=mlflow_uri,
                experiment_name="test_exp",
            )

            results = evaluator.evaluate_generator(
                generator=mock_generator,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                generator_name="MockGenerator",
                additional_params=additional_params,
            )

            # Check that evaluation still works
            assert results is not None
            assert "metrics" in results

    def test_evaluate_generator_optimal_threshold(self, sample_data, mock_generator):
        """Test that optimal threshold is calculated."""
        X_train, y_train, X_test, y_test = sample_data

        with tempfile.TemporaryDirectory() as tmpdir:
            mlflow_uri = f"file://{tmpdir}/mlruns"
            evaluator = BiomarkerEvaluator(
                mlflow_tracking_uri=mlflow_uri,
                experiment_name="test_exp",
            )

            results = evaluator.evaluate_generator(
                generator=mock_generator,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                generator_name="MockGenerator",
            )

            # Check that optimal threshold was calculated
            assert "optimal_threshold" in results
            if results["optimal_threshold"] is not None:
                assert "threshold" in results["optimal_threshold"]
                assert "youden_index" in results["optimal_threshold"]

    def test_evaluate_generator_confusion_matrix(self, sample_data, mock_generator):
        """Test that confusion matrix is calculated."""
        X_train, y_train, X_test, y_test = sample_data

        with tempfile.TemporaryDirectory() as tmpdir:
            mlflow_uri = f"file://{tmpdir}/mlruns"
            evaluator = BiomarkerEvaluator(
                mlflow_tracking_uri=mlflow_uri,
                experiment_name="test_exp",
            )

            results = evaluator.evaluate_generator(
                generator=mock_generator,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                generator_name="MockGenerator",
            )

            # Check that confusion matrix metrics were calculated
            assert "sensitivity" in results["metrics"]
            assert "specificity" in results["metrics"]

    def test_evaluate_generator_with_plots_disabled(self, sample_data, mock_generator):
        """Test evaluation with plots disabled."""
        X_train, y_train, X_test, y_test = sample_data

        with tempfile.TemporaryDirectory() as tmpdir:
            mlflow_uri = f"file://{tmpdir}/mlruns"
            evaluator = BiomarkerEvaluator(
                mlflow_tracking_uri=mlflow_uri,
                experiment_name="test_exp",
                metrics_config={"create_plots": False},
            )

            results = evaluator.evaluate_generator(
                generator=mock_generator,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                generator_name="MockGenerator",
            )

            # Should still have results but no plots
            assert "metrics" in results
            assert len(results.get("plots", {})) == 0


class TestCompareGenerators:
    """Tests for compare_generators method."""

    def test_compare_generators_basic(self, sample_data):
        """Test basic comparison of multiple generators."""
        X_train, y_train, X_test, y_test = sample_data

        # Create two different generators
        gen1 = MockBiomarkerGenerator(method="simple")
        gen2 = MockBiomarkerGenerator(method="random")

        with tempfile.TemporaryDirectory() as tmpdir:
            mlflow_uri = f"file://{tmpdir}/mlruns"
            evaluator = BiomarkerEvaluator(
                mlflow_tracking_uri=mlflow_uri,
                experiment_name="test_exp",
                metrics_config={"create_plots": False},  # Disable plots for speed
            )

            # Evaluate both generators
            results1 = evaluator.evaluate_generator(
                generator=gen1,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                generator_name="SimpleGenerator",
            )

            results2 = evaluator.evaluate_generator(
                generator=gen2,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                generator_name="RandomGenerator",
            )

            # Compare generators
            comparison_df = evaluator.compare_generators([results1, results2])

            # Check comparison DataFrame
            assert isinstance(comparison_df, pd.DataFrame)
            assert len(comparison_df) == 2
            assert "Generator" in comparison_df.columns

            # Check that generator names are present
            assert "SimpleGenerator" in comparison_df["Generator"].values
            assert "RandomGenerator" in comparison_df["Generator"].values

            # Check that metrics are present
            assert "Auc" in comparison_df.columns or "auc" in comparison_df.columns

    def test_compare_generators_single_result(self, sample_data, mock_generator):
        """Test comparison with single generator result."""
        X_train, y_train, X_test, y_test = sample_data

        with tempfile.TemporaryDirectory() as tmpdir:
            mlflow_uri = f"file://{tmpdir}/mlruns"
            evaluator = BiomarkerEvaluator(
                mlflow_tracking_uri=mlflow_uri,
                experiment_name="test_exp",
                metrics_config={"create_plots": False},
            )

            results = evaluator.evaluate_generator(
                generator=mock_generator,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                generator_name="SingleGenerator",
            )

            # Compare with single result
            comparison_df = evaluator.compare_generators([results])

            assert isinstance(comparison_df, pd.DataFrame)
            assert len(comparison_df) == 1

    def test_compare_generators_empty_list(self):
        """Test that empty results list raises error."""
        evaluator = BiomarkerEvaluator()

        with pytest.raises(ValueError, match="results_list cannot be empty"):
            evaluator.compare_generators([])

    def test_compare_generators_with_output_path(self, sample_data, mock_generator):
        """Test comparison with output path for saving plots."""
        X_train, y_train, X_test, y_test = sample_data

        with tempfile.TemporaryDirectory() as tmpdir:
            mlflow_uri = f"file://{tmpdir}/mlruns"
            evaluator = BiomarkerEvaluator(
                mlflow_tracking_uri=mlflow_uri,
                experiment_name="test_exp",
                metrics_config={"create_plots": False},
            )

            results = evaluator.evaluate_generator(
                generator=mock_generator,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                generator_name="TestGenerator",
            )

            output_path = Path(tmpdir) / "comparison.png"

            # Use matplotlib non-interactive backend for testing
            plt.switch_backend("Agg")

            comparison_df = evaluator.compare_generators(
                [results],
                output_path=str(output_path),
            )

            # Check that plot was saved
            assert output_path.exists()

            # Clean up
            plt.close("all")

    def test_compare_generators_sorting(self, sample_data):
        """Test that generators are sorted by AUC."""
        X_train, y_train, X_test, y_test = sample_data

        # Manually create results with different AUC values
        results1 = {
            "generator_name": "Generator_A",
            "metrics": {"auc": 0.7, "precision": 0.6, "recall": 0.5},
        }

        results2 = {
            "generator_name": "Generator_B",
            "metrics": {"auc": 0.9, "precision": 0.8, "recall": 0.7},
        }

        results3 = {
            "generator_name": "Generator_C",
            "metrics": {"auc": 0.8, "precision": 0.7, "recall": 0.6},
        }

        plt.switch_backend("Agg")

        evaluator = BiomarkerEvaluator()
        comparison_df = evaluator.compare_generators([results1, results2, results3])

        # Check that results are sorted by AUC (descending)
        auc_col = "Auc" if "Auc" in comparison_df.columns else "auc"
        auc_values = comparison_df[auc_col].tolist()

        assert auc_values == sorted(auc_values, reverse=True)
        assert comparison_df.iloc[0]["Generator"] == "Generator_B"  # Highest AUC

        plt.close("all")


class TestIntegrationScenarios:
    """Integration tests for complete evaluation workflows."""

    def test_complete_evaluation_workflow(self, sample_data):
        """Test complete workflow: evaluate multiple generators and compare."""
        X_train, y_train, X_test, y_test = sample_data

        # Create different generators
        gen1 = MockBiomarkerGenerator(method="simple")
        gen2 = MockBiomarkerGenerator(method="random")

        with tempfile.TemporaryDirectory() as tmpdir:
            mlflow_uri = f"file://{tmpdir}/mlruns"
            evaluator = BiomarkerEvaluator(
                mlflow_tracking_uri=mlflow_uri,
                experiment_name="integration_test",
                metrics_config={"create_plots": False},  # Disable for speed
            )

            # Evaluate multiple generators
            results_list = []

            for idx, gen in enumerate([gen1, gen2]):
                results = evaluator.evaluate_generator(
                    generator=gen,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    generator_name=f"Generator_{idx+1}",
                    disease="TestDisease",
                )
                results_list.append(results)

            # Compare all generators
            comparison_df = evaluator.compare_generators(results_list)

            # Verify comparison
            assert len(comparison_df) == 2
            assert "Generator" in comparison_df.columns

            # Verify metrics are present
            metric_cols = [col for col in comparison_df.columns if col != "Generator"]
            assert len(metric_cols) > 0

    def test_edge_case_perfect_predictions(self):
        """Test evaluation with perfect predictions."""
        # Create data where biomarker perfectly predicts outcome
        np.random.seed(42)
        n = 100

        X_train = pd.DataFrame({"feature": np.random.rand(n)})
        y_train = pd.Series((X_train["feature"] > 0.5).astype(int))

        X_test = pd.DataFrame({"feature": np.random.rand(50)})
        y_test = pd.Series((X_test["feature"] > 0.5).astype(int))

        # Create a perfect generator
        class PerfectGenerator(BaseBiomarkerGenerator):
            def fit(self, df):
                self.thresholds = {"feature": {"threshold": 0.5}}
                return self

            def generate_labels(self, df):
                return pd.DataFrame({"biomarker": (df["feature"] > 0.5).astype(int)})

            def get_thresholds(self):
                return self.thresholds

        with tempfile.TemporaryDirectory() as tmpdir:
            mlflow_uri = f"file://{tmpdir}/mlruns"
            evaluator = BiomarkerEvaluator(
                mlflow_tracking_uri=mlflow_uri,
                experiment_name="perfect_test",
                metrics_config={"create_plots": False},
            )

            results = evaluator.evaluate_generator(
                generator=PerfectGenerator(),
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                generator_name="PerfectGenerator",
            )

            # Check that AUC is 1.0 (perfect)
            assert results["metrics"]["auc"] == 1.0
            assert results["metrics"]["precision"] == 1.0
            assert results["metrics"]["recall"] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
