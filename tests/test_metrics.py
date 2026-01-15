"""Unit tests for BiomarkerMetrics class."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.evaluation.metrics import BiomarkerMetrics

# Use non-interactive backend for testing
matplotlib.use("Agg")


class TestBiomarkerMetrics:
    """Test suite for BiomarkerMetrics class."""

    @pytest.fixture
    def perfect_predictions(self):
        """Perfect classifier predictions."""
        np.random.seed(42)
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred_proba = np.array([0.1, 0.2, 0.15, 0.85, 0.9, 0.95])
        y_pred = np.array([0, 0, 0, 1, 1, 1])
        return y_true, y_pred_proba, y_pred

    @pytest.fixture
    def balanced_predictions(self):
        """Balanced classifier predictions with some errors."""
        np.random.seed(42)
        n_samples = 100
        y_true = np.random.randint(0, 2, n_samples)
        # Add some noise to create realistic probabilities
        y_pred_proba = y_true + np.random.normal(0, 0.3, n_samples)
        y_pred_proba = np.clip(y_pred_proba, 0, 1)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        return y_true, y_pred_proba, y_pred

    @pytest.fixture
    def imbalanced_predictions(self):
        """Imbalanced dataset predictions."""
        np.random.seed(42)
        n_samples = 100
        # 90% negative, 10% positive
        y_true = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
        y_pred_proba = y_true + np.random.normal(0, 0.3, n_samples)
        y_pred_proba = np.clip(y_pred_proba, 0, 1)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        return y_true, y_pred_proba, y_pred

    # Test calculate_auc
    def test_calculate_auc_perfect_classifier(self, perfect_predictions):
        """Test AUC calculation with perfect classifier."""
        y_true, y_pred_proba, _ = perfect_predictions
        auc = BiomarkerMetrics.calculate_auc(y_true, y_pred_proba)
        assert auc == 1.0, "Perfect classifier should have AUC of 1.0"

    def test_calculate_auc_random_classifier(self):
        """Test AUC calculation with random classifier."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 1000)
        y_pred_proba = np.random.rand(1000)
        auc = BiomarkerMetrics.calculate_auc(y_true, y_pred_proba)
        # Random classifier should have AUC around 0.5
        assert 0.4 <= auc <= 0.6, f"Random classifier AUC should be ~0.5, got {auc}"

    def test_calculate_auc_realistic_classifier(self, balanced_predictions):
        """Test AUC calculation with realistic classifier."""
        y_true, y_pred_proba, _ = balanced_predictions
        auc = BiomarkerMetrics.calculate_auc(y_true, y_pred_proba)
        assert 0.5 <= auc <= 1.0, f"AUC should be between 0.5 and 1.0, got {auc}"

    def test_calculate_auc_length_mismatch(self):
        """Test AUC with mismatched input lengths."""
        y_true = np.array([0, 1, 0])
        y_pred_proba = np.array([0.3, 0.7])
        with pytest.raises(ValueError, match="must have the same length"):
            BiomarkerMetrics.calculate_auc(y_true, y_pred_proba)

    def test_calculate_auc_single_class(self):
        """Test AUC with single class."""
        y_true = np.array([1, 1, 1, 1])
        y_pred_proba = np.array([0.8, 0.9, 0.7, 0.85])
        with pytest.raises(ValueError, match="at least two classes"):
            BiomarkerMetrics.calculate_auc(y_true, y_pred_proba)

    # Test calculate_precision_recall
    def test_calculate_precision_recall_perfect(self, perfect_predictions):
        """Test precision and recall with perfect predictions."""
        y_true, _, y_pred = perfect_predictions
        metrics = BiomarkerMetrics.calculate_precision_recall(y_true, y_pred)
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0

    def test_calculate_precision_recall_with_threshold(self, balanced_predictions):
        """Test precision and recall with threshold application."""
        y_true, y_pred_proba, _ = balanced_predictions
        metrics = BiomarkerMetrics.calculate_precision_recall(
            y_true, y_pred_proba, threshold=0.5
        )
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0

    def test_calculate_precision_recall_invalid_threshold(self, balanced_predictions):
        """Test precision and recall with invalid threshold."""
        y_true, y_pred_proba, _ = balanced_predictions
        with pytest.raises(ValueError, match="Threshold must be between 0 and 1"):
            BiomarkerMetrics.calculate_precision_recall(y_true, y_pred_proba, threshold=1.5)

    def test_calculate_precision_recall_zero_positives(self):
        """Test precision and recall with no positive predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0])
        metrics = BiomarkerMetrics.calculate_precision_recall(y_true, y_pred)
        # With zero_division=0, precision should be 0 when no positive predictions
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0

    def test_calculate_precision_recall_length_mismatch(self):
        """Test precision and recall with mismatched input lengths."""
        y_true = np.array([0, 1, 0])
        y_pred = np.array([0, 1])
        with pytest.raises(ValueError, match="must have the same length"):
            BiomarkerMetrics.calculate_precision_recall(y_true, y_pred)

    # Test calculate_confusion_matrix
    def test_calculate_confusion_matrix_perfect(self, perfect_predictions):
        """Test confusion matrix with perfect predictions."""
        y_true, _, y_pred = perfect_predictions
        cm = BiomarkerMetrics.calculate_confusion_matrix(y_true, y_pred)
        # Perfect predictions: TN=3, FP=0, FN=0, TP=3
        assert cm[0, 0] == 3  # TN
        assert cm[0, 1] == 0  # FP
        assert cm[1, 0] == 0  # FN
        assert cm[1, 1] == 3  # TP

    def test_calculate_confusion_matrix_balanced(self, balanced_predictions):
        """Test confusion matrix with balanced predictions."""
        y_true, _, y_pred = balanced_predictions
        cm = BiomarkerMetrics.calculate_confusion_matrix(y_true, y_pred)
        # Check shape
        assert cm.shape == (2, 2)
        # Check that total equals sample size
        assert cm.sum() == len(y_true)

    def test_calculate_confusion_matrix_all_negative(self):
        """Test confusion matrix with all negative predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0])
        cm = BiomarkerMetrics.calculate_confusion_matrix(y_true, y_pred)
        assert cm[0, 0] == 2  # TN
        assert cm[0, 1] == 0  # FP
        assert cm[1, 0] == 2  # FN
        assert cm[1, 1] == 0  # TP

    def test_calculate_confusion_matrix_length_mismatch(self):
        """Test confusion matrix with mismatched input lengths."""
        y_true = np.array([0, 1, 0])
        y_pred = np.array([0, 1])
        with pytest.raises(ValueError, match="must have the same length"):
            BiomarkerMetrics.calculate_confusion_matrix(y_true, y_pred)

    # Test find_optimal_threshold
    def test_find_optimal_threshold_perfect(self, perfect_predictions):
        """Test optimal threshold with perfect predictions."""
        y_true, y_pred_proba, _ = perfect_predictions
        result = BiomarkerMetrics.find_optimal_threshold(y_true, y_pred_proba)

        assert "threshold" in result
        assert "youden_index" in result
        assert "sensitivity" in result
        assert "specificity" in result

        # Perfect classifier should have Youden's Index of 1.0
        assert result["youden_index"] == 1.0
        assert result["sensitivity"] == 1.0
        assert result["specificity"] == 1.0

    def test_find_optimal_threshold_balanced(self, balanced_predictions):
        """Test optimal threshold with balanced predictions."""
        y_true, y_pred_proba, _ = balanced_predictions
        result = BiomarkerMetrics.find_optimal_threshold(y_true, y_pred_proba)

        # Check that threshold is reasonable
        assert 0.0 <= result["threshold"] <= 1.0
        # Check that Youden's Index is between -1 and 1
        assert -1.0 <= result["youden_index"] <= 1.0
        # Check that sensitivity and specificity are valid
        assert 0.0 <= result["sensitivity"] <= 1.0
        assert 0.0 <= result["specificity"] <= 1.0

    def test_find_optimal_threshold_youden_formula(self):
        """Test that Youden's Index formula is correct."""
        # Create a known scenario
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

        result = BiomarkerMetrics.find_optimal_threshold(y_true, y_pred_proba)

        # Youden's Index = Sensitivity + Specificity - 1
        calculated_youden = result["sensitivity"] + result["specificity"] - 1
        assert abs(result["youden_index"] - calculated_youden) < 1e-6

    def test_find_optimal_threshold_unsupported_metric(self, balanced_predictions):
        """Test optimal threshold with unsupported metric."""
        y_true, y_pred_proba, _ = balanced_predictions
        with pytest.raises(ValueError, match="Unsupported metric"):
            BiomarkerMetrics.find_optimal_threshold(y_true, y_pred_proba, metric="f1")

    def test_find_optimal_threshold_single_class(self):
        """Test optimal threshold with single class."""
        y_true = np.array([1, 1, 1, 1])
        y_pred_proba = np.array([0.8, 0.9, 0.7, 0.85])
        with pytest.raises(ValueError, match="at least two classes"):
            BiomarkerMetrics.find_optimal_threshold(y_true, y_pred_proba)

    # Test plot_roc_curve
    def test_plot_roc_curve_creates_plot(self, balanced_predictions):
        """Test that ROC curve plot is created."""
        y_true, y_pred_proba, _ = balanced_predictions
        ax = BiomarkerMetrics.plot_roc_curve(y_true, y_pred_proba)

        assert ax is not None
        assert ax.get_xlabel() == "False Positive Rate"
        assert ax.get_ylabel() == "True Positive Rate"
        assert "ROC Curve" in ax.get_title()

        # Close the figure to free memory
        plt.close(ax.figure)

    def test_plot_roc_curve_with_custom_title(self, balanced_predictions):
        """Test ROC curve with custom title."""
        y_true, y_pred_proba, _ = balanced_predictions
        custom_title = "Custom ROC Curve Title"
        ax = BiomarkerMetrics.plot_roc_curve(y_true, y_pred_proba, title=custom_title)

        assert custom_title in ax.get_title()
        plt.close(ax.figure)

    def test_plot_roc_curve_with_existing_axes(self, balanced_predictions):
        """Test ROC curve with existing axes."""
        y_true, y_pred_proba, _ = balanced_predictions
        fig, ax = plt.subplots()
        result_ax = BiomarkerMetrics.plot_roc_curve(y_true, y_pred_proba, ax=ax)

        assert result_ax is ax
        plt.close(fig)

    def test_plot_roc_curve_invalid_input(self):
        """Test ROC curve with invalid input."""
        y_true = np.array([0, 1, 0])
        y_pred_proba = np.array([0.3, 0.7])

        with pytest.raises(ValueError, match="must have the same length"):
            BiomarkerMetrics.plot_roc_curve(y_true, y_pred_proba)

    # Test plot_precision_recall_curve
    def test_plot_precision_recall_curve_creates_plot(self, balanced_predictions):
        """Test that precision-recall curve plot is created."""
        y_true, y_pred_proba, _ = balanced_predictions
        ax = BiomarkerMetrics.plot_precision_recall_curve(y_true, y_pred_proba)

        assert ax is not None
        assert ax.get_xlabel() == "Recall"
        assert ax.get_ylabel() == "Precision"
        assert "Precision-Recall Curve" in ax.get_title()

        plt.close(ax.figure)

    def test_plot_precision_recall_curve_with_custom_title(self, balanced_predictions):
        """Test precision-recall curve with custom title."""
        y_true, y_pred_proba, _ = balanced_predictions
        custom_title = "Custom PR Curve"
        ax = BiomarkerMetrics.plot_precision_recall_curve(
            y_true, y_pred_proba, title=custom_title
        )

        assert custom_title in ax.get_title()
        plt.close(ax.figure)

    def test_plot_precision_recall_curve_baseline(self, imbalanced_predictions):
        """Test that baseline is shown correctly."""
        y_true, y_pred_proba, _ = imbalanced_predictions
        ax = BiomarkerMetrics.plot_precision_recall_curve(y_true, y_pred_proba)

        # Check that legend contains baseline
        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert any("Baseline" in text for text in legend_texts)

        plt.close(ax.figure)

    def test_plot_precision_recall_curve_invalid_input(self):
        """Test precision-recall curve with invalid input."""
        y_true = np.array([0, 1, 0])
        y_pred_proba = np.array([0.3, 0.7])

        with pytest.raises(ValueError, match="must have the same length"):
            BiomarkerMetrics.plot_precision_recall_curve(y_true, y_pred_proba)

    # Test plot_confusion_matrix_heatmap
    def test_plot_confusion_matrix_heatmap_creates_plot(self, balanced_predictions):
        """Test that confusion matrix heatmap is created."""
        y_true, _, y_pred = balanced_predictions
        ax = BiomarkerMetrics.plot_confusion_matrix_heatmap(y_true, y_pred)

        assert ax is not None
        assert ax.get_xlabel() == "Predicted Label"
        assert ax.get_ylabel() == "True Label"
        assert "Confusion Matrix" in ax.get_title()

        plt.close(ax.figure)

    def test_plot_confusion_matrix_heatmap_custom_labels(self, balanced_predictions):
        """Test confusion matrix with custom labels."""
        y_true, _, y_pred = balanced_predictions
        custom_labels = ["Healthy", "Disease"]
        ax = BiomarkerMetrics.plot_confusion_matrix_heatmap(
            y_true, y_pred, labels=custom_labels
        )

        # Check that custom labels are used
        x_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert "Healthy" in x_labels or "Disease" in x_labels

        plt.close(ax.figure)

    def test_plot_confusion_matrix_heatmap_custom_title(self, balanced_predictions):
        """Test confusion matrix with custom title."""
        y_true, _, y_pred = balanced_predictions
        custom_title = "Custom Confusion Matrix"
        ax = BiomarkerMetrics.plot_confusion_matrix_heatmap(
            y_true, y_pred, title=custom_title
        )

        assert custom_title in ax.get_title()
        plt.close(ax.figure)

    def test_plot_confusion_matrix_heatmap_invalid_input(self):
        """Test confusion matrix heatmap with invalid input."""
        y_true = np.array([0, 1, 0])
        y_pred = np.array([0, 1])

        with pytest.raises(ValueError, match="must have the same length"):
            BiomarkerMetrics.plot_confusion_matrix_heatmap(y_true, y_pred)

    # Integration tests
    def test_full_workflow_perfect_classifier(self, perfect_predictions):
        """Test full workflow with perfect classifier."""
        y_true, y_pred_proba, y_pred = perfect_predictions

        # Calculate metrics
        auc = BiomarkerMetrics.calculate_auc(y_true, y_pred_proba)
        pr_metrics = BiomarkerMetrics.calculate_precision_recall(y_true, y_pred)
        cm = BiomarkerMetrics.calculate_confusion_matrix(y_true, y_pred)
        optimal = BiomarkerMetrics.find_optimal_threshold(y_true, y_pred_proba)

        # All metrics should be perfect
        assert auc == 1.0
        assert pr_metrics["precision"] == 1.0
        assert pr_metrics["recall"] == 1.0
        assert optimal["youden_index"] == 1.0

    def test_full_workflow_realistic_classifier(self, balanced_predictions):
        """Test full workflow with realistic classifier."""
        y_true, y_pred_proba, y_pred = balanced_predictions

        # Calculate all metrics
        auc = BiomarkerMetrics.calculate_auc(y_true, y_pred_proba)
        pr_metrics = BiomarkerMetrics.calculate_precision_recall(y_true, y_pred)
        cm = BiomarkerMetrics.calculate_confusion_matrix(y_true, y_pred)
        optimal = BiomarkerMetrics.find_optimal_threshold(y_true, y_pred_proba)

        # Create all plots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        BiomarkerMetrics.plot_roc_curve(y_true, y_pred_proba, ax=axes[0])
        BiomarkerMetrics.plot_precision_recall_curve(y_true, y_pred_proba, ax=axes[1])
        BiomarkerMetrics.plot_confusion_matrix_heatmap(y_true, y_pred, ax=axes[2])

        # Verify all metrics are reasonable
        assert 0.5 <= auc <= 1.0
        assert 0.0 <= pr_metrics["precision"] <= 1.0
        assert 0.0 <= pr_metrics["recall"] <= 1.0
        assert cm.shape == (2, 2)
        assert 0.0 <= optimal["threshold"] <= 1.0

        plt.close(fig)

    def test_pandas_series_input(self):
        """Test that methods accept pandas Series input."""
        import pandas as pd

        y_true = pd.Series([0, 0, 1, 1])
        y_pred_proba = pd.Series([0.2, 0.3, 0.7, 0.8])
        y_pred = pd.Series([0, 0, 1, 1])

        # All methods should work with pandas Series
        auc = BiomarkerMetrics.calculate_auc(y_true, y_pred_proba)
        pr_metrics = BiomarkerMetrics.calculate_precision_recall(y_true, y_pred)
        cm = BiomarkerMetrics.calculate_confusion_matrix(y_true, y_pred)
        optimal = BiomarkerMetrics.find_optimal_threshold(y_true, y_pred_proba)

        assert isinstance(auc, float)
        assert isinstance(pr_metrics, dict)
        assert isinstance(cm, np.ndarray)
        assert isinstance(optimal, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
