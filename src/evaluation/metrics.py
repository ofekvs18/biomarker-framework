"""Evaluation metrics for biomarker and model assessment."""

from typing import Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_proba: Optional[pd.Series] = None,
) -> Dict[str, float]:
    """Compute classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities (optional, for AUC).

    Returns:
        Dictionary with metric names and values.
    """
    raise NotImplementedError


def compute_biomarker_metrics(
    df_biomarkers: pd.DataFrame,
    df_diagnoses: pd.DataFrame,
    disease_col: str,
) -> Dict[str, Dict[str, float]]:
    """Compute metrics for biomarker-disease associations.

    Args:
        df_biomarkers: DataFrame with binary biomarker columns.
        df_diagnoses: DataFrame with disease diagnosis labels.
        disease_col: Column name for target disease.

    Returns:
        Dictionary mapping biomarker names to their metrics.
    """
    raise NotImplementedError


def compute_feature_importance_correlation(
    importance_1: pd.Series,
    importance_2: pd.Series,
) -> Dict[str, float]:
    """Compute correlation between two feature importance rankings.

    Args:
        importance_1: First feature importance series.
        importance_2: Second feature importance series.

    Returns:
        Dictionary with correlation metrics (Spearman, Kendall, etc.).
    """
    raise NotImplementedError


class BiomarkerMetrics:
    """Comprehensive metrics for biomarker evaluation."""

    @staticmethod
    def calculate_auc(
        y_true: Union[np.ndarray, pd.Series],
        y_pred_proba: Union[np.ndarray, pd.Series],
    ) -> float:
        """Calculate Area Under the ROC Curve (AUC-ROC).

        Args:
            y_true: True binary labels (0 or 1).
            y_pred_proba: Predicted probabilities for the positive class.

        Returns:
            AUC-ROC score (0 to 1, higher is better).

        Raises:
            ValueError: If inputs have invalid shapes or values.
        """
        y_true = np.asarray(y_true)
        y_pred_proba = np.asarray(y_pred_proba)

        if len(y_true) != len(y_pred_proba):
            raise ValueError("y_true and y_pred_proba must have the same length")

        if len(np.unique(y_true)) < 2:
            raise ValueError("y_true must contain at least two classes")

        return roc_auc_score(y_true, y_pred_proba)

    @staticmethod
    def calculate_precision_recall(
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series],
        threshold: Optional[float] = None,
    ) -> Dict[str, float]:
        """Calculate precision and recall metrics.

        Args:
            y_true: True binary labels (0 or 1).
            y_pred: Predicted labels or probabilities.
            threshold: If provided and y_pred are probabilities, apply threshold.

        Returns:
            Dictionary with 'precision' and 'recall' keys.

        Raises:
            ValueError: If inputs have invalid shapes or values.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")

        # Apply threshold if provided and predictions are probabilities
        if threshold is not None:
            if not (0 <= threshold <= 1):
                raise ValueError("Threshold must be between 0 and 1")
            y_pred = (y_pred >= threshold).astype(int)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)

        return {"precision": float(precision), "recall": float(recall)}

    @staticmethod
    def calculate_confusion_matrix(
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """Calculate confusion matrix.

        Args:
            y_true: True binary labels (0 or 1).
            y_pred: Predicted binary labels (0 or 1).

        Returns:
            2x2 confusion matrix as numpy array.
            Format: [[TN, FP], [FN, TP]]

        Raises:
            ValueError: If inputs have invalid shapes or values.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")

        return confusion_matrix(y_true, y_pred)

    @staticmethod
    def find_optimal_threshold(
        y_true: Union[np.ndarray, pd.Series],
        y_pred_proba: Union[np.ndarray, pd.Series],
        metric: str = "youden",
    ) -> Dict[str, float]:
        """Find optimal classification threshold.

        Implements Youden's Index (J = Sensitivity + Specificity - 1).
        Finds the threshold that maximizes J.

        Args:
            y_true: True binary labels (0 or 1).
            y_pred_proba: Predicted probabilities for the positive class.
            metric: Optimization metric ('youden' is currently supported).

        Returns:
            Dictionary containing:
                - 'threshold': Optimal threshold value
                - 'youden_index': Maximum Youden's Index (J)
                - 'sensitivity': Sensitivity at optimal threshold
                - 'specificity': Specificity at optimal threshold

        Raises:
            ValueError: If inputs are invalid or metric is unsupported.
        """
        y_true = np.asarray(y_true)
        y_pred_proba = np.asarray(y_pred_proba)

        if len(y_true) != len(y_pred_proba):
            raise ValueError("y_true and y_pred_proba must have the same length")

        if len(np.unique(y_true)) < 2:
            raise ValueError("y_true must contain at least two classes")

        if metric != "youden":
            raise ValueError(f"Unsupported metric: {metric}. Only 'youden' is supported.")

        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)

        # Calculate Youden's Index: J = Sensitivity + Specificity - 1
        # Sensitivity = TPR (True Positive Rate)
        # Specificity = 1 - FPR (1 - False Positive Rate)
        youden_index = tpr - fpr

        # Find threshold that maximizes Youden's Index
        optimal_idx = np.argmax(youden_index)
        optimal_threshold = thresholds[optimal_idx]
        max_youden = youden_index[optimal_idx]

        sensitivity = tpr[optimal_idx]
        specificity = 1 - fpr[optimal_idx]

        return {
            "threshold": float(optimal_threshold),
            "youden_index": float(max_youden),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
        }

    @staticmethod
    def plot_roc_curve(
        y_true: Union[np.ndarray, pd.Series],
        y_pred_proba: Union[np.ndarray, pd.Series],
        title: str = "ROC Curve",
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """Plot ROC curve with AUC score.

        Args:
            y_true: True binary labels (0 or 1).
            y_pred_proba: Predicted probabilities for the positive class.
            title: Plot title.
            ax: Matplotlib axes object. If None, creates new figure.

        Returns:
            Matplotlib axes object with the plot.

        Raises:
            ValueError: If inputs have invalid shapes or values.
        """
        y_true = np.asarray(y_true)
        y_pred_proba = np.asarray(y_pred_proba)

        if len(y_true) != len(y_pred_proba):
            raise ValueError("y_true and y_pred_proba must have the same length")

        if len(np.unique(y_true)) < 2:
            raise ValueError("y_true must contain at least two classes")

        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        # Create plot
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label=f"ROC curve (AUC = {roc_auc:.3f})",
        )
        ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Chance")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)

        return ax

    @staticmethod
    def plot_precision_recall_curve(
        y_true: Union[np.ndarray, pd.Series],
        y_pred_proba: Union[np.ndarray, pd.Series],
        title: str = "Precision-Recall Curve",
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """Plot precision-recall curve.

        Args:
            y_true: True binary labels (0 or 1).
            y_pred_proba: Predicted probabilities for the positive class.
            title: Plot title.
            ax: Matplotlib axes object. If None, creates new figure.

        Returns:
            Matplotlib axes object with the plot.

        Raises:
            ValueError: If inputs have invalid shapes or values.
        """
        y_true = np.asarray(y_true)
        y_pred_proba = np.asarray(y_pred_proba)

        if len(y_true) != len(y_pred_proba):
            raise ValueError("y_true and y_pred_proba must have the same length")

        if len(np.unique(y_true)) < 2:
            raise ValueError("y_true must contain at least two classes")

        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)

        # Create plot
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(
            recall,
            precision,
            color="darkorange",
            lw=2,
            label=f"PR curve (AUC = {pr_auc:.3f})",
        )

        # Baseline (proportion of positive class)
        baseline = np.sum(y_true) / len(y_true)
        ax.axhline(
            y=baseline,
            color="navy",
            lw=2,
            linestyle="--",
            label=f"Baseline ({baseline:.3f})",
        )

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)

        return ax

    @staticmethod
    def plot_confusion_matrix_heatmap(
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series],
        title: str = "Confusion Matrix",
        ax: Optional[plt.Axes] = None,
        labels: Optional[list] = None,
    ) -> plt.Axes:
        """Plot confusion matrix as a heatmap.

        Args:
            y_true: True binary labels (0 or 1).
            y_pred: Predicted binary labels (0 or 1).
            title: Plot title.
            ax: Matplotlib axes object. If None, creates new figure.
            labels: Class labels for display. Defaults to ['Negative', 'Positive'].

        Returns:
            Matplotlib axes object with the plot.

        Raises:
            ValueError: If inputs have invalid shapes or values.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")

        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Create plot
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        # Default labels
        if labels is None:
            labels = ["Negative", "Positive"]

        # Plot heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={"label": "Count"},
        )

        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")

        # Add TN, FP, FN, TP labels
        if cm.shape == (2, 2):
            annotations = [["TN", "FP"], ["FN", "TP"]]
            for i in range(2):
                for j in range(2):
                    text = ax.text(
                        j + 0.5,
                        i + 0.7,
                        annotations[i][j],
                        ha="center",
                        va="center",
                        color="gray",
                        fontsize=9,
                        fontstyle="italic",
                    )

        return ax
