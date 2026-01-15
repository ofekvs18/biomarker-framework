"""Model and biomarker evaluation orchestration."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns

from .metrics import BiomarkerMetrics

try:
    from generators.base import BaseBiomarkerGenerator, BiomarkerGenerator
except ImportError:
    from src.generators.base import BaseBiomarkerGenerator, BiomarkerGenerator


class Evaluator:
    """Orchestrate model and biomarker evaluation."""

    def __init__(self, metrics: Optional[List[str]] = None):
        """Initialize Evaluator.

        Args:
            metrics: List of metrics to compute.
        """
        self.metrics = metrics or [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "auc_roc",
        ]
        self.results: Dict[str, Any] = {}

    def evaluate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, float]:
        """Evaluate a trained model.

        Args:
            model: Trained model with predict/predict_proba methods.
            X_test: Test features.
            y_test: Test labels.

        Returns:
            Dictionary of metric values.
        """
        raise NotImplementedError

    def evaluate_biomarkers(
        self,
        df_biomarkers: pd.DataFrame,
        df_labels: pd.DataFrame,
        target_col: str,
    ) -> pd.DataFrame:
        """Evaluate biomarker predictive power.

        Args:
            df_biomarkers: DataFrame with binary biomarker columns.
            df_labels: DataFrame with target labels.
            target_col: Column name for target variable.

        Returns:
            DataFrame with metrics per biomarker.
        """
        raise NotImplementedError

    def cross_validate(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
    ) -> Dict[str, List[float]]:
        """Perform cross-validation.

        Args:
            model: Model to evaluate.
            X: Feature matrix.
            y: Target labels.
            cv: Number of folds.

        Returns:
            Dictionary mapping metrics to list of fold scores.
        """
        raise NotImplementedError

    def compare_generators(
        self,
        results_by_generator: Dict[str, Dict],
    ) -> pd.DataFrame:
        """Compare results across different biomarker generators.

        Args:
            results_by_generator: Dictionary mapping generator names to results.

        Returns:
            Comparison DataFrame.
        """
        raise NotImplementedError


class BiomarkerEvaluator:
    """Comprehensive evaluation framework for biomarker generators with MLflow integration."""

    def __init__(
        self,
        metrics_config: Optional[Dict] = None,
        mlflow_tracking_uri: Optional[str] = None,
        experiment_name: str = "biomarker_evaluation",
    ):
        """Initialize BiomarkerEvaluator.

        Args:
            metrics_config: Configuration for which metrics to compute.
            mlflow_tracking_uri: URI for MLflow tracking server. If None, uses SQLite (mlflow.db).
            experiment_name: Name for the MLflow experiment.
        """
        self.metrics = BiomarkerMetrics()
        self.metrics_config = metrics_config or {
            "calculate_auc": True,
            "calculate_precision_recall": True,
            "calculate_confusion_matrix": True,
            "find_optimal_threshold": True,
            "create_plots": True,
        }

        # Initialize MLflow with SQLite by default
        if mlflow_tracking_uri is None:
            mlflow_tracking_uri = "sqlite:///mlflow.db"
        mlflow.set_tracking_uri(mlflow_tracking_uri)

        # Set or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            print(f"Warning: Could not set up MLflow experiment: {e}")

    def evaluate_generator(
        self,
        generator: BaseBiomarkerGenerator,
        X_train: pd.DataFrame,
        y_train: Union[pd.Series, np.ndarray],
        X_test: pd.DataFrame,
        y_test: Union[pd.Series, np.ndarray],
        generator_name: str,
        disease: Optional[str] = None,
        additional_params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Evaluate a biomarker generator on test data with MLflow logging.

        This method:
        1. Fits the generator on training data
        2. Generates biomarker labels on test data
        3. Calculates all configured metrics
        4. Creates visualizations
        5. Logs everything to MLflow
        6. Returns comprehensive results dictionary

        Args:
            generator: BiomarkerGenerator instance to evaluate.
            X_train: Training features (CBC measurements).
            y_train: Training labels (disease diagnosis).
            X_test: Test features (CBC measurements).
            y_test: Test labels (disease diagnosis).
            generator_name: Name of the generator method for logging.
            disease: Disease being predicted (for logging).
            additional_params: Additional parameters to log to MLflow.

        Returns:
            Dictionary containing:
                - 'metrics': Dict of calculated metrics (AUC, precision, recall, etc.)
                - 'optimal_threshold': Optimal classification threshold info
                - 'predictions': Test set predictions
                - 'probabilities': Test set prediction probabilities
                - 'plots': Dictionary of matplotlib figure objects
                - 'thresholds': Biomarker thresholds used by generator
                - 'generator_name': Name of the generator
        """
        # Convert to pandas Series if needed
        if isinstance(y_train, np.ndarray):
            y_train = pd.Series(y_train)
        if isinstance(y_test, np.ndarray):
            y_test = pd.Series(y_test)

        results = {
            "generator_name": generator_name,
            "metrics": {},
            "plots": {},
            "thresholds": {},
        }

        # Start MLflow run
        with mlflow.start_run(run_name=f"{generator_name}_{disease or 'unknown'}"):
            try:
                # Log parameters
                mlflow.log_param("generator_method", generator_name)
                mlflow.log_param("n_train_samples", len(X_train))
                mlflow.log_param("n_test_samples", len(X_test))
                mlflow.log_param("n_features", X_train.shape[1])
                mlflow.log_param("feature_names", list(X_train.columns))

                if disease:
                    mlflow.log_param("disease", disease)

                # Log class distribution
                mlflow.log_param("train_pos_rate", float(y_train.mean()))
                mlflow.log_param("test_pos_rate", float(y_test.mean()))

                # Log additional parameters if provided
                if additional_params:
                    for key, value in additional_params.items():
                        mlflow.log_param(key, value)

                # Step 1: Fit generator on training data
                print(f"Fitting {generator_name} on training data...")
                generator.fit(X_train)

                # Get and log thresholds
                thresholds = generator.get_thresholds()
                results["thresholds"] = thresholds
                for feature, threshold_info in thresholds.items():
                    if isinstance(threshold_info, dict):
                        for key, value in threshold_info.items():
                            mlflow.log_param(f"threshold_{feature}_{key}", value)
                    else:
                        mlflow.log_param(f"threshold_{feature}", threshold_info)

                # Step 2: Generate biomarker labels on test data
                print(f"Generating biomarker labels on test data...")
                biomarker_labels = generator.generate_labels(X_test)

                # For evaluation, we need a single combined biomarker score
                # We'll use the mean of all biomarker columns as a probability score
                if isinstance(biomarker_labels, pd.DataFrame):
                    y_pred_proba = biomarker_labels.mean(axis=1).values
                    y_pred = (y_pred_proba >= 0.5).astype(int)
                else:
                    y_pred_proba = biomarker_labels.values
                    y_pred = (y_pred_proba >= 0.5).astype(int)

                results["predictions"] = y_pred
                results["probabilities"] = y_pred_proba

                # Step 3: Calculate all metrics
                print("Calculating metrics...")

                # AUC
                if self.metrics_config.get("calculate_auc", True):
                    try:
                        auc = self.metrics.calculate_auc(y_test, y_pred_proba)
                        results["metrics"]["auc"] = float(auc)
                        mlflow.log_metric("auc", auc)
                        print(f"  AUC: {auc:.4f}")
                    except Exception as e:
                        print(f"  Warning: Could not calculate AUC: {e}")
                        results["metrics"]["auc"] = None

                # Precision and Recall
                if self.metrics_config.get("calculate_precision_recall", True):
                    try:
                        pr_metrics = self.metrics.calculate_precision_recall(
                            y_test, y_pred
                        )
                        results["metrics"]["precision"] = pr_metrics["precision"]
                        results["metrics"]["recall"] = pr_metrics["recall"]

                        # Calculate F1 score
                        if pr_metrics["precision"] + pr_metrics["recall"] > 0:
                            f1 = (
                                2
                                * pr_metrics["precision"]
                                * pr_metrics["recall"]
                                / (pr_metrics["precision"] + pr_metrics["recall"])
                            )
                        else:
                            f1 = 0.0
                        results["metrics"]["f1"] = f1

                        mlflow.log_metric("precision", pr_metrics["precision"])
                        mlflow.log_metric("recall", pr_metrics["recall"])
                        mlflow.log_metric("f1", f1)

                        print(f"  Precision: {pr_metrics['precision']:.4f}")
                        print(f"  Recall: {pr_metrics['recall']:.4f}")
                        print(f"  F1: {f1:.4f}")
                    except Exception as e:
                        print(f"  Warning: Could not calculate precision/recall: {e}")

                # Confusion Matrix
                if self.metrics_config.get("calculate_confusion_matrix", True):
                    try:
                        cm = self.metrics.calculate_confusion_matrix(y_test, y_pred)
                        results["metrics"]["confusion_matrix"] = cm

                        # Log individual confusion matrix values
                        if cm.shape == (2, 2):
                            mlflow.log_metric("tn", int(cm[0, 0]))
                            mlflow.log_metric("fp", int(cm[0, 1]))
                            mlflow.log_metric("fn", int(cm[1, 0]))
                            mlflow.log_metric("tp", int(cm[1, 1]))

                            # Calculate specificity and sensitivity
                            specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
                            sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0

                            results["metrics"]["specificity"] = specificity
                            results["metrics"]["sensitivity"] = sensitivity

                            mlflow.log_metric("specificity", specificity)
                            mlflow.log_metric("sensitivity", sensitivity)

                            print(f"  Specificity: {specificity:.4f}")
                            print(f"  Sensitivity: {sensitivity:.4f}")
                    except Exception as e:
                        print(f"  Warning: Could not calculate confusion matrix: {e}")

                # Optimal Threshold
                if self.metrics_config.get("find_optimal_threshold", True):
                    try:
                        threshold_info = self.metrics.find_optimal_threshold(
                            y_test, y_pred_proba
                        )
                        results["optimal_threshold"] = threshold_info

                        mlflow.log_metric("optimal_threshold", threshold_info["threshold"])
                        mlflow.log_metric("youden_index", threshold_info["youden_index"])

                        print(f"  Optimal Threshold: {threshold_info['threshold']:.4f}")
                        print(f"  Youden Index: {threshold_info['youden_index']:.4f}")
                    except Exception as e:
                        print(f"  Warning: Could not find optimal threshold: {e}")
                        results["optimal_threshold"] = None

                # Step 4: Create visualizations
                if self.metrics_config.get("create_plots", True):
                    print("Creating visualizations...")

                    # ROC Curve
                    try:
                        fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
                        self.metrics.plot_roc_curve(
                            y_test,
                            y_pred_proba,
                            title=f"ROC Curve - {generator_name}",
                            ax=ax_roc,
                        )
                        results["plots"]["roc_curve"] = fig_roc

                        # Save and log to MLflow
                        roc_path = f"roc_curve_{generator_name}.png"
                        fig_roc.savefig(roc_path, dpi=150, bbox_inches="tight")
                        mlflow.log_artifact(roc_path)
                        Path(roc_path).unlink()  # Clean up
                        plt.close(fig_roc)
                    except Exception as e:
                        print(f"  Warning: Could not create ROC curve: {e}")

                    # Precision-Recall Curve
                    try:
                        fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
                        self.metrics.plot_precision_recall_curve(
                            y_test,
                            y_pred_proba,
                            title=f"Precision-Recall Curve - {generator_name}",
                            ax=ax_pr,
                        )
                        results["plots"]["pr_curve"] = fig_pr

                        # Save and log to MLflow
                        pr_path = f"pr_curve_{generator_name}.png"
                        fig_pr.savefig(pr_path, dpi=150, bbox_inches="tight")
                        mlflow.log_artifact(pr_path)
                        Path(pr_path).unlink()  # Clean up
                        plt.close(fig_pr)
                    except Exception as e:
                        print(f"  Warning: Could not create PR curve: {e}")

                    # Confusion Matrix
                    try:
                        fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
                        self.metrics.plot_confusion_matrix_heatmap(
                            y_test,
                            y_pred,
                            title=f"Confusion Matrix - {generator_name}",
                            ax=ax_cm,
                        )
                        results["plots"]["confusion_matrix"] = fig_cm

                        # Save and log to MLflow
                        cm_path = f"confusion_matrix_{generator_name}.png"
                        fig_cm.savefig(cm_path, dpi=150, bbox_inches="tight")
                        mlflow.log_artifact(cm_path)
                        Path(cm_path).unlink()  # Clean up
                        plt.close(fig_cm)
                    except Exception as e:
                        print(f"  Warning: Could not create confusion matrix: {e}")

                # Log biomarker formula/thresholds as text artifact
                threshold_text = f"Biomarker Thresholds for {generator_name}\n"
                threshold_text += "=" * 50 + "\n\n"
                for feature, threshold_info in thresholds.items():
                    threshold_text += f"{feature}:\n"
                    if isinstance(threshold_info, dict):
                        for key, value in threshold_info.items():
                            threshold_text += f"  {key}: {value}\n"
                    else:
                        threshold_text += f"  {threshold_info}\n"
                    threshold_text += "\n"

                threshold_file = f"thresholds_{generator_name}.txt"
                with open(threshold_file, "w") as f:
                    f.write(threshold_text)
                mlflow.log_artifact(threshold_file)
                Path(threshold_file).unlink()  # Clean up

                print(f"\nEvaluation complete for {generator_name}!")
                print(f"Results logged to MLflow experiment: {mlflow.get_experiment(mlflow.active_run().info.experiment_id).name}")

            except Exception as e:
                print(f"Error during evaluation: {e}")
                mlflow.log_param("error", str(e))
                raise

        return results

    def compare_generators(
        self,
        results_list: List[Dict[str, Any]],
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """Compare results across multiple biomarker generators.

        Creates a comprehensive comparison table and visualization showing
        performance metrics for different generator methods.

        Args:
            results_list: List of result dictionaries from evaluate_generator.
            output_path: Optional path to save comparison plot.

        Returns:
            DataFrame with comparison metrics across all generators.
        """
        if not results_list:
            raise ValueError("results_list cannot be empty")

        # Extract metrics from each result
        comparison_data = []
        for result in results_list:
            row = {"Generator": result["generator_name"]}

            # Add all metrics
            metrics = result.get("metrics", {})
            for metric_name, metric_value in metrics.items():
                if metric_name != "confusion_matrix":  # Skip confusion matrix
                    row[metric_name.replace("_", " ").title()] = metric_value

            # Add optimal threshold info if available
            if "optimal_threshold" in result and result["optimal_threshold"]:
                row["Optimal Threshold"] = result["optimal_threshold"]["threshold"]

            comparison_data.append(row)

        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data)

        # Sort by AUC (descending) if available
        if "Auc" in comparison_df.columns:
            comparison_df = comparison_df.sort_values("Auc", ascending=False)

        print("\n" + "=" * 80)
        print("GENERATOR COMPARISON SUMMARY")
        print("=" * 80)
        print(comparison_df.to_string(index=False))
        print("=" * 80)

        # Create comparison visualization
        self._create_comparison_plots(comparison_df, output_path)

        return comparison_df

    def _create_comparison_plots(
        self,
        comparison_df: pd.DataFrame,
        output_path: Optional[str] = None,
    ) -> None:
        """Create visualization comparing generator performance.

        Args:
            comparison_df: DataFrame with comparison metrics.
            output_path: Optional path to save plot.
        """
        # Identify numeric metric columns (exclude Generator name)
        metric_cols = [
            col
            for col in comparison_df.columns
            if col != "Generator" and pd.api.types.is_numeric_dtype(comparison_df[col])
        ]

        if not metric_cols:
            print("No numeric metrics to plot")
            return

        # Create subplots for key metrics
        key_metrics = ["Auc", "Precision", "Recall", "F1"]
        available_metrics = [m for m in key_metrics if m in metric_cols]

        if not available_metrics:
            # Use all available metrics
            available_metrics = metric_cols[:4]  # Limit to 4 for readability

        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))

        if n_metrics == 1:
            axes = [axes]

        for idx, metric in enumerate(available_metrics):
            ax = axes[idx]

            # Create bar plot
            bars = ax.bar(
                comparison_df["Generator"],
                comparison_df[metric],
                color=sns.color_palette("husl", len(comparison_df)),
            )

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                if not pd.isna(height):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{height:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )

            ax.set_ylabel(metric, fontsize=12, fontweight="bold")
            ax.set_xlabel("Generator Method", fontsize=11)
            ax.set_title(f"{metric} Comparison", fontsize=13, fontweight="bold")
            ax.set_ylim([0, 1.1])
            ax.grid(axis="y", alpha=0.3)
            ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()

        # Save plot
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"\nComparison plot saved to: {output_path}")

            # Log to MLflow if within a run
            try:
                if mlflow.active_run():
                    mlflow.log_artifact(output_path)
            except Exception:
                pass

        plt.show()

        # Create a heatmap for all metrics
        if len(metric_cols) > 4:
            fig_heatmap, ax_heatmap = plt.subplots(figsize=(12, max(6, len(comparison_df) * 0.8)))

            # Prepare data for heatmap
            heatmap_data = comparison_df.set_index("Generator")[metric_cols]

            # Create heatmap
            sns.heatmap(
                heatmap_data.T,  # Transpose so metrics are rows
                annot=True,
                fmt=".3f",
                cmap="YlGnBu",
                ax=ax_heatmap,
                cbar_kws={"label": "Score"},
                linewidths=0.5,
            )

            ax_heatmap.set_title(
                "Comprehensive Metrics Heatmap",
                fontsize=14,
                fontweight="bold",
                pad=20,
            )
            ax_heatmap.set_xlabel("Generator Method", fontsize=12)
            ax_heatmap.set_ylabel("Metric", fontsize=12)

            plt.tight_layout()

            if output_path:
                heatmap_path = output_path.replace(".png", "_heatmap.png")
                plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
                print(f"Heatmap saved to: {heatmap_path}")

                # Log to MLflow if within a run
                try:
                    if mlflow.active_run():
                        mlflow.log_artifact(heatmap_path)
                except Exception:
                    pass

            plt.show()

    def evaluate_ml_generator(
        self,
        generator: BiomarkerGenerator,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
        feature_names: List[str],
        generator_name: str,
        disease: Optional[str] = None,
        additional_params: Optional[Dict] = None,
        save_shap_plots: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate a ML-based biomarker generator (with generate/apply interface).

        This method is designed for the modern BiomarkerGenerator interface used by
        SHAP-based and other ML-driven generators. It:
        1. Generates biomarker from training data
        2. Applies biomarker to test data
        3. Calculates all configured metrics
        4. Creates visualizations (including SHAP plots if available)
        5. Logs everything to MLflow
        6. Returns comprehensive results dictionary

        Args:
            generator: BiomarkerGenerator instance to evaluate.
            X_train: Training features (n_samples, n_features).
            y_train: Training labels.
            X_test: Test features (n_samples, n_features).
            y_test: Test labels.
            feature_names: List of feature names.
            generator_name: Name of the generator method for logging.
            disease: Disease being predicted (for logging).
            additional_params: Additional parameters to log to MLflow.
            save_shap_plots: Whether to save SHAP plots (if generator supports it).

        Returns:
            Dictionary containing:
                - 'generator_name': str
                - 'metrics': Dict of calculated metrics
                - 'biomarker': Dict with biomarker formula and metadata
                - 'predictions': Test set predictions
                - 'plots': Dictionary of matplotlib figure objects
        """
        # Convert to numpy arrays if needed
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        if isinstance(y_test, pd.Series):
            y_test = y_test.values

        results = {
            "generator_name": generator_name,
            "metrics": {},
            "plots": {},
            "biomarker": {},
        }

        # Start MLflow run
        with mlflow.start_run(run_name=f"{generator_name}_{disease or 'unknown'}"):
            try:
                # Log parameters
                mlflow.log_param("generator_method", generator_name)
                mlflow.log_param("n_train_samples", len(X_train))
                mlflow.log_param("n_test_samples", len(X_test))
                mlflow.log_param("n_features", X_train.shape[1])
                mlflow.log_param("feature_names", feature_names)

                if disease:
                    mlflow.log_param("disease", disease)

                # Log class distribution
                mlflow.log_param("train_pos_rate", float(np.mean(y_train)))
                mlflow.log_param("test_pos_rate", float(np.mean(y_test)))

                # Log additional parameters if provided
                if additional_params:
                    for key, value in additional_params.items():
                        mlflow.log_param(key, value)

                # Step 1: Generate biomarker from training data
                print(f"Generating biomarker using {generator_name}...")
                biomarker_info = generator.generate(X_train, y_train, feature_names)
                results["biomarker"] = biomarker_info

                # Log biomarker information
                mlflow.log_param("biomarker_formula", biomarker_info["formula"])
                mlflow.log_param("biomarker_threshold", biomarker_info["threshold"])
                mlflow.log_param("features_used", biomarker_info["features_used"])

                # Log metadata
                if "metadata" in biomarker_info:
                    for key, value in biomarker_info["metadata"].items():
                        if isinstance(value, (int, float, str, bool)):
                            mlflow.log_param(f"metadata_{key}", value)
                        elif isinstance(value, dict) and all(
                            isinstance(v, (int, float)) for v in value.values()
                        ):
                            # Log dict of numeric values as individual params
                            for sub_key, sub_value in value.items():
                                mlflow.log_param(f"metadata_{key}_{sub_key}", sub_value)

                print(f"  Formula: {biomarker_info['formula']}")
                print(f"  Features used: {biomarker_info['features_used']}")

                # Step 2: Apply biomarker to test data
                print("Applying biomarker to test data...")
                y_pred = generator.apply(X_test)
                results["predictions"] = y_pred

                # For ROC/AUC, we need probabilities. Use predictions as binary scores.
                # If this is a binary threshold-based method, we can't compute a proper AUC,
                # but we can still compute other metrics.
                y_pred_proba = y_pred.astype(float)

                # Step 3: Calculate all metrics
                print("Calculating metrics...")

                # Precision and Recall
                try:
                    pr_metrics = self.metrics.calculate_precision_recall(y_test, y_pred)
                    results["metrics"]["precision"] = pr_metrics["precision"]
                    results["metrics"]["recall"] = pr_metrics["recall"]

                    # Calculate F1 score
                    if pr_metrics["precision"] + pr_metrics["recall"] > 0:
                        f1 = (
                            2
                            * pr_metrics["precision"]
                            * pr_metrics["recall"]
                            / (pr_metrics["precision"] + pr_metrics["recall"])
                        )
                    else:
                        f1 = 0.0
                    results["metrics"]["f1"] = f1

                    mlflow.log_metric("precision", pr_metrics["precision"])
                    mlflow.log_metric("recall", pr_metrics["recall"])
                    mlflow.log_metric("f1", f1)

                    print(f"  Precision: {pr_metrics['precision']:.4f}")
                    print(f"  Recall: {pr_metrics['recall']:.4f}")
                    print(f"  F1: {f1:.4f}")
                except Exception as e:
                    print(f"  Warning: Could not calculate precision/recall: {e}")

                # Confusion Matrix
                try:
                    cm = self.metrics.calculate_confusion_matrix(y_test, y_pred)
                    results["metrics"]["confusion_matrix"] = cm

                    # Log individual confusion matrix values
                    if cm.shape == (2, 2):
                        mlflow.log_metric("tn", int(cm[0, 0]))
                        mlflow.log_metric("fp", int(cm[0, 1]))
                        mlflow.log_metric("fn", int(cm[1, 0]))
                        mlflow.log_metric("tp", int(cm[1, 1]))

                        # Calculate specificity, sensitivity, and accuracy
                        specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
                        sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
                        accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum() if cm.sum() > 0 else 0

                        results["metrics"]["specificity"] = specificity
                        results["metrics"]["sensitivity"] = sensitivity
                        results["metrics"]["accuracy"] = accuracy

                        mlflow.log_metric("specificity", specificity)
                        mlflow.log_metric("sensitivity", sensitivity)
                        mlflow.log_metric("accuracy", accuracy)

                        print(f"  Specificity: {specificity:.4f}")
                        print(f"  Sensitivity: {sensitivity:.4f}")
                        print(f"  Accuracy: {accuracy:.4f}")
                except Exception as e:
                    print(f"  Warning: Could not calculate confusion matrix: {e}")

                # Step 4: Create visualizations
                if self.metrics_config.get("create_plots", True):
                    print("Creating visualizations...")

                    # Confusion Matrix
                    try:
                        fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
                        self.metrics.plot_confusion_matrix_heatmap(
                            y_test,
                            y_pred,
                            title=f"Confusion Matrix - {generator_name}",
                            ax=ax_cm,
                        )
                        results["plots"]["confusion_matrix"] = fig_cm

                        # Save and log to MLflow
                        cm_path = f"confusion_matrix_{generator_name}.png"
                        fig_cm.savefig(cm_path, dpi=150, bbox_inches="tight")
                        mlflow.log_artifact(cm_path)
                        Path(cm_path).unlink()  # Clean up
                        plt.close(fig_cm)
                    except Exception as e:
                        print(f"  Warning: Could not create confusion matrix: {e}")

                    # SHAP plots if available
                    if save_shap_plots and hasattr(generator, "get_shap_plots"):
                        try:
                            print("Creating SHAP visualizations...")
                            shap_plots = generator.get_shap_plots(max_display=15)

                            # Save summary plot
                            if "summary" in shap_plots:
                                summary_path = f"shap_summary_{generator_name}.png"
                                shap_plots["summary"].savefig(
                                    summary_path, dpi=150, bbox_inches="tight"
                                )
                                mlflow.log_artifact(summary_path)
                                Path(summary_path).unlink()
                                plt.close(shap_plots["summary"])

                            # Save bar plot
                            if "bar" in shap_plots:
                                bar_path = f"shap_bar_{generator_name}.png"
                                shap_plots["bar"].savefig(
                                    bar_path, dpi=150, bbox_inches="tight"
                                )
                                mlflow.log_artifact(bar_path)
                                Path(bar_path).unlink()
                                plt.close(shap_plots["bar"])

                            print("  SHAP plots saved and logged to MLflow")
                        except Exception as e:
                            print(f"  Warning: Could not create SHAP plots: {e}")

                # Log biomarker description as text artifact
                biomarker_text = f"Biomarker Description for {generator_name}\n"
                biomarker_text += "=" * 50 + "\n\n"
                biomarker_text += generator.get_description() + "\n"

                biomarker_file = f"biomarker_{generator_name}.txt"
                with open(biomarker_file, "w") as f:
                    f.write(biomarker_text)
                mlflow.log_artifact(biomarker_file)
                Path(biomarker_file).unlink()  # Clean up

                print(f"\nEvaluation complete for {generator_name}!")
                exp_id = mlflow.active_run().info.experiment_id
                exp_name = mlflow.get_experiment(exp_id).name
                print(f"Results logged to MLflow experiment: {exp_name}")

            except Exception as e:
                print(f"Error during evaluation: {e}")
                mlflow.log_param("error", str(e))
                raise

        return results
