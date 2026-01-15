"""SHAP Baseline Experiment Script.

This script runs SHAP-based biomarker generation methods and compares them
with coefficient-based methods from Phase 1. It evaluates all methods across
multiple target diseases and generates comprehensive comparison reports.

Usage:
    python scripts/run_shap_baseline.py --data_dir data/processed --output_dir experiments/shap_baseline_results

Example:
    # Run on specific disease
    python scripts/run_shap_baseline.py --disease rheumatoid_arthritis

    # Run on all diseases with custom model
    python scripts/run_shap_baseline.py --model_type XGBoost --n_estimators 200
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.splitter import DataSplitter
from src.evaluation.evaluator import BiomarkerEvaluator
from src.generators.datadriven_threshold import YoudensIndexGenerator
from src.generators.literature_threshold import SingleFeatureLiteratureGenerator
from src.generators.shap_datadriven_threshold import SHAPDataDrivenThresholdGenerator
from src.generators.shap_literature_threshold import SHAPLiteratureThresholdGenerator


def load_config(config_path: Path) -> Dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def extract_literature_thresholds(config: Dict, disease: str) -> Dict[str, float]:
    """Extract literature thresholds for a specific disease.

    Args:
        config: Full configuration dictionary
        disease: Disease name

    Returns:
        Dictionary mapping feature names to threshold values
    """
    disease_thresholds = config["literature_thresholds"].get(disease, {})

    # Flatten to simple feature: threshold mapping
    thresholds = {}
    for feature, threshold_info in disease_thresholds.items():
        if isinstance(threshold_info, dict):
            # Use 'low' threshold by default
            thresholds[feature] = threshold_info.get("low", threshold_info.get("high"))
        else:
            thresholds[feature] = threshold_info

    return thresholds


def run_experiment_for_disease(
    disease: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    config: Dict,
    output_dir: Path,
    model_type: str = "RandomForest",
    n_estimators: int = 100,
) -> Dict[str, any]:
    """Run all generators for a single disease and compare results.

    Args:
        disease: Disease name
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        feature_names: List of feature names
        config: Configuration dictionary
        output_dir: Output directory for results
        model_type: Model type for SHAP generators
        n_estimators: Number of estimators for tree models

    Returns:
        Dictionary with results from all generators
    """
    print(f"\n{'=' * 80}")
    print(f"Running experiments for: {disease.upper()}")
    print(f"{'=' * 80}\n")

    # Create disease-specific output directory
    disease_dir = output_dir / disease
    disease_dir.mkdir(parents=True, exist_ok=True)

    # Initialize evaluator
    evaluator = BiomarkerEvaluator(
        experiment_name=f"shap_baseline_{disease}",
        mlflow_tracking_uri=f"sqlite:///{output_dir}/mlflow.db",
    )

    # Extract literature thresholds for this disease
    literature_thresholds = extract_literature_thresholds(config, disease)

    if not literature_thresholds:
        print(f"Warning: No literature thresholds found for {disease}")
        print("Using default thresholds from CBC features config")
        literature_thresholds = {
            feature: config["cbc_features"][feature]["reference_range"]["low"]
            for feature in feature_names
            if feature in config["cbc_features"]
        }

    # Common generator config
    base_config = {
        "random_state": 42,
        "model_type": model_type,
        "n_estimators": n_estimators,
    }

    all_results = {}

    # ====================
    # Phase 1: Coefficient-Based Methods (Baseline)
    # ====================
    print("\n--- Phase 1: Coefficient-Based Methods ---\n")

    # Method 1A: Single Feature + Literature Threshold (Coefficient-based)
    print("1A. Single Feature + Literature Threshold (Coefficient-based)")
    try:
        gen_1a = SingleFeatureLiteratureGenerator(
            config={"random_state": 42},
            literature_thresholds=literature_thresholds,
        )
        result_1a = evaluator.evaluate_ml_generator(
            generator=gen_1a,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_names,
            generator_name="Coef_LitThreshold",
            disease=disease,
            save_shap_plots=False,
        )
        all_results["coef_literature"] = result_1a
    except Exception as e:
        print(f"  Error: {e}")
        all_results["coef_literature"] = None

    # Method 1B: Single Feature + Youden's Index (Coefficient-based)
    print("\n1B. Single Feature + Youden's Index (Coefficient-based)")
    try:
        gen_1b = YoudensIndexGenerator(config={"random_state": 42, "use_cv": True})
        result_1b = evaluator.evaluate_ml_generator(
            generator=gen_1b,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_names,
            generator_name="Coef_Youdens",
            disease=disease,
            save_shap_plots=False,
        )
        all_results["coef_youdens"] = result_1b
    except Exception as e:
        print(f"  Error: {e}")
        all_results["coef_youdens"] = None

    # ====================
    # Phase 2: SHAP-Based Methods
    # ====================
    print("\n--- Phase 2: SHAP-Based Methods ---\n")

    # Method 2A: SHAP + Literature Threshold
    print("2A. SHAP + Literature Threshold")
    try:
        gen_2a = SHAPLiteratureThresholdGenerator(
            config=base_config,
            literature_thresholds=literature_thresholds,
        )
        result_2a = evaluator.evaluate_ml_generator(
            generator=gen_2a,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_names,
            generator_name="SHAP_LitThreshold",
            disease=disease,
            additional_params={"model_type": model_type},
            save_shap_plots=True,
        )
        all_results["shap_literature"] = result_2a
    except Exception as e:
        print(f"  Error: {e}")
        all_results["shap_literature"] = None

    # Method 2B: SHAP + Youden's Index
    print("\n2B. SHAP + Youden's Index")
    try:
        gen_2b = SHAPDataDrivenThresholdGenerator(
            config={**base_config, "use_cv": True, "cv_folds": 5}
        )
        result_2b = evaluator.evaluate_ml_generator(
            generator=gen_2b,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_names,
            generator_name="SHAP_Youdens",
            disease=disease,
            additional_params={"model_type": model_type},
            save_shap_plots=True,
        )
        all_results["shap_youdens"] = result_2b
    except Exception as e:
        print(f"  Error: {e}")
        all_results["shap_youdens"] = None

    # ====================
    # Comparison and Analysis
    # ====================
    print("\n--- Generating Comparison Reports ---\n")

    # Create comparison table
    comparison_data = []
    for method_key, result in all_results.items():
        if result is None:
            continue

        row = {
            "Method": result["generator_name"],
            "Selected Feature": result["biomarker"]["features_used"][0],
            "Threshold": result["biomarker"]["threshold"],
            "Formula": result["biomarker"]["formula"],
        }

        # Add metrics
        metrics = result.get("metrics", {})
        row["Precision"] = metrics.get("precision", np.nan)
        row["Recall"] = metrics.get("recall", np.nan)
        row["F1"] = metrics.get("f1", np.nan)
        row["Specificity"] = metrics.get("specificity", np.nan)
        row["Sensitivity"] = metrics.get("sensitivity", np.nan)
        row["Accuracy"] = metrics.get("accuracy", np.nan)

        # Add method-specific info
        metadata = result["biomarker"].get("metadata", {})
        if "shap_importance" in metadata:
            row["Feature Importance"] = f"{metadata['shap_importance']:.4f} (SHAP)"
        elif "coefficient" in metadata:
            row["Feature Importance"] = f"{metadata['coefficient']:.4f} (Coef)"

        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)

    # Save comparison table
    comparison_path = disease_dir / "comparison_table.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Comparison table saved to: {comparison_path}")

    # Print comparison
    print("\n" + "=" * 100)
    print(f"RESULTS SUMMARY: {disease.upper()}")
    print("=" * 100)
    print(comparison_df.to_string(index=False))
    print("=" * 100 + "\n")

    # Create feature selection comparison visualization
    create_feature_comparison_plot(comparison_df, disease, disease_dir)

    # Create performance comparison visualization
    create_performance_comparison_plot(comparison_df, disease, disease_dir)

    return all_results


def create_feature_comparison_plot(
    comparison_df: pd.DataFrame, disease: str, output_dir: Path
):
    """Create visualization comparing feature selection across methods.

    Args:
        comparison_df: Comparison DataFrame
        disease: Disease name
        output_dir: Output directory
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    methods = comparison_df["Method"].tolist()
    features = comparison_df["Selected Feature"].tolist()

    # Create bar plot
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
    bars = ax.barh(methods, range(len(methods)), color=colors[: len(methods)])

    # Add feature names as text
    for i, (method, feature) in enumerate(zip(methods, features)):
        ax.text(
            i,
            i,
            f"  {feature}",
            va="center",
            ha="left",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=11)
    ax.set_xlim(-0.5, len(methods) - 0.5)
    ax.set_xticks([])
    ax.set_title(
        f"Feature Selection Comparison - {disease.replace('_', ' ').title()}",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("Selected Feature", fontsize=12)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "feature_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Feature comparison plot saved to: {plot_path}")


def create_performance_comparison_plot(
    comparison_df: pd.DataFrame, disease: str, output_dir: Path
):
    """Create visualization comparing performance metrics across methods.

    Args:
        comparison_df: Comparison DataFrame
        disease: Disease name
        output_dir: Output directory
    """
    metric_cols = ["Precision", "Recall", "F1", "Specificity", "Sensitivity"]
    available_metrics = [col for col in metric_cols if col in comparison_df.columns]

    if not available_metrics:
        print("No metrics available for plotting")
        return

    fig, axes = plt.subplots(1, len(available_metrics), figsize=(4 * len(available_metrics), 6))

    if len(available_metrics) == 1:
        axes = [axes]

    methods = comparison_df["Method"].tolist()
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]

    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        values = comparison_df[metric].tolist()

        bars = ax.bar(methods, values, color=colors[: len(methods)])

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if not pd.isna(height):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

        ax.set_ylabel(metric, fontsize=12, fontweight="bold")
        ax.set_ylim([0, 1.1])
        ax.set_title(f"{metric}", fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=45)

    plt.suptitle(
        f"Performance Comparison - {disease.replace('_', ' ').title()}",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    plot_path = output_dir / "performance_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Performance comparison plot saved to: {plot_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Run SHAP baseline experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="Directory containing processed data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/shap_baseline_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cbc_features.yaml",
        help="Path to CBC features config file",
    )
    parser.add_argument(
        "--disease",
        type=str,
        default=None,
        help="Specific disease to run (if None, runs all available)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="RandomForest",
        choices=["RandomForest", "XGBoost", "LogisticRegression"],
        help="Model type for SHAP generators",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100,
        help="Number of estimators for tree-based models",
    )

    args = parser.parse_args()

    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)

    # Determine which diseases to run
    if args.disease:
        diseases = [args.disease]
    else:
        # Get diseases from config or data directory
        diseases = list(config.get("literature_thresholds", {}).keys())

    if not diseases:
        print("Error: No diseases specified or found in config")
        sys.exit(1)

    print(f"Running experiments for diseases: {diseases}")
    print(f"Model type: {args.model_type}")
    print(f"Output directory: {output_dir}\n")

    # Run experiments for each disease
    all_disease_results = {}

    for disease in diseases:
        # For this template, we'll assume data is loaded from splits
        # In practice, you'd need to implement data loading logic
        print(f"\nNote: This is a template script.")
        print(f"To run experiments, you need to:")
        print(f"  1. Load data for {disease} from {data_dir}")
        print(f"  2. Split into train/test sets using DataSplitter")
        print(f"  3. Extract features and labels")
        print(f"\nExample:")
        print(f"  train_df, val_df, test_df = DataSplitter.load_splits('{disease}', '{data_dir}')")
        print(f"  X_train = train_df[feature_names].values")
        print(f"  y_train = train_df['label'].values")
        print()

        # TEMPLATE: Replace this with actual data loading
        # try:
        #     train_df, val_df, test_df = DataSplitter.load_splits(disease, data_dir)
        #
        #     # Extract features (assuming CBC features are in the dataframe)
        #     feature_names = list(config['feature_groups']['core_cbc'])
        #     X_train = train_df[feature_names].values
        #     y_train = train_df['has_disease'].values
        #     X_test = test_df[feature_names].values
        #     y_test = test_df['has_disease'].values
        #
        #     results = run_experiment_for_disease(
        #         disease=disease,
        #         X_train=X_train,
        #         y_train=y_train,
        #         X_test=X_test,
        #         y_test=y_test,
        #         feature_names=feature_names,
        #         config=config,
        #         output_dir=output_dir,
        #         model_type=args.model_type,
        #         n_estimators=args.n_estimators,
        #     )
        #
        #     all_disease_results[disease] = results
        # except Exception as e:
        #     print(f"Error processing {disease}: {e}")
        #     continue

    print("\n" + "=" * 80)
    print("EXPERIMENT TEMPLATE COMPLETE")
    print("=" * 80)
    print(f"\nTo use this script with real data:")
    print(f"1. Implement data loading logic in main()")
    print(f"2. Ensure disease-specific data is available in {data_dir}")
    print(f"3. Update feature selection to match your data schema")
    print(f"\nResults will be saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
