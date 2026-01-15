"""End-to-end pipeline for Phase 1 biomarker discovery experiments.

This script runs complete experiments for Phase 1 biomarker generation methods:
- Method 1A: Single feature with literature-defined thresholds
- Method 1B: Single feature with data-driven thresholds (Youden's Index)

For each disease:
1. Load and preprocess CBC data
2. Split into train/test sets
3. Generate biomarkers using both methods
4. Evaluate performance (AUC-ROC, sensitivity, specificity, etc.)
5. Compare methods and save results
6. Generate visualizations (ROC curves, confusion matrices)

Usage:
    python scripts/run_phase1_experiments.py
    python scripts/run_phase1_experiments.py --disease rheumatoid_arthritis
    python scripts/run_phase1_experiments.py --all-diseases --output experiments/phase1_results/
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.loader import MIMICLoader
from data.preprocessor import CBCPreprocessor
from evaluation.metrics import BiomarkerMetrics
from generators.datadriven_threshold import YoudensIndexGenerator
from generators.literature_threshold import SingleFeatureLiteratureGenerator


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("phase1_experiments.log"),
    ],
)
logger = logging.getLogger(__name__)


def load_config() -> Dict:
    """Load configuration from YAML files.

    Returns:
        Dictionary containing merged disease and CBC feature configs.
    """
    config_dir = Path(__file__).parent.parent / "configs"

    with open(config_dir / "diseases.yaml") as f:
        diseases_config = yaml.safe_load(f)

    with open(config_dir / "cbc_features.yaml") as f:
        cbc_config = yaml.safe_load(f)

    # Merge configs
    config = {**diseases_config, **cbc_config}
    return config


def prepare_dataset(
    disease_key: str, config: Dict, data_dir: Path, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], Dict]:
    """Load and preprocess data for a specific disease.

    Args:
        disease_key: Disease identifier (e.g., 'rheumatoid_arthritis')
        config: Configuration dictionary
        data_dir: Path to MIMIC-IV data directory
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names, dataset_info)
    """
    logger.info(f"Loading data for {disease_key}...")

    # Initialize loader and preprocessor
    loader = MIMICLoader(data_dir=str(data_dir), config=config)
    preprocessor = CBCPreprocessor(config=config)

    # Create preprocessed dataset
    try:
        dataset = preprocessor.create_dataset(
            loader=loader,
            disease_key=disease_key,
            lookback_days=30,
            missing_strategy="median",
            normalize_method="standard",
        )
    except Exception as e:
        logger.error(f"Failed to create dataset for {disease_key}: {e}")
        raise

    logger.info(f"Dataset shape: {dataset.shape}")

    # Extract features and labels
    id_cols = ["subject_id", "label"]
    feature_cols = [col for col in dataset.columns if col not in id_cols]

    X = dataset.select(feature_cols).to_numpy()
    y = dataset.select("label").to_numpy().flatten()

    logger.info(f"Features: {len(feature_cols)} columns")
    logger.info(f"Samples: {len(y)} total")
    logger.info(f"Class distribution: {np.bincount(y)} (0=negative, 1=positive)")
    logger.info(f"Positive rate: {np.mean(y):.2%}")

    # Check for class imbalance
    positive_rate = np.mean(y)
    if positive_rate < 0.01:
        logger.warning(
            f"Very low positive rate ({positive_rate:.2%}). "
            "Results may be unreliable."
        )

    # Split into train/test (70/30)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y
    )

    logger.info(f"Train set: {len(y_train)} samples ({np.mean(y_train):.2%} positive)")
    logger.info(f"Test set: {len(y_test)} samples ({np.mean(y_test):.2%} positive)")

    # Prepare dataset info
    dataset_info = {
        "disease_key": disease_key,
        "disease_name": config["diseases"][disease_key]["name"],
        "n_samples": len(y),
        "n_features": len(feature_cols),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "positive_rate": float(positive_rate),
        "train_positive_rate": float(np.mean(y_train)),
        "test_positive_rate": float(np.mean(y_test)),
    }

    return X_train, X_test, y_train, y_test, feature_cols, dataset_info


def evaluate_biomarker(
    generator,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    features: List[str],
    method_name: str,
) -> Dict:
    """Generate and evaluate a biomarker using the given generator.

    Args:
        generator: BiomarkerGenerator instance (SingleFeatureLiteratureGenerator or YoudensIndexGenerator)
        X_train: Training feature matrix
        X_test: Test feature matrix
        y_train: Training labels
        y_test: Test labels
        features: List of feature names
        method_name: Name of the method for logging

    Returns:
        Dictionary containing biomarker info and evaluation metrics
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Running {method_name}")
    logger.info(f"{'='*60}")

    # Generate biomarker
    try:
        biomarker = generator.generate(X_train, y_train, features)
        logger.info(f"Generated biomarker: {biomarker['formula']}")
        logger.info(f"Description:\n{generator.get_description()}")
    except Exception as e:
        logger.error(f"Failed to generate biomarker: {e}")
        raise

    # Apply to test set
    y_pred = generator.apply(X_test)
    logger.info(f"Test predictions: {np.bincount(y_pred)} (0=negative, 1=positive)")

    # Calculate metrics
    try:
        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # ROC-AUC (using predictions as probabilities for simple threshold-based methods)
        if len(np.unique(y_pred)) > 1:
            auc = roc_auc_score(y_test, y_pred)
        else:
            logger.warning("Only one class predicted. AUC set to 0.5")
            auc = 0.5

        # Precision and recall
        pr_metrics = BiomarkerMetrics.calculate_precision_recall(y_test, y_pred)
        precision = pr_metrics["precision"]
        recall = pr_metrics["recall"]

        # Confusion matrix
        cm = BiomarkerMetrics.calculate_confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Calculate sensitivity and specificity
        sensitivity = recall  # Same as recall
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # Log metrics
        logger.info(f"\nTest Set Performance:")
        logger.info(f"  Accuracy:    {accuracy:.4f}")
        logger.info(f"  AUC-ROC:     {auc:.4f}")
        logger.info(f"  Precision:   {precision:.4f}")
        logger.info(f"  Recall:      {recall:.4f}")
        logger.info(f"  F1 Score:    {f1:.4f}")
        logger.info(f"  Sensitivity: {sensitivity:.4f}")
        logger.info(f"  Specificity: {specificity:.4f}")
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"  TN={tn}, FP={fp}")
        logger.info(f"  FN={fn}, TP={tp}")

    except Exception as e:
        logger.error(f"Failed to calculate metrics: {e}")
        raise

    # Prepare results
    results = {
        "method": method_name,
        "biomarker": biomarker,
        "predictions": {
            "y_pred": y_pred.tolist(),
            "y_test": y_test.tolist(),
        },
        "metrics": {
            "accuracy": float(accuracy),
            "auc_roc": float(auc),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
        },
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }

    return results


def generate_comparison_plots(
    results_1a: Dict,
    results_1b: Dict,
    disease_name: str,
    output_dir: Path,
) -> None:
    """Generate comparison plots for both methods.

    Args:
        results_1a: Results from Method 1A (Literature-based)
        results_1b: Results from Method 1B (Data-driven)
        disease_name: Name of the disease for plot titles
        output_dir: Directory to save plots
    """
    logger.info("\nGenerating comparison plots...")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"Phase 1 Biomarker Comparison: {disease_name}", fontsize=16, fontweight="bold")

    # Extract data
    y_test_1a = np.array(results_1a["predictions"]["y_test"])
    y_pred_1a = np.array(results_1a["predictions"]["y_pred"])
    y_test_1b = np.array(results_1b["predictions"]["y_test"])
    y_pred_1b = np.array(results_1b["predictions"]["y_pred"])

    # Plot 1: ROC curves for both methods
    ax1 = axes[0, 0]
    if len(np.unique(y_pred_1a)) > 1:
        BiomarkerMetrics.plot_roc_curve(
            y_test_1a,
            y_pred_1a,
            title=f"ROC Curve - {results_1a['method']}",
            ax=ax1,
        )
    else:
        ax1.text(0.5, 0.5, "Insufficient predictions for ROC", ha="center", va="center")
        ax1.set_title(f"ROC Curve - {results_1a['method']}")

    # Plot 2: ROC curve for Method 1B
    ax2 = axes[0, 1]
    if len(np.unique(y_pred_1b)) > 1:
        BiomarkerMetrics.plot_roc_curve(
            y_test_1b,
            y_pred_1b,
            title=f"ROC Curve - {results_1b['method']}",
            ax=ax2,
        )
    else:
        ax2.text(0.5, 0.5, "Insufficient predictions for ROC", ha="center", va="center")
        ax2.set_title(f"ROC Curve - {results_1b['method']}")

    # Plot 3: Confusion matrix for Method 1A
    ax3 = axes[1, 0]
    BiomarkerMetrics.plot_confusion_matrix_heatmap(
        y_test_1a,
        y_pred_1a,
        title=f"Confusion Matrix - {results_1a['method']}",
        ax=ax3,
    )

    # Plot 4: Confusion matrix for Method 1B
    ax4 = axes[1, 1]
    BiomarkerMetrics.plot_confusion_matrix_heatmap(
        y_test_1b,
        y_pred_1b,
        title=f"Confusion Matrix - {results_1b['method']}",
        ax=ax4,
    )

    plt.tight_layout()

    # Save figure
    output_path = output_dir / f"{disease_name.lower().replace(' ', '_')}_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved comparison plot to {output_path}")
    plt.close()


def run_experiment_for_disease(
    disease_key: str,
    config: Dict,
    data_dir: Path,
    output_dir: Path,
    random_state: int = 42,
) -> Dict:
    """Run complete Phase 1 experiment for a single disease.

    Args:
        disease_key: Disease identifier (e.g., 'rheumatoid_arthritis')
        config: Configuration dictionary
        data_dir: Path to MIMIC-IV data directory
        output_dir: Directory to save results
        random_state: Random seed for reproducibility

    Returns:
        Dictionary containing all experiment results
    """
    disease_name = config["diseases"][disease_key]["name"]
    logger.info(f"\n{'#'*70}")
    logger.info(f"# Running Phase 1 Experiments for: {disease_name}")
    logger.info(f"{'#'*70}\n")

    # Prepare dataset
    try:
        X_train, X_test, y_train, y_test, features, dataset_info = prepare_dataset(
            disease_key, config, data_dir, random_state
        )
    except Exception as e:
        logger.error(f"Failed to prepare dataset: {e}")
        return {
            "disease_key": disease_key,
            "disease_name": disease_name,
            "status": "failed",
            "error": str(e),
        }

    # Get literature thresholds for this disease
    literature_thresholds = config["literature_thresholds"].get(disease_key, {})
    if not literature_thresholds:
        logger.warning(f"No literature thresholds found for {disease_key}. Using defaults.")
        # Use reference ranges as fallback
        literature_thresholds = {}
        for feature_name, feature_config in config["cbc_features"].items():
            if "reference_range" in feature_config:
                literature_thresholds[feature_name] = feature_config["reference_range"]

    # Method 1A: Literature-based threshold
    try:
        generator_1a = SingleFeatureLiteratureGenerator(
            config={"random_state": random_state},
            literature_thresholds=literature_thresholds,
        )
        results_1a = evaluate_biomarker(
            generator_1a,
            X_train,
            X_test,
            y_train,
            y_test,
            features,
            "Method 1A: Literature-Based Threshold",
        )
    except Exception as e:
        logger.error(f"Method 1A failed: {e}")
        results_1a = {"method": "Method 1A", "status": "failed", "error": str(e)}

    # Method 1B: Data-driven threshold (Youden's Index with CV)
    try:
        generator_1b = YoudensIndexGenerator(
            config={
                "random_state": random_state,
                "use_cv": True,
                "cv_folds": 5,
            }
        )
        results_1b = evaluate_biomarker(
            generator_1b,
            X_train,
            X_test,
            y_train,
            y_test,
            features,
            "Method 1B: Data-Driven Threshold (Youden's Index)",
        )
    except Exception as e:
        logger.error(f"Method 1B failed: {e}")
        results_1b = {"method": "Method 1B", "status": "failed", "error": str(e)}

    # Generate comparison plots
    if "status" not in results_1a and "status" not in results_1b:
        try:
            generate_comparison_plots(results_1a, results_1b, disease_name, output_dir)
        except Exception as e:
            logger.error(f"Failed to generate plots: {e}")

    # Compare methods
    logger.info(f"\n{'='*60}")
    logger.info(f"Method Comparison for {disease_name}")
    logger.info(f"{'='*60}")

    if "metrics" in results_1a and "metrics" in results_1b:
        comparison_table = pd.DataFrame(
            {
                "Method 1A (Literature)": results_1a["metrics"],
                "Method 1B (Data-Driven)": results_1b["metrics"],
            }
        ).T

        logger.info(f"\n{comparison_table.to_string()}")

        # Determine winner for each metric
        logger.info("\nBest Method by Metric:")
        for metric in results_1a["metrics"].keys():
            val_1a = results_1a["metrics"][metric]
            val_1b = results_1b["metrics"][metric]
            winner = "1A" if val_1a > val_1b else "1B" if val_1b > val_1a else "Tie"
            logger.info(f"  {metric:15s}: Method {winner}")

    # Compile experiment results
    experiment_results = {
        "disease_key": disease_key,
        "disease_name": disease_name,
        "dataset_info": dataset_info,
        "method_1a": results_1a,
        "method_1b": results_1b,
        "timestamp": datetime.now().isoformat(),
        "random_state": random_state,
    }

    # Save results to JSON
    results_path = output_dir / f"{disease_key}_results.json"
    with open(results_path, "w") as f:
        json.dump(experiment_results, f, indent=2)
    logger.info(f"\nSaved results to {results_path}")

    return experiment_results


def generate_summary_report(
    all_results: List[Dict], output_dir: Path
) -> None:
    """Generate comprehensive summary report across all diseases.

    Args:
        all_results: List of experiment results for all diseases
        output_dir: Directory to save summary report
    """
    logger.info("\n" + "="*70)
    logger.info("GENERATING SUMMARY REPORT")
    logger.info("="*70)

    # Create summary DataFrame
    summary_data = []
    for result in all_results:
        if result.get("status") == "failed":
            continue

        disease_name = result["disease_name"]
        dataset_info = result["dataset_info"]

        # Method 1A metrics
        if "metrics" in result["method_1a"]:
            metrics_1a = result["method_1a"]["metrics"]
            biomarker_1a = result["method_1a"]["biomarker"]["formula"]
        else:
            metrics_1a = {}
            biomarker_1a = "Failed"

        # Method 1B metrics
        if "metrics" in result["method_1b"]:
            metrics_1b = result["method_1b"]["metrics"]
            biomarker_1b = result["method_1b"]["biomarker"]["formula"]
        else:
            metrics_1b = {}
            biomarker_1b = "Failed"

        summary_data.append({
            "Disease": disease_name,
            "N_Samples": dataset_info["n_samples"],
            "Positive_Rate": f"{dataset_info['positive_rate']:.2%}",
            "1A_Formula": biomarker_1a,
            "1A_AUC": metrics_1a.get("auc_roc", 0),
            "1A_F1": metrics_1a.get("f1_score", 0),
            "1A_Sensitivity": metrics_1a.get("sensitivity", 0),
            "1A_Specificity": metrics_1a.get("specificity", 0),
            "1B_Formula": biomarker_1b,
            "1B_AUC": metrics_1b.get("auc_roc", 0),
            "1B_F1": metrics_1b.get("f1_score", 0),
            "1B_Sensitivity": metrics_1b.get("sensitivity", 0),
            "1B_Specificity": metrics_1b.get("specificity", 0),
        })

    summary_df = pd.DataFrame(summary_data)

    # Display summary table
    logger.info("\nPerformance Summary Table:")
    logger.info("\n" + summary_df.to_string(index=False))

    # Save summary to CSV
    csv_path = output_dir / "phase1_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    logger.info(f"\nSaved summary table to {csv_path}")

    # Calculate average performance
    logger.info("\n" + "-"*70)
    logger.info("Average Performance Across All Diseases:")
    logger.info("-"*70)
    for method in ["1A", "1B"]:
        auc_col = f"{method}_AUC"
        f1_col = f"{method}_F1"
        sens_col = f"{method}_Sensitivity"
        spec_col = f"{method}_Specificity"

        avg_auc = summary_df[auc_col].mean()
        avg_f1 = summary_df[f1_col].mean()
        avg_sens = summary_df[sens_col].mean()
        avg_spec = summary_df[spec_col].mean()

        logger.info(f"\nMethod {method}:")
        logger.info(f"  Average AUC-ROC:     {avg_auc:.4f}")
        logger.info(f"  Average F1 Score:    {avg_f1:.4f}")
        logger.info(f"  Average Sensitivity: {avg_sens:.4f}")
        logger.info(f"  Average Specificity: {avg_spec:.4f}")

    # Determine overall winner
    avg_auc_1a = summary_df["1A_AUC"].mean()
    avg_auc_1b = summary_df["1B_AUC"].mean()

    logger.info("\n" + "="*70)
    if avg_auc_1a > avg_auc_1b:
        logger.info("OVERALL WINNER: Method 1A (Literature-Based Threshold)")
        logger.info(f"  Average AUC advantage: {avg_auc_1a - avg_auc_1b:.4f}")
    elif avg_auc_1b > avg_auc_1a:
        logger.info("OVERALL WINNER: Method 1B (Data-Driven Threshold)")
        logger.info(f"  Average AUC advantage: {avg_auc_1b - avg_auc_1a:.4f}")
    else:
        logger.info("RESULT: Methods are tied on average AUC")
    logger.info("="*70)


def main():
    """Main entry point for Phase 1 experiments."""
    parser = argparse.ArgumentParser(
        description="Run Phase 1 biomarker discovery experiments"
    )
    parser.add_argument(
        "--disease",
        type=str,
        help="Specific disease to run (e.g., 'rheumatoid_arthritis'). "
        "If not specified, runs only on default disease.",
    )
    parser.add_argument(
        "--all-diseases",
        action="store_true",
        help="Run experiments on all 5 diseases",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Path to MIMIC-IV data directory (default: data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/phase1_results",
        help="Output directory for results (default: experiments/phase1_results)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    config = load_config()

    # Determine which diseases to run
    if args.all_diseases:
        diseases_to_run = list(config["diseases"].keys())
        logger.info(f"Running experiments on all {len(diseases_to_run)} diseases")
    elif args.disease:
        if args.disease not in config["diseases"]:
            logger.error(f"Unknown disease: {args.disease}")
            logger.error(f"Available diseases: {list(config['diseases'].keys())}")
            sys.exit(1)
        diseases_to_run = [args.disease]
        logger.info(f"Running experiment on {args.disease}")
    else:
        # Run on default disease (Rheumatoid Arthritis)
        default_disease = config.get("default_disease", "rheumatoid_arthritis")
        diseases_to_run = [default_disease]
        logger.info(f"Running experiment on default disease: {default_disease}")

    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Random state: {args.random_state}")

    # Run experiments
    all_results = []
    for disease_key in diseases_to_run:
        try:
            result = run_experiment_for_disease(
                disease_key=disease_key,
                config=config,
                data_dir=data_dir,
                output_dir=output_dir,
                random_state=args.random_state,
            )
            all_results.append(result)
        except Exception as e:
            logger.error(f"Experiment failed for {disease_key}: {e}")
            import traceback
            logger.error(traceback.format_exc())

    # Generate summary report if multiple diseases were run
    if len(all_results) > 1:
        try:
            generate_summary_report(all_results, output_dir)
        except Exception as e:
            logger.error(f"Failed to generate summary report: {e}")
            import traceback
            logger.error(traceback.format_exc())

    logger.info("\n" + "="*70)
    logger.info("Phase 1 Experiments Complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("="*70)


if __name__ == "__main__":
    main()
