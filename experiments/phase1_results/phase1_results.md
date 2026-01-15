# Phase 1: Single Feature Biomarker Discovery Results

## Overview

This document describes the Phase 1 biomarker discovery pipeline and results structure. Phase 1 implements two baseline methods for discovering single-feature CBC biomarkers:

- **Method 1A**: Literature-based threshold selection
- **Method 1B**: Data-driven threshold optimization using Youden's Index

## Methodology

### Method 1A: Literature-Based Threshold

This method combines data-driven feature selection with domain knowledge:

1. Train logistic regression on all CBC features
2. Select the feature with the highest absolute coefficient (most predictive)
3. Apply clinically-established threshold for that feature from medical literature

**Rationale**: Leverages clinical expertise while using data to identify the most relevant biomarker.

**Implementation**: `src/generators/literature_threshold.py::SingleFeatureLiteratureGenerator`

### Method 1B: Data-Driven Threshold (Youden's Index)

This method optimizes both feature selection and threshold from data:

1. Train logistic regression on all CBC features
2. Select the feature with the highest absolute coefficient
3. Optimize threshold using Youden's Index (maximize sensitivity + specificity - 1)
4. Use 5-fold cross-validation to avoid overfitting

**Rationale**: Fully data-driven approach that adapts to specific disease patterns in the dataset.

**Implementation**: `src/generators/datadriven_threshold.py::YoudensIndexGenerator`

## Pipeline Execution

### Running Experiments

The end-to-end pipeline is implemented in `scripts/run_phase1_experiments.py`.

**Single disease:**
```bash
python scripts/run_phase1_experiments.py --disease rheumatoid_arthritis
```

**All diseases:**
```bash
python scripts/run_phase1_experiments.py --all-diseases
```

**Custom configuration:**
```bash
python scripts/run_phase1_experiments.py \
    --all-diseases \
    --data-dir data/raw \
    --output-dir experiments/phase1_results \
    --random-state 42
```

### Pipeline Steps

For each disease, the pipeline:

1. **Load Data**: Loads MIMIC-IV admissions, diagnoses, and lab results
2. **Create Labels**: Identifies patients with disease using ICD-9/10 codes
3. **Aggregate CBC**: Aggregates multiple CBC tests per patient (mean, min, max, std)
4. **Preprocess**: Handles missing values (median imputation) and normalizes features (z-score)
5. **Split Data**: Creates 70/30 train/test split with stratification
6. **Generate Biomarkers**: Applies both Method 1A and 1B
7. **Evaluate**: Computes performance metrics on test set
8. **Visualize**: Generates ROC curves and confusion matrices
9. **Compare**: Compares both methods and identifies best performer

## Output Structure

### Per-Disease Results

For each disease, the following files are generated:

```
experiments/phase1_results/
├── {disease_key}_results.json       # Complete experiment results
└── {disease_name}_comparison.png    # Visual comparison plots
```

#### Results JSON Structure

```json
{
  "disease_key": "rheumatoid_arthritis",
  "disease_name": "Rheumatoid Arthritis",
  "dataset_info": {
    "n_samples": 1000,
    "n_features": 14,
    "n_train": 700,
    "n_test": 300,
    "positive_rate": 0.05,
    "train_positive_rate": 0.05,
    "test_positive_rate": 0.05
  },
  "method_1a": {
    "method": "Method 1A: Literature-Based Threshold",
    "biomarker": {
      "formula": "wbc >= 11.0",
      "threshold": 11.0,
      "features_used": ["wbc"],
      "metadata": {
        "selected_feature": "wbc",
        "coefficient": 0.523,
        "threshold_direction": "high",
        "all_coefficients": {...},
        "training_accuracy": 0.954
      }
    },
    "metrics": {
      "accuracy": 0.950,
      "auc_roc": 0.678,
      "precision": 0.750,
      "recall": 0.600,
      "f1_score": 0.667,
      "sensitivity": 0.600,
      "specificity": 0.968
    },
    "confusion_matrix": {
      "tn": 276,
      "fp": 9,
      "fn": 6,
      "tp": 9
    }
  },
  "method_1b": {
    "method": "Method 1B: Data-Driven Threshold (Youden's Index)",
    "biomarker": {
      "formula": "wbc >= 8.4523",
      "threshold": 8.4523,
      "features_used": ["wbc"],
      "metadata": {
        "selected_feature": "wbc",
        "coefficient": 0.523,
        "threshold_direction": "high",
        "youden_index": 0.312,
        "optimization_method": "cv",
        "cv_folds": 5,
        "cv_thresholds": [8.23, 8.56, 8.41, 8.78, 8.29],
        "cv_threshold_std": 0.219,
        "training_accuracy": 0.957
      }
    },
    "metrics": {
      "accuracy": 0.940,
      "auc_roc": 0.712,
      "precision": 0.667,
      "recall": 0.733,
      "f1_score": 0.698,
      "sensitivity": 0.733,
      "specificity": 0.947
    },
    "confusion_matrix": {
      "tn": 270,
      "fp": 15,
      "fn": 4,
      "tp": 11
    }
  },
  "timestamp": "2026-01-15T13:45:00.000000",
  "random_state": 42
}
```

### Summary Report

When running on multiple diseases, a comprehensive summary is generated:

```
experiments/phase1_results/
├── phase1_summary.csv              # Performance metrics table
├── rheumatoid_arthritis_results.json
├── rheumatoid_arthritis_comparison.png
├── diabetes_type1_results.json
├── diabetes_type1_comparison.png
├── diabetes_type2_results.json
├── diabetes_type2_comparison.png
├── crohns_disease_results.json
├── crohns_disease_comparison.png
├── psoriasis_results.json
└── psoriasis_comparison.png
```

#### Summary CSV Format

| Disease | N_Samples | Positive_Rate | 1A_Formula | 1A_AUC | 1A_F1 | 1A_Sensitivity | 1A_Specificity | 1B_Formula | 1B_AUC | 1B_F1 | 1B_Sensitivity | 1B_Specificity |
|---------|-----------|---------------|------------|--------|-------|----------------|----------------|------------|--------|-------|----------------|----------------|
| Rheumatoid Arthritis | 1000 | 5.0% | wbc >= 11.0 | 0.678 | 0.667 | 0.600 | 0.968 | wbc >= 8.452 | 0.712 | 0.698 | 0.733 | 0.947 |
| Diabetes Type 1 | 850 | 3.0% | platelets <= 150.0 | 0.612 | 0.545 | 0.462 | 0.977 | platelets <= 142.3 | 0.651 | 0.587 | 0.538 | 0.965 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

## Visualization Outputs

Each disease generates a 2x2 comparison plot containing:

1. **Top Left**: ROC Curve for Method 1A (Literature-Based)
2. **Top Right**: ROC Curve for Method 1B (Data-Driven)
3. **Bottom Left**: Confusion Matrix for Method 1A
4. **Bottom Right**: Confusion Matrix for Method 1B

These plots enable visual comparison of:
- Discriminative power (AUC-ROC)
- True positive vs false positive trade-offs
- Prediction distributions

## Evaluation Metrics

The pipeline computes comprehensive performance metrics:

### Primary Metrics

- **AUC-ROC**: Area under the receiver operating characteristic curve (0-1, higher is better)
  - Measures overall discriminative ability
  - 0.5 = random guess, 1.0 = perfect separation

- **F1 Score**: Harmonic mean of precision and recall (0-1, higher is better)
  - Balances precision and recall
  - Useful for imbalanced datasets

- **Sensitivity (Recall)**: True positive rate
  - Proportion of disease cases correctly identified
  - Critical for medical screening applications

- **Specificity**: True negative rate
  - Proportion of healthy cases correctly identified
  - Important to avoid unnecessary interventions

### Secondary Metrics

- **Accuracy**: Overall correctness (can be misleading with class imbalance)
- **Precision**: Positive predictive value (proportion of positive predictions that are correct)
- **Confusion Matrix**: Detailed breakdown of predictions (TN, FP, FN, TP)

## Diseases Analyzed

Phase 1 experiments are conducted on 5 autoimmune/inflammatory diseases:

### 1. Rheumatoid Arthritis
- **ICD-9**: 714.0-714.9
- **ICD-10**: M05, M06
- **Relevant CBC**: WBC, platelets, hemoglobin (anemia from chronic inflammation)
- **Expected Pattern**: Elevated WBC and platelets, reduced hemoglobin

### 2. Diabetes Type 1
- **ICD-9**: 250.01, 250.03, 250.11, etc.
- **ICD-10**: E10
- **Relevant CBC**: WBC, neutrophils, lymphocytes
- **Expected Pattern**: Immune-related changes

### 3. Diabetes Type 2
- **ICD-9**: 250.00, 250.02, 250.10, etc.
- **ICD-10**: E11
- **Relevant CBC**: WBC, platelets, hemoglobin
- **Expected Pattern**: Elevated WBC (metabolic inflammation)

### 4. Crohn's Disease
- **ICD-9**: 555.0-555.9
- **ICD-10**: K50
- **Relevant CBC**: Hemoglobin, hematocrit, WBC, platelets, MCV, RDW
- **Expected Pattern**: Anemia (lower hemoglobin), elevated WBC/platelets, abnormal RDW

### 5. Psoriasis
- **ICD-9**: 696.1
- **ICD-10**: L40
- **Relevant CBC**: WBC, neutrophils, platelets
- **Expected Pattern**: Systemic inflammation markers

## Data Preprocessing

The pipeline applies consistent preprocessing to all diseases:

1. **Label Creation**:
   - Patient-level labels (not admission-level)
   - Prefix matching on ICD codes (e.g., "714" matches "714.0", "714.1", etc.)
   - Positive if disease present in ANY admission

2. **CBC Aggregation**:
   - Multiple tests per patient aggregated using mean values
   - 30-day lookback window from admission
   - Statistics computed: mean, min, max, std, count

3. **Missing Value Handling**:
   - Strategy: Median imputation
   - Applied after aggregation
   - Preserves distributional properties

4. **Feature Normalization**:
   - Method: Z-score standardization (mean=0, std=1)
   - Fitted on training set only
   - Same scaler applied to test set
   - Prevents data leakage

5. **Train/Test Split**:
   - Split: 70% train / 30% test
   - Stratified sampling to maintain class balance
   - Fixed random seed (42) for reproducibility

## Reproducibility

All experiments are fully reproducible:

- **Fixed Random Seed**: `random_state=42` used throughout
- **Deterministic Splits**: Train/test splits use fixed seed
- **CV Fold Consistency**: Cross-validation folds use same seed
- **Version Control**: All code and configs tracked in git
- **Logging**: Complete execution logs saved to `phase1_experiments.log`

## Expected Results

Based on the methodology, we expect:

### Method Comparison

**Method 1A (Literature-Based)** strengths:
- Higher specificity (fewer false positives)
- More interpretable (clinically-established thresholds)
- Robust across different datasets
- Requires less data

**Method 1B (Data-Driven)** strengths:
- Better adapted to specific dataset patterns
- Higher sensitivity (more true positives)
- Can discover non-standard cut-offs
- Higher AUC in most cases

### Performance Expectations

- **AUC-ROC**: 0.60-0.75 (single feature biomarkers have limited power)
- **F1 Score**: 0.50-0.70 (challenging due to class imbalance)
- **Sensitivity**: 0.50-0.80 (varies by disease and method)
- **Specificity**: 0.90-0.98 (high due to class imbalance)

### Winner Prediction

Method 1B (Data-Driven) is expected to outperform Method 1A on average AUC due to:
- Threshold optimization specific to dataset
- Cross-validation for robust threshold selection
- Flexibility to adapt to disease-specific patterns

However, Method 1A may win on:
- Interpretability and clinical acceptance
- Generalization to external datasets
- Specific diseases where literature thresholds are well-validated

## Next Steps: Phase 2

Phase 1 establishes baseline single-feature biomarkers. Phase 2 will extend to:

1. **Multi-Feature Combinations**: Weighted sums, ratios, and products of CBC features
2. **Machine Learning Models**: Logistic regression, Random Forest, XGBoost
3. **Feature Engineering**: Interaction terms, polynomial features, domain-specific ratios
4. **Ensemble Methods**: Combining multiple biomarker generators
5. **External Validation**: Testing on held-out time periods or different hospitals

## References

### Literature Thresholds

All literature-based thresholds are sourced from:
- Clinical laboratory reference ranges (Mayo Clinic, LabCorp)
- Disease-specific guidelines (ACR for RA, ADA for diabetes, etc.)
- Published biomarker studies

See `configs/cbc_features.yaml` for complete threshold specifications.

### Methods

- **Youden's Index**: Youden WJ. Index for rating diagnostic tests. Cancer. 1950;3(1):32-35.
- **Logistic Regression**: Hosmer DW, Lemeshow S. Applied Logistic Regression. 2nd ed. Wiley; 2000.
- **Cross-Validation**: Stone M. Cross-validatory choice and assessment of statistical predictions. J R Stat Soc Ser B. 1974;36(2):111-147.

## Contact

For questions about the Phase 1 pipeline or results:
- Review pipeline code: `scripts/run_phase1_experiments.py`
- Check generator implementations: `src/generators/`
- See tests for usage examples: `tests/test_*_generator.py`
