# Biomarker Discovery Framework

A framework for discovering and validating CBC (Complete Blood Count) biomarkers using MIMIC-IV clinical data.

## Project Structure

```
biomarker-framework/
├── data/
│   ├── raw/              # Link to mimic_data/
│   ├── processed/        # Cleaned datasets
│   └── splits/           # Train/val/test splits
├── src/
│   ├── data/             # Data loading and preprocessing
│   │   ├── loader.py
│   │   ├── preprocessor.py
│   │   └── feature_engineer.py
│   ├── models/           # Prediction models
│   │   ├── baseline.py
│   │   └── feature_selector.py
│   ├── generators/       # Biomarker generation strategies
│   │   ├── base.py
│   │   ├── literature_threshold.py
│   │   └── datadriven_threshold.py
│   └── evaluation/       # Metrics and evaluation
│       ├── metrics.py
│       └── evaluator.py
├── notebooks/            # Jupyter notebooks for EDA
├── configs/              # Configuration files
│   ├── diseases.yaml     # Disease definitions and ICD codes
│   └── cbc_features.yaml # CBC feature mappings
├── experiments/          # MLflow outputs
├── tests/                # Unit tests
└── scripts/              # Utility scripts
```

## Installation

```bash
# Clone and setup
cd biomarker-framework
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
conda install -c conda-forge google-cloud-sdk
```

## Data Setup

1. Download MIMIC-IV data from BigQuery:
   ```bash
   python scripts/download_mimic_from_bigquery.py --output-dir ../mimic_data
   ```

2. Create symbolic link to data (requires admin on Windows):
   ```bash
   # Windows (run as administrator)
   mklink /D data\raw\mimic_data ..\..\..\mimic_data

   # Linux/Mac
   ln -s ../../../mimic_data data/raw/mimic_data
   ```

## Quick Start

```python
from src.data import DataLoader, Preprocessor, FeatureEngineer
from src.generators import LiteratureThresholdGenerator
from src.models import BaselineModel
from src.evaluation import Evaluator

# Load data
loader = DataLoader("data/raw/mimic_data")
labevents = loader.load_labevents()

# Preprocess
preprocessor = Preprocessor()
clean_data = preprocessor.clean_labevents(labevents)

# Generate biomarker labels
generator = LiteratureThresholdGenerator()
biomarkers = generator.generate_labels(clean_data)

# Train model
model = BaselineModel(model_type="logistic_regression")
model.fit(X_train, y_train)

# Evaluate
evaluator = Evaluator()
results = evaluator.evaluate_model(model, X_test, y_test)
```

## Biomarker Generation Strategies

The framework supports interchangeable biomarker generation strategies:

### Coefficient-Based Methods (Phase 1)

1. **SingleFeatureLiteratureGenerator**: Uses logistic regression coefficients to select the most predictive feature, then applies literature-defined thresholds
2. **YoudensIndexGenerator**: Uses logistic regression coefficients for feature selection and optimizes threshold using Youden's Index

### SHAP-Based Methods (Phase 2)

3. **SHAPLiteratureThresholdGenerator**: Uses SHAP values from tree-based models to identify the most important feature, then applies literature-defined thresholds
4. **SHAPDataDrivenThresholdGenerator**: Uses SHAP values for feature selection and optimizes threshold using Youden's Index

All modern generators implement the `BiomarkerGenerator` interface with `generate()` and `apply()` methods.

## Configuration

### diseases.yaml
Defines target diseases with ICD-9/ICD-10 codes and relevant biomarkers.

### cbc_features.yaml
Maps MIMIC-IV lab item IDs to standardized CBC feature names with reference ranges, literature thresholds, and SHAP model configurations.

## SHAP-Based Feature Selection

### Overview

SHAP (SHapley Additive exPlanations) provides a unified, theoretically-grounded approach to explaining machine learning model predictions. Unlike simple coefficients from linear models, SHAP values:

- **Account for feature interactions**: Capture non-linear relationships and feature dependencies
- **Provide consistent attributions**: Based on game theory (Shapley values)
- **Work across model types**: Support tree-based models (RandomForest, XGBoost), linear models, and any model via KernelExplainer
- **Show directional effects**: Indicate whether high or low values of a feature predict the positive class

### When to Use SHAP vs Coefficients

**Use SHAP when:**
- Working with non-linear models (RandomForest, XGBoost, GradientBoosting)
- Features have complex interactions
- You need robust importance rankings that account for model behavior
- Interpretability across different model types is important

**Use Coefficients when:**
- Using linear models (LogisticRegression, LinearSVM)
- Speed is critical (SHAP is slower, especially KernelExplainer)
- Simple linear relationships are expected
- Computational resources are limited

**Computational Cost:**
- TreeExplainer: Fast (~seconds for 1000 samples)
- LinearExplainer: Fast (~seconds)
- KernelExplainer: Slow (~minutes for 100 samples) - use as fallback only

### Usage Example

```python
from src.generators.shap_literature_threshold import SHAPLiteratureThresholdGenerator
from src.generators.shap_datadriven_threshold import SHAPDataDrivenThresholdGenerator
from src.evaluation.evaluator import BiomarkerEvaluator
import yaml

# Load configuration
with open("configs/cbc_features.yaml") as f:
    config = yaml.safe_load(f)

# Extract literature thresholds for disease
literature_thresholds = {
    "hemoglobin": 12.0,
    "wbc": 11.0,
    "platelets": 150
}

# Method 1: SHAP + Literature Threshold
shap_lit_config = {
    "random_state": 42,
    "model_type": "RandomForest",  # or "XGBoost", "LogisticRegression"
    "n_estimators": 100,
    "max_depth": 10,
}
generator_1 = SHAPLiteratureThresholdGenerator(
    config=shap_lit_config,
    literature_thresholds=literature_thresholds
)

# Method 2: SHAP + Optimized Threshold (Youden's Index)
shap_opt_config = {
    "random_state": 42,
    "model_type": "RandomForest",
    "n_estimators": 100,
    "use_cv": True,  # Cross-validation for robust threshold
    "cv_folds": 5,
}
generator_2 = SHAPDataDrivenThresholdGenerator(config=shap_opt_config)

# Train and evaluate
evaluator = BiomarkerEvaluator(experiment_name="shap_comparison")

result_1 = evaluator.evaluate_ml_generator(
    generator=generator_1,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    feature_names=["hemoglobin", "wbc", "platelets"],
    generator_name="SHAP_Literature",
    disease="rheumatoid_arthritis",
    save_shap_plots=True  # Automatically saves SHAP visualizations
)

# Access results
print(f"Selected feature: {result_1['biomarker']['features_used'][0]}")
print(f"Formula: {result_1['biomarker']['formula']}")
print(f"F1 Score: {result_1['metrics']['f1']:.3f}")
print(f"SHAP importance: {result_1['biomarker']['metadata']['shap_importance']:.3f}")
```

### Running SHAP Baseline Experiments

```bash
# Run experiments for all diseases
python scripts/run_shap_baseline.py \
    --data_dir data/processed \
    --output_dir experiments/shap_baseline_results \
    --model_type RandomForest \
    --n_estimators 200

# Run for specific disease
python scripts/run_shap_baseline.py \
    --disease rheumatoid_arthritis \
    --model_type XGBoost

# Run with different model
python scripts/run_shap_baseline.py \
    --model_type LogisticRegression
```

The script will:
1. Run both coefficient-based and SHAP-based generators
2. Compare feature selections across methods
3. Generate performance comparison plots
4. Save SHAP visualizations (summary and bar plots)
5. Log all results to MLflow for tracking

### Interpreting SHAP Visualizations

#### SHAP Summary Plot (Beeswarm)
- **X-axis**: SHAP value (impact on model output)
- **Y-axis**: Features ranked by importance
- **Color**: Feature value (red=high, blue=low)
- **Each dot**: One patient sample

Example interpretation:
- High hemoglobin values (red dots) push predictions toward negative class (left)
- Low hemoglobin values (blue dots) push predictions toward positive class (right)
- The spread shows how consistently the feature affects predictions

#### SHAP Bar Plot
- **X-axis**: Mean absolute SHAP value (importance)
- **Y-axis**: Features ranked by importance
- Shows overall feature importance without directional effects
- Use for quick comparison of feature rankings

### Configuration Options

SHAP-specific settings in `configs/cbc_features.yaml`:

```yaml
shap_config:
  default_model:
    model_type: "RandomForest"
    n_estimators: 100
    max_depth: 10
    min_samples_split: 5
    random_state: 42

  explainer:
    type: "auto"  # auto-detect best explainer
    background_samples: 100  # for KernelExplainer

  cross_validation:
    use_cv: true
    cv_folds: 5
```

### Performance Considerations

**Model Training:**
- RandomForest (100 trees, depth=10): ~1-2 seconds for 1000 samples
- XGBoost (100 trees): ~2-3 seconds for 1000 samples
- LogisticRegression: ~0.5 seconds for 1000 samples

**SHAP Computation:**
- TreeExplainer (RF/XGBoost): ~1-5 seconds for 1000 samples
- LinearExplainer (LR): ~1-2 seconds for 1000 samples
- KernelExplainer: ~30-300 seconds for 100 samples (avoid if possible)

**Total Pipeline:**
- SHAP-based method: ~5-10 seconds per disease
- Coefficient-based method: ~1-2 seconds per disease

### SHAP vs Coefficient Comparison

| Aspect | Coefficients | SHAP |
|--------|-------------|------|
| Speed | Fast (< 1s) | Moderate (5-10s) |
| Model Types | Linear only | Any model |
| Feature Interactions | No | Yes |
| Theoretical Foundation | Linear regression | Game theory |
| Interpretability | Simple | Rich (plots + values) |
| Best For | Linear relationships | Complex patterns |

### Testing

Run SHAP-specific tests:

```bash
# Test SHAP selector
pytest tests/test_shap_selector.py -v

# Test SHAP generators
pytest tests/test_shap_generators.py -v

# Run all tests
pytest tests/ -v
```

## Development

```bash
# Run tests
pytest tests/

# Format code
black src/
isort src/
```

## License

This project uses MIMIC-IV data, which requires PhysioNet credentialed access.
