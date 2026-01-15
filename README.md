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

1. **LiteratureThresholdGenerator**: Uses clinically-established reference ranges
2. **DataDrivenThresholdGenerator**: Learns thresholds from data distribution

Both implement the `BaseBiomarkerGenerator` interface for easy swapping.

## Configuration

### diseases.yaml
Defines target diseases with ICD-9/ICD-10 codes and relevant biomarkers.

### cbc_features.yaml
Maps MIMIC-IV lab item IDs to standardized CBC feature names with reference ranges.

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
