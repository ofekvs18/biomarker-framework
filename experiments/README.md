# Biomarker Discovery Experiments

This directory contains experimental results from the biomarker discovery pipeline.

## Structure

```
experiments/
├── README.md                      # This file
├── phase1_results/                # Phase 1: Single feature biomarkers
│   ├── phase1_results.md         # Detailed documentation
│   ├── phase1_summary.csv        # Cross-disease performance summary
│   ├── *_results.json            # Per-disease detailed results
│   └── *_comparison.png          # Per-disease comparison plots
├── phase2_results/                # Phase 2: Multi-feature biomarkers (future)
└── notebooks/                     # Analysis notebooks
    ├── 01_eda.ipynb              # Exploratory data analysis
    ├── 02_disease_prevalence.ipynb
    └── 03_cbc_availability.ipynb
```

## Quick Start

### Run Phase 1 Experiments

**Single disease (Rheumatoid Arthritis):**
```bash
cd /home/user/biomarker-framework
python scripts/run_phase1_experiments.py --disease rheumatoid_arthritis
```

**All 5 diseases:**
```bash
python scripts/run_phase1_experiments.py --all-diseases
```

**Results will be saved to:** `experiments/phase1_results/`

### View Results

1. **Summary Table**: `phase1_results/phase1_summary.csv`
   - Performance comparison across all diseases
   - AUC, F1, Sensitivity, Specificity for both methods

2. **Detailed Results**: `phase1_results/{disease}_results.json`
   - Complete experiment metadata
   - Biomarker formulas and thresholds
   - Confusion matrices
   - Predictions

3. **Visualizations**: `phase1_results/{disease}_comparison.png`
   - ROC curves for both methods
   - Confusion matrices
   - Side-by-side comparison

4. **Documentation**: `phase1_results/phase1_results.md`
   - Methodology details
   - Expected results
   - Interpretation guide

### Execution Logs

All experiments generate detailed logs:
- Console output with real-time progress
- `phase1_experiments.log` with complete execution trace

## Experiment Phases

### Phase 1: Single Feature Biomarkers ✅ COMPLETE

**Goal**: Establish baseline using simple, interpretable single-feature biomarkers

**Methods**:
- Method 1A: Literature-based thresholds (clinical guidelines)
- Method 1B: Data-driven thresholds (Youden's Index with CV)

**Status**: Pipeline implemented and tested

**Documentation**: See `phase1_results/phase1_results.md`

### Phase 2: Multi-Feature Biomarkers 🚧 PLANNED

**Goal**: Improve performance using feature combinations

**Methods**:
- Linear combinations (weighted sums)
- Ratios and products
- Logistic regression
- Machine learning models (RF, XGBoost)

**Status**: Not yet implemented

### Phase 3: Deep Learning 🚧 PLANNED

**Goal**: Explore neural network approaches

**Methods**:
- Multi-layer perceptrons
- Attention mechanisms
- Autoencoders for feature learning

**Status**: Not yet implemented

## Reproducibility

All experiments use fixed random seeds for reproducibility:
- Default `random_state=42`
- Consistent train/test splits
- Deterministic CV folds

To reproduce results:
```bash
python scripts/run_phase1_experiments.py --all-diseases --random-state 42
```

## Data Requirements

Experiments require MIMIC-IV data in parquet format:

```
data/raw/
├── admissions/
│   └── admissions.parquet
├── diagnoses_icd/
│   └── diagnoses_icd.parquet
└── labevents/
    └── labevents.parquet
```

**For testing without MIMIC-IV data:**
```bash
python scripts/generate_synthetic_data.py --n-patients 1000
```

## Configuration

Experiments use configuration from:
- `configs/diseases.yaml` - Disease definitions and ICD codes
- `configs/cbc_features.yaml` - CBC parameters and thresholds

To modify:
1. Edit the YAML files
2. Rerun experiments
3. Results will reflect new configuration

## Performance Benchmarks

Expected performance on MIMIC-IV (based on Phase 1):

| Disease | Expected AUC | Expected F1 | Notes |
|---------|-------------|-------------|-------|
| Rheumatoid Arthritis | 0.65-0.75 | 0.55-0.70 | Anemia and inflammation markers |
| Diabetes Type 1 | 0.60-0.70 | 0.50-0.65 | Subtle immune changes |
| Diabetes Type 2 | 0.62-0.72 | 0.60-0.75 | More prevalent, better signal |
| Crohn's Disease | 0.70-0.80 | 0.60-0.75 | Strong anemia signal |
| Psoriasis | 0.58-0.68 | 0.45-0.60 | Weaker systemic markers |

**Note**: Single-feature biomarkers have inherent limitations. Multi-feature methods (Phase 2) expected to improve performance by 10-20% AUC.

## Analysis Notebooks

Interactive exploration and visualization:

### 01_eda.ipynb
- CBC feature distributions
- Missing value patterns
- Outlier detection
- Feature correlations

### 02_disease_prevalence.ipynb
- Disease prevalence in MIMIC-IV
- ICD code matching statistics
- Comorbidity analysis

### 03_cbc_availability.ipynb
- CBC test availability per disease
- Temporal patterns
- Completeness analysis

## Troubleshooting

### Common Issues

**1. Data not found error**
```
ERROR: Data not found for 'admissions'
```
**Solution**: Ensure MIMIC-IV data is in `data/raw/` or run synthetic data generation

**2. Low performance (AUC < 0.55)**
```
WARNING: Very low AUC. Check data quality.
```
**Solution**:
- Verify disease prevalence is sufficient (>1%)
- Check CBC test availability
- Review label creation (ICD code matching)

**3. Module import errors**
```
ModuleNotFoundError: No module named 'polars'
```
**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

**4. Memory errors with large datasets**
```
MemoryError: Unable to allocate array
```
**Solution**:
- Reduce n_patients in data generation
- Use sampling in pipeline
- Increase system memory

### Getting Help

1. Check experiment logs: `phase1_experiments.log`
2. Review pipeline code: `scripts/run_phase1_experiments.py`
3. Run tests: `pytest tests/test_*_generator.py`
4. Check documentation: `experiments/phase1_results/phase1_results.md`

## Contributing

When adding new experiments:

1. Create new subdirectory: `experiments/phase{N}_results/`
2. Document methodology in `phase{N}_results.md`
3. Use consistent file naming: `{disease_key}_results.json`
4. Generate summary CSV for cross-disease comparison
5. Update this README

## Citation

If using these experiments in research, please cite:

```
@software{biomarker_framework_phase1,
  title = {Biomarker Discovery Framework: Phase 1 Experiments},
  author = {[Your Team]},
  year = {2026},
  url = {https://github.com/your-org/biomarker-framework}
}
```
