# Project Context: Medical ML Pipeline (Stateless/GCS-Centric)

## 1. Project Overview
This is a research-grade Machine Learning pipeline designed for medical data analysis. 
**Core Constraint:** The code runs on a university cluster with ephemeral (temporary) compute nodes and limited local storage.
**Storage Strategy:** All persistent data (raw, processed, models, results) lives in Google Cloud Storage (GCS).

## 2. Architecture: "Pull-Process-Push"
Because the compute environment is stateless, we strictly follow this flow:
1.  **Pull:** Stream data from GCS to memory (using `gcsfs`/`pandas`) or temp disk.
2.  **Process:** Execute logic (Cohort Generation, Training, Evaluation).
3.  **Push:** Immediately write artifacts (models, logs, metrics) back to GCS.

**Strict Rule:** Never rely on the local disk for long-term storage. If the node dies, local data is lost.

## 3. Tech Stack
* **Language:** Python 3.10+
* **Configuration:** `Hydra` (Compositional config management)
* **Storage I/O:** `gcsfs` (Direct GCS interface), `pandas`, `pyarrow`
* **ML Libraries:** `scikit-learn`, `xgboost`, `pytorch` (interchangeable)
* **Environment:** Conda

## 4. Directory Structure
```text
project_root/
├── conf/                  # Hydra Configuration
│   ├── config.yaml        # Main entry point
│   ├── model/             # Model-specific params (xgboost.yaml, etc.)
│   ├── processing/        # Cohort definition params (dates, diseases)
│   └── hydra/             # Hydra overrides (logging, launcher)
├── src/
│   ├── data_loader.py     # GCS reading logic
│   ├── cohort_gen.py      # Phase 1: Cleaning, Censoring, Labeling
│   ├── preprocessor.py    # Phase 2: Statistical Transforms (Scaler/Imputer)
│   ├── models.py          # Abstract Base Class + Implementations
│   └── utils.py           # Helper functions (GCS paths, logging)
├── main.py                # Entry point for Training
├── generate_cohort.py     # Entry point for Cohort Generation
└── environment.yaml       # Conda env definition