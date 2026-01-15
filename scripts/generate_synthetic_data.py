"""Generate synthetic MIMIC-IV data for testing Phase 1 pipeline.

This script creates synthetic parquet files that mimic the structure of MIMIC-IV data
to enable testing of the biomarker discovery pipeline without requiring actual patient data.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import polars as pl

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def generate_synthetic_mimic_data(output_dir: Path, n_patients: int = 1000):
    """Generate synthetic MIMIC-IV data files.

    Args:
        output_dir: Directory to save parquet files
        n_patients: Number of synthetic patients to generate
    """
    print(f"Generating synthetic MIMIC-IV data for {n_patients} patients...")
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)

    # Generate patient IDs
    patient_ids = np.arange(10000, 10000 + n_patients)

    # 1. Generate admissions data
    print("Generating admissions...")
    admissions_data = []
    admission_id = 20000

    for patient_id in patient_ids:
        # Each patient has 1-3 admissions
        n_admissions = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])

        base_date = datetime(2020, 1, 1) + timedelta(days=np.random.randint(0, 365))

        for i in range(n_admissions):
            admit_date = base_date + timedelta(days=i*90)
            discharge_date = admit_date + timedelta(days=np.random.randint(2, 14))

            admissions_data.append({
                "subject_id": patient_id,
                "hadm_id": admission_id,
                "admittime": admit_date,
                "dischtime": discharge_date,
                "admission_type": np.random.choice(["EMERGENCY", "ELECTIVE", "URGENT"]),
            })
            admission_id += 1

    admissions_df = pl.DataFrame(admissions_data)
    admissions_dir = output_dir / "admissions"
    admissions_dir.mkdir(exist_ok=True)
    admissions_df.write_parquet(admissions_dir / "admissions.parquet")
    print(f"  Created {len(admissions_df)} admissions")

    # 2. Generate diagnoses data with disease prevalence
    print("Generating diagnoses...")

    # Disease prevalence rates (approximate)
    diseases = {
        "rheumatoid_arthritis": {"icd9": ["714.0", "714.1"], "icd10": ["M05", "M06"], "rate": 0.05},
        "diabetes_type1": {"icd9": ["250.01", "250.03"], "icd10": ["E10"], "rate": 0.03},
        "diabetes_type2": {"icd9": ["250.00", "250.02"], "icd10": ["E11"], "rate": 0.12},
        "crohns_disease": {"icd9": ["555.0", "555.1"], "icd10": ["K50"], "rate": 0.02},
        "psoriasis": {"icd9": ["696.1"], "icd10": ["L40"], "rate": 0.04},
    }

    diagnoses_data = []

    for _, row in admissions_df.iter_rows(named=True):
        patient_id = row["subject_id"]
        hadm_id = row["hadm_id"]

        # Assign diseases based on prevalence
        for disease_name, disease_info in diseases.items():
            if np.random.random() < disease_info["rate"]:
                # Use ICD-9 or ICD-10 randomly
                if np.random.random() < 0.5:
                    code = np.random.choice(disease_info["icd9"])
                    version = 9
                else:
                    code = np.random.choice(disease_info["icd10"])
                    version = 10

                diagnoses_data.append({
                    "subject_id": patient_id,
                    "hadm_id": hadm_id,
                    "icd_code": code,
                    "icd_version": version,
                })

        # Add some random other diagnoses
        for _ in range(np.random.randint(1, 5)):
            diagnoses_data.append({
                "subject_id": patient_id,
                "hadm_id": hadm_id,
                "icd_code": f"{np.random.randint(100, 999)}.{np.random.randint(0, 99)}",
                "icd_version": 9,
            })

    diagnoses_df = pl.DataFrame(diagnoses_data)
    diagnoses_dir = output_dir / "diagnoses_icd"
    diagnoses_dir.mkdir(exist_ok=True)
    diagnoses_df.write_parquet(diagnoses_dir / "diagnoses_icd.parquet")
    print(f"  Created {len(diagnoses_df)} diagnoses")

    # 3. Generate CBC lab results
    print("Generating lab results...")

    # CBC itemids from config
    cbc_tests = {
        51222: {"name": "hemoglobin", "normal_mean": 14.0, "normal_std": 2.0},
        51221: {"name": "hematocrit", "normal_mean": 42.0, "normal_std": 5.0},
        51279: {"name": "rbc", "normal_mean": 4.5, "normal_std": 0.5},
        51250: {"name": "mcv", "normal_mean": 90.0, "normal_std": 8.0},
        51248: {"name": "mch", "normal_mean": 30.0, "normal_std": 3.0},
        51249: {"name": "mchc", "normal_mean": 34.0, "normal_std": 2.0},
        51277: {"name": "rdw", "normal_mean": 13.0, "normal_std": 1.5},
        51301: {"name": "wbc", "normal_mean": 7.5, "normal_std": 2.5},
        51256: {"name": "neutrophils", "normal_mean": 55.0, "normal_std": 10.0},
        51244: {"name": "lymphocytes", "normal_mean": 30.0, "normal_std": 8.0},
        51254: {"name": "monocytes", "normal_mean": 5.0, "normal_std": 2.0},
        51200: {"name": "eosinophils", "normal_mean": 2.5, "normal_std": 1.5},
        51146: {"name": "basophils", "normal_mean": 0.5, "normal_std": 0.3},
        51265: {"name": "platelets", "normal_mean": 250.0, "normal_std": 60.0},
    }

    lab_data = []

    for _, row in admissions_df.iter_rows(named=True):
        patient_id = row["subject_id"]
        hadm_id = row["hadm_id"]
        admit_time = row["admittime"]

        # Generate CBC tests for this admission (1-5 tests within 30 days before admission)
        n_tests = np.random.randint(1, 6)

        for test_num in range(n_tests):
            # Test date within 30 days before admission
            test_date = admit_time - timedelta(days=np.random.randint(0, 30))

            # Generate values for all CBC parameters
            for itemid, test_info in cbc_tests.items():
                # Add some noise and occasional abnormal values
                if np.random.random() < 0.15:  # 15% abnormal values
                    value = np.random.normal(
                        test_info["normal_mean"] * np.random.choice([0.7, 1.3]),
                        test_info["normal_std"] * 1.5
                    )
                else:
                    value = np.random.normal(
                        test_info["normal_mean"],
                        test_info["normal_std"]
                    )

                # Ensure positive values
                value = max(0.1, value)

                lab_data.append({
                    "subject_id": patient_id,
                    "hadm_id": hadm_id,
                    "itemid": itemid,
                    "charttime": test_date,
                    "valuenum": value,
                })

    lab_df = pl.DataFrame(lab_data)
    lab_dir = output_dir / "labevents"
    lab_dir.mkdir(exist_ok=True)
    lab_df.write_parquet(lab_dir / "labevents.parquet")
    print(f"  Created {len(lab_df)} lab results")

    print("\nSynthetic data generation complete!")
    print(f"Data saved to: {output_dir}")
    print(f"\nSummary:")
    print(f"  Patients: {n_patients}")
    print(f"  Admissions: {len(admissions_df)}")
    print(f"  Diagnoses: {len(diagnoses_df)}")
    print(f"  Lab results: {len(lab_df)}")

    # Print disease prevalence
    print(f"\nDisease prevalence:")
    for disease_name, disease_info in diseases.items():
        disease_codes = disease_info["icd9"] + disease_info["icd10"]
        n_patients_with_disease = diagnoses_df.filter(
            pl.col("icd_code").str.starts_with(disease_codes[0])
        ).select("subject_id").unique().height
        print(f"  {disease_name}: {n_patients_with_disease} patients ({n_patients_with_disease/n_patients:.1%})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic MIMIC-IV data for testing")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Output directory for synthetic data (default: data/raw)"
    )
    parser.add_argument(
        "--n-patients",
        type=int,
        default=1000,
        help="Number of patients to generate (default: 1000)"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    generate_synthetic_mimic_data(output_dir, args.n_patients)
