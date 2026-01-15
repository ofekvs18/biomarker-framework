"""Data loading utilities for MIMIC-IV datasets."""

import pickle
from pathlib import Path
from typing import Optional, Union

import polars as pl
import yaml


class MIMICLoader:
    """Load and manage MIMIC-IV data files with caching support."""

    def __init__(self, data_dir: Path, config: dict):
        """Initialize MIMICLoader with data directory and configuration.

        Args:
            data_dir: Path to the raw MIMIC data directory containing parquet files.
            config: Configuration dictionary containing disease definitions and CBC features.
        """
        self.data_dir = Path(data_dir)
        self.config = config
        self.cache_dir = self.data_dir.parent / "processed" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache for loaded dataframes to avoid reloading
        self._cache = {}

        # Extract CBC itemids from config
        self._cbc_itemids = self._extract_cbc_itemids()

    def _extract_cbc_itemids(self) -> list[int]:
        """Extract all CBC-related itemids from the configuration.

        Returns:
            List of itemids for CBC features.
        """
        itemids = []
        cbc_features = self.config.get("cbc_features", {})
        for feature_name, feature_config in cbc_features.items():
            itemids.extend(feature_config.get("itemids", []))
        return itemids

    def _get_cache_path(self, dataset_name: str) -> Path:
        """Get the cache file path for a dataset.

        Args:
            dataset_name: Name of the dataset (e.g., 'admissions', 'diagnoses').

        Returns:
            Path to the cache file.
        """
        return self.cache_dir / f"{dataset_name}.pkl"

    def _load_from_cache(self, dataset_name: str) -> Optional[pl.DataFrame]:
        """Load dataset from cache if available.

        Args:
            dataset_name: Name of the dataset.

        Returns:
            Cached DataFrame or None if not cached.
        """
        # Check in-memory cache first
        if dataset_name in self._cache:
            return self._cache[dataset_name]

        # Check disk cache
        cache_path = self._get_cache_path(dataset_name)
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                df = pickle.load(f)
                self._cache[dataset_name] = df
                return df

        return None

    def _save_to_cache(self, dataset_name: str, df: pl.DataFrame) -> None:
        """Save dataset to cache.

        Args:
            dataset_name: Name of the dataset.
            df: DataFrame to cache.
        """
        # Save to in-memory cache
        self._cache[dataset_name] = df

        # Save to disk cache
        cache_path = self._get_cache_path(dataset_name)
        with open(cache_path, "wb") as f:
            pickle.dump(df, f)

    def load_admissions(self) -> pl.DataFrame:
        """Load hospital admissions data from parquet file.

        Returns:
            Polars DataFrame containing admission records with columns:
            - subject_id: Patient identifier
            - hadm_id: Hospital admission identifier
            - admittime: Admission timestamp
            - dischtime: Discharge timestamp
            - admission_type: Type of admission
            - And other admission-related fields
        """
        cached = self._load_from_cache("admissions")
        if cached is not None:
            return cached

        parquet_path = self.data_dir / "admissions.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Admissions file not found at {parquet_path}. "
                "Please run data download script first."
            )

        df = pl.read_parquet(parquet_path)
        self._save_to_cache("admissions", df)
        return df

    def load_diagnoses(self) -> pl.DataFrame:
        """Load diagnosis codes data from parquet file.

        Returns:
            Polars DataFrame containing diagnosis records with columns:
            - subject_id: Patient identifier
            - hadm_id: Hospital admission identifier
            - icd_code: ICD diagnosis code
            - icd_version: ICD version (9 or 10)
            - seq_num: Sequence number of diagnosis
        """
        cached = self._load_from_cache("diagnoses")
        if cached is not None:
            return cached

        parquet_path = self.data_dir / "diagnoses_icd.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Diagnoses file not found at {parquet_path}. "
                "Please run data download script first."
            )

        df = pl.read_parquet(parquet_path)
        self._save_to_cache("diagnoses", df)
        return df

    def load_patients(self) -> pl.DataFrame:
        """Load patient demographics data from parquet file.

        Returns:
            Polars DataFrame containing patient records with columns:
            - subject_id: Patient identifier
            - gender: Patient gender
            - anchor_age: Patient age at anchor year
            - anchor_year: De-identified anchor year
            - dod: Date of death (if applicable)
        """
        cached = self._load_from_cache("patients")
        if cached is not None:
            return cached

        parquet_path = self.data_dir / "patients.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Patients file not found at {parquet_path}. "
                "Please run data download script first."
            )

        df = pl.read_parquet(parquet_path)
        self._save_to_cache("patients", df)
        return df

    def load_lab_results(self) -> pl.DataFrame:
        """Load CBC laboratory test results from parquet file.

        Filters the labevents data to include only CBC-related tests based on
        itemids defined in the configuration.

        Returns:
            Polars DataFrame containing CBC lab results with columns:
            - subject_id: Patient identifier
            - hadm_id: Hospital admission identifier
            - itemid: Lab test item identifier
            - charttime: Time of measurement
            - value: Numeric lab value
            - valuenum: Numeric value
            - valueuom: Unit of measurement
            - And other lab-related fields
        """
        cached = self._load_from_cache("lab_results_cbc")
        if cached is not None:
            return cached

        parquet_path = self.data_dir / "labevents.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Lab events file not found at {parquet_path}. "
                "Please run data download script first."
            )

        # Load full labevents and filter for CBC itemids
        # Use lazy evaluation for efficiency with large files
        df = (
            pl.scan_parquet(parquet_path)
            .filter(pl.col("itemid").is_in(self._cbc_itemids))
            .collect()
        )

        self._save_to_cache("lab_results_cbc", df)
        return df

    def get_patients_with_disease(self, disease_key: str) -> list[int]:
        """Identify patients diagnosed with a specific disease.

        Joins diagnosis data with disease ICD codes from configuration to find
        all patients with the target disease.

        Args:
            disease_key: Key for the disease in the config (e.g., 'anemia', 'sepsis').

        Returns:
            List of subject_ids for patients with the disease.

        Raises:
            ValueError: If disease_key is not found in configuration.
        """
        # Get disease configuration
        diseases = self.config.get("diseases", {})
        if disease_key not in diseases:
            available_diseases = list(diseases.keys())
            raise ValueError(
                f"Disease '{disease_key}' not found in configuration. "
                f"Available diseases: {available_diseases}"
            )

        disease_config = diseases[disease_key]
        icd9_codes = disease_config.get("icd9_codes", [])
        icd10_codes = disease_config.get("icd10_codes", [])

        # Load diagnoses
        diagnoses_df = self.load_diagnoses()

        # Create ICD code patterns for matching
        # ICD codes may be prefixes (e.g., "250" matches "250.00", "250.01", etc.)
        icd9_patterns = []
        icd10_patterns = []

        for code in icd9_codes:
            # Check if code is a range (e.g., "714.0-714.9")
            if "-" in code:
                start, end = code.split("-")
                # Extract the base and generate range
                start_base = start.split(".")[0]
                start_decimal = float(start.split(".")[1]) if "." in start else 0
                end_decimal = float(end.split(".")[1]) if "." in end else 9

                # Add all codes in range
                for i in range(int(start_decimal), int(end_decimal) + 1):
                    icd9_patterns.append(f"{start_base}.{i}")
            else:
                icd9_patterns.append(code)

        for code in icd10_codes:
            icd10_patterns.append(code)

        # Filter diagnoses by ICD codes
        # For ICD-9, we need to handle prefix matching (e.g., "250" matches "250.00")
        # For ICD-10, exact matching or prefix for categories (e.g., "E10" matches "E10.9")

        patients_with_disease = set()

        # Filter ICD-9 codes (version 9)
        if icd9_patterns:
            icd9_matches = diagnoses_df.filter(
                pl.col("icd_version") == 9
            )

            for pattern in icd9_patterns:
                # Match codes that start with the pattern
                matches = icd9_matches.filter(
                    pl.col("icd_code").str.starts_with(pattern)
                )
                patient_ids = matches.select("subject_id").unique().to_series().to_list()
                patients_with_disease.update(patient_ids)

        # Filter ICD-10 codes (version 10)
        if icd10_patterns:
            icd10_matches = diagnoses_df.filter(
                pl.col("icd_version") == 10
            )

            for pattern in icd10_patterns:
                # Match codes that start with the pattern
                matches = icd10_matches.filter(
                    pl.col("icd_code").str.starts_with(pattern)
                )
                patient_ids = matches.select("subject_id").unique().to_series().to_list()
                patients_with_disease.update(patient_ids)

        return sorted(list(patients_with_disease))

    def clear_cache(self) -> None:
        """Clear all cached data (both in-memory and disk)."""
        self._cache.clear()

        # Remove cache files
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()


# Maintain backward compatibility with original DataLoader class
class DataLoader:
    """Load and manage MIMIC-IV data files (legacy interface).

    This class is kept for backward compatibility. New code should use MIMICLoader.
    """

    def __init__(self, data_dir: Union[str, Path] = "data/raw/mimic_data"):
        """Initialize DataLoader with data directory path.

        Args:
            data_dir: Path to the raw MIMIC data directory.
        """
        self.data_dir = Path(data_dir)

    def load_labevents(self) -> pl.DataFrame:
        """Load laboratory events data."""
        parquet_path = self.data_dir / "labevents.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"Lab events file not found at {parquet_path}")
        return pl.read_parquet(parquet_path)

    def load_patients(self) -> pl.DataFrame:
        """Load patient demographics data."""
        parquet_path = self.data_dir / "patients.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"Patients file not found at {parquet_path}")
        return pl.read_parquet(parquet_path)

    def load_admissions(self) -> pl.DataFrame:
        """Load hospital admissions data."""
        parquet_path = self.data_dir / "admissions.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"Admissions file not found at {parquet_path}")
        return pl.read_parquet(parquet_path)

    def load_diagnoses(self) -> pl.DataFrame:
        """Load diagnosis codes data."""
        parquet_path = self.data_dir / "diagnoses_icd.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"Diagnoses file not found at {parquet_path}")
        return pl.read_parquet(parquet_path)


def load_config(config_dir: Union[str, Path] = "configs") -> dict:
    """Load configuration files (diseases and CBC features).

    Args:
        config_dir: Path to the configuration directory.

    Returns:
        Dictionary containing combined configuration from all config files.
    """
    config_dir = Path(config_dir)
    config = {}

    # Load diseases config
    diseases_path = config_dir / "diseases.yaml"
    if diseases_path.exists():
        with open(diseases_path, "r") as f:
            diseases_config = yaml.safe_load(f)
            config.update(diseases_config)

    # Load CBC features config
    cbc_path = config_dir / "cbc_features.yaml"
    if cbc_path.exists():
        with open(cbc_path, "r") as f:
            cbc_config = yaml.safe_load(f)
            config.update(cbc_config)

    return config
