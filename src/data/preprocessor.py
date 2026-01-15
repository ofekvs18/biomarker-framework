"""Data preprocessing and cleaning utilities for CBC biomarker analysis."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .loader import MIMICLoader, load_config


class CBCPreprocessor:
    """Preprocess CBC data for biomarker discovery.

    This class handles the complete preprocessing pipeline including:
    - Label creation from diagnosis codes
    - CBC test aggregation for patients with multiple tests
    - Missing value imputation
    - Feature normalization
    - Dataset creation for modeling
    """

    def __init__(self, config: dict):
        """Initialize CBCPreprocessor with configuration.

        Args:
            config: Configuration dictionary containing disease definitions
                   and CBC features. Should include 'diseases' and 'cbc_features' keys.
        """
        self.config = config
        self.diseases = config.get("diseases", {})
        self.cbc_features = config.get("cbc_features", {})

        # Create mapping from itemid to feature name for easier processing
        self.itemid_to_feature = {}
        for feature_name, feature_config in self.cbc_features.items():
            for itemid in feature_config.get("itemids", []):
                self.itemid_to_feature[itemid] = feature_name

        # Store fitted scalers for normalization
        self._scalers: Dict[str, Union[StandardScaler, MinMaxScaler]] = {}

    def create_labels(
        self,
        admissions_df: pl.DataFrame,
        diagnoses_df: pl.DataFrame,
        disease_key: str
    ) -> pl.DataFrame:
        """Create binary labels indicating disease presence for each patient.

        Creates labels at the patient level (not admission level) to handle
        patients with multiple admissions. A patient is labeled as positive (1)
        if they have the disease in any admission, negative (0) otherwise.

        Args:
            admissions_df: DataFrame containing admission records with subject_id and hadm_id.
            diagnoses_df: DataFrame containing diagnosis records with subject_id, hadm_id,
                         icd_code, and icd_version.
            disease_key: Key for the disease in config (e.g., 'rheumatoid_arthritis').

        Returns:
            DataFrame with columns:
                - subject_id: Patient identifier
                - label: Binary label (1 if patient has disease, 0 otherwise)

        Raises:
            ValueError: If disease_key is not found in configuration.
        """
        # Validate disease key
        if disease_key not in self.diseases:
            available_diseases = list(self.diseases.keys())
            raise ValueError(
                f"Disease '{disease_key}' not found in configuration. "
                f"Available diseases: {available_diseases}"
            )

        disease_config = self.diseases[disease_key]
        icd9_codes = disease_config.get("icd9_codes", [])
        icd10_codes = disease_config.get("icd10_codes", [])

        # Get all unique patients from admissions
        all_patients = admissions_df.select("subject_id").unique()

        # Find patients with the disease using ICD code matching
        patients_with_disease = set()

        # Match ICD-9 codes
        if icd9_codes:
            for code in icd9_codes:
                matches = diagnoses_df.filter(
                    (pl.col("icd_version") == 9) &
                    (pl.col("icd_code").str.starts_with(code))
                )
                patient_ids = matches.select("subject_id").unique().to_series().to_list()
                patients_with_disease.update(patient_ids)

        # Match ICD-10 codes
        if icd10_codes:
            for code in icd10_codes:
                matches = diagnoses_df.filter(
                    (pl.col("icd_version") == 10) &
                    (pl.col("icd_code").str.starts_with(code))
                )
                patient_ids = matches.select("subject_id").unique().to_series().to_list()
                patients_with_disease.update(patient_ids)

        # Create labels dataframe
        labels_df = all_patients.with_columns(
            pl.col("subject_id").map_elements(
                lambda x: 1 if x in patients_with_disease else 0,
                return_dtype=pl.Int32
            ).alias("label")
        )

        return labels_df

    def aggregate_cbc_tests(
        self,
        lab_df: pl.DataFrame,
        lookback_days: int = 30
    ) -> pl.DataFrame:
        """Aggregate multiple CBC tests per patient into summary statistics.

        For patients with multiple CBC tests, computes mean, min, max, and
        standard deviation for each feature. Optionally filters to tests
        within a lookback window.

        Args:
            lab_df: DataFrame containing CBC lab results with columns:
                   - subject_id: Patient identifier
                   - itemid: Lab test item identifier
                   - charttime: Time of measurement
                   - valuenum: Numeric lab value
            lookback_days: Number of days to look back from most recent test.
                          If None, use all tests. Default is 30 days.

        Returns:
            DataFrame with aggregated features. Columns include:
                - subject_id: Patient identifier
                - {feature}_mean: Mean value for each CBC feature
                - {feature}_min: Minimum value
                - {feature}_max: Maximum value
                - {feature}_std: Standard deviation
                - {feature}_count: Number of tests
        """
        # Convert itemid to feature name
        lab_df = lab_df.with_columns(
            pl.col("itemid").map_elements(
                lambda x: self.itemid_to_feature.get(x, f"unknown_{x}"),
                return_dtype=pl.Utf8
            ).alias("feature_name")
        )

        # Filter out unknown features
        lab_df = lab_df.filter(~pl.col("feature_name").str.starts_with("unknown_"))

        # Apply lookback window if specified
        if lookback_days is not None and "charttime" in lab_df.columns:
            # For each patient, find their most recent test
            max_times = lab_df.group_by("subject_id").agg(
                pl.col("charttime").max().alias("max_charttime")
            )

            # Join back and filter
            lab_df = lab_df.join(max_times, on="subject_id")
            lab_df = lab_df.with_columns(
                (pl.col("max_charttime") - pl.col("charttime")).alias("days_back")
            )
            # Filter to lookback window
            lab_df = lab_df.filter(
                pl.col("days_back") <= pl.duration(days=lookback_days)
            )

        # Aggregate by patient and feature
        aggregated = lab_df.group_by(["subject_id", "feature_name"]).agg([
            pl.col("valuenum").mean().alias("mean_value"),
            pl.col("valuenum").min().alias("min_value"),
            pl.col("valuenum").max().alias("max_value"),
            pl.col("valuenum").std().alias("std_value"),
            pl.col("valuenum").count().alias("count_value")
        ])

        # Pivot to wide format with feature names as columns
        # Create separate dataframes for each statistic and join them
        result_df = None

        for stat in ["mean", "min", "max", "std", "count"]:
            stat_df = aggregated.select([
                "subject_id",
                "feature_name",
                f"{stat}_value"
            ]).pivot(
                values=f"{stat}_value",
                index="subject_id",
                on="feature_name"
            )

            # Rename columns to include statistic suffix
            if stat != "mean":  # Keep mean as default without suffix for backwards compatibility
                rename_dict = {
                    col: f"{col}_{stat}"
                    for col in stat_df.columns
                    if col != "subject_id"
                }
                stat_df = stat_df.rename(rename_dict)

            # Join with result
            if result_df is None:
                result_df = stat_df
            else:
                result_df = result_df.join(stat_df, on="subject_id", how="full", coalesce=True)

        return result_df

    def handle_missing_values(
        self,
        cbc_df: pl.DataFrame,
        strategy: str = "median"
    ) -> pl.DataFrame:
        """Handle missing values in CBC feature data.

        Applies imputation strategy to handle missing CBC values. Missing values
        can occur when:
        - A patient doesn't have a particular CBC test
        - A test result is invalid or out of range
        - Data is incomplete

        Args:
            cbc_df: DataFrame with CBC features (may contain null values).
            strategy: Imputation strategy. Options:
                     - 'median': Replace with median of non-null values (default)
                     - 'mean': Replace with mean of non-null values
                     - 'drop': Drop rows with any missing values
                     - 'zero': Replace with zero

        Returns:
            DataFrame with missing values handled according to strategy.

        Raises:
            ValueError: If strategy is not recognized.
        """
        valid_strategies = ["median", "mean", "drop", "zero"]
        if strategy not in valid_strategies:
            raise ValueError(
                f"Invalid strategy '{strategy}'. "
                f"Must be one of {valid_strategies}"
            )

        # Get feature columns (exclude subject_id and label if present)
        id_cols = ["subject_id", "label", "hadm_id"]
        feature_cols = [col for col in cbc_df.columns if col not in id_cols]

        if strategy == "drop":
            # Drop rows with any missing values in feature columns
            return cbc_df.drop_nulls(subset=feature_cols)

        elif strategy == "median":
            # Compute median for each feature and fill nulls
            fill_values = {}
            for col in feature_cols:
                median_val = cbc_df.select(pl.col(col).median()).item()
                if median_val is not None:
                    fill_values[col] = median_val

            # Fill nulls with computed medians
            for col, fill_val in fill_values.items():
                cbc_df = cbc_df.with_columns(
                    pl.col(col).fill_null(fill_val)
                )

        elif strategy == "mean":
            # Compute mean for each feature and fill nulls
            fill_values = {}
            for col in feature_cols:
                mean_val = cbc_df.select(pl.col(col).mean()).item()
                if mean_val is not None:
                    fill_values[col] = mean_val

            # Fill nulls with computed means
            for col, fill_val in fill_values.items():
                cbc_df = cbc_df.with_columns(
                    pl.col(col).fill_null(fill_val)
                )

        elif strategy == "zero":
            # Fill nulls with zero
            for col in feature_cols:
                cbc_df = cbc_df.with_columns(
                    pl.col(col).fill_null(0.0)
                )

        return cbc_df

    def normalize_features(
        self,
        cbc_df: pl.DataFrame,
        method: str = "standard",
        fit: bool = True
    ) -> pl.DataFrame:
        """Normalize CBC features using standard or min-max scaling.

        Normalization is important for:
        - Features with different scales (e.g., WBC in thousands, hemoglobin in g/dL)
        - Machine learning algorithms sensitive to feature scales
        - Comparing feature importance across different CBC parameters

        Args:
            cbc_df: DataFrame with CBC features to normalize.
            method: Normalization method. Options:
                   - 'standard': Z-score normalization (mean=0, std=1)
                   - 'minmax': Min-max scaling to [0, 1] range
            fit: Whether to fit the scaler on this data (True for training)
                or use previously fitted scaler (False for test/validation).

        Returns:
            DataFrame with normalized features.

        Raises:
            ValueError: If method is not recognized.
            RuntimeError: If fit=False but no scaler has been fitted yet.
        """
        valid_methods = ["standard", "minmax"]
        if method not in valid_methods:
            raise ValueError(
                f"Invalid method '{method}'. "
                f"Must be one of {valid_methods}"
            )

        # Get feature columns (exclude subject_id and label if present)
        id_cols = ["subject_id", "label", "hadm_id"]
        feature_cols = [col for col in cbc_df.columns if col not in id_cols]

        if not feature_cols:
            return cbc_df

        # Create or retrieve scaler
        if fit:
            if method == "standard":
                scaler = StandardScaler()
            else:  # minmax
                scaler = MinMaxScaler()

            # Fit scaler on feature data
            feature_data = cbc_df.select(feature_cols).to_numpy()
            scaler.fit(feature_data)

            # Store for later use
            self._scalers[method] = scaler
        else:
            # Use previously fitted scaler
            if method not in self._scalers:
                raise RuntimeError(
                    f"No scaler fitted for method '{method}'. "
                    "Call with fit=True first on training data."
                )
            scaler = self._scalers[method]

        # Transform features
        feature_data = cbc_df.select(feature_cols).to_numpy()
        normalized_data = scaler.transform(feature_data)

        # Create dictionary with normalized features
        normalized_dict = {}
        for i, col in enumerate(feature_cols):
            normalized_dict[col] = normalized_data[:, i]

        # Add back ID columns
        for col in id_cols:
            if col in cbc_df.columns:
                normalized_dict[col] = cbc_df[col].to_numpy()

        # Create new dataframe with all columns
        normalized_df = pl.DataFrame(normalized_dict)

        # Reorder columns to match original
        normalized_df = normalized_df.select(cbc_df.columns)

        return normalized_df

    def create_dataset(
        self,
        loader: MIMICLoader,
        disease_key: str,
        lookback_days: int = 30,
        missing_strategy: str = "median",
        normalize_method: Optional[str] = "standard",
        min_tests: int = 1
    ) -> pl.DataFrame:
        """Create a complete preprocessed dataset for a specific disease.

        This is the main pipeline method that orchestrates all preprocessing steps:
        1. Load data from MIMICLoader
        2. Create labels for the disease
        3. Aggregate CBC tests per patient
        4. Handle missing values
        5. Normalize features (optional)
        6. Join labels with features

        Args:
            loader: MIMICLoader instance for data access.
            disease_key: Key for the disease in config (e.g., 'rheumatoid_arthritis').
            lookback_days: Number of days to look back for CBC tests. Default 30.
            missing_strategy: Strategy for handling missing values. Default 'median'.
            normalize_method: Normalization method ('standard', 'minmax', or None).
                            Default 'standard'.
            min_tests: Minimum number of CBC tests required per patient. Default 1.

        Returns:
            Complete dataset with features and labels ready for modeling.
            Columns include:
                - subject_id: Patient identifier
                - label: Binary disease label
                - CBC features (normalized if requested)
        """
        # Load data
        admissions_df = loader.load_admissions()
        diagnoses_df = loader.load_diagnoses()
        lab_df = loader.load_lab_results()

        # Create labels
        labels_df = self.create_labels(admissions_df, diagnoses_df, disease_key)

        # Aggregate CBC tests
        aggregated_df = self.aggregate_cbc_tests(lab_df, lookback_days)

        # Filter patients with minimum number of tests
        if min_tests > 1:
            # Check if we have count columns
            count_cols = [col for col in aggregated_df.columns if col.endswith("_count")]
            if count_cols:
                # Keep patients who have at least min_tests for any feature
                # This is a reasonable heuristic - patients engaged with CBC testing
                for col in count_cols:
                    aggregated_df = aggregated_df.filter(
                        (pl.col(col).is_null()) | (pl.col(col) >= min_tests)
                    )

        # Join features with labels
        dataset = aggregated_df.join(labels_df, on="subject_id", how="inner")

        # Handle missing values
        dataset = self.handle_missing_values(dataset, strategy=missing_strategy)

        # Normalize features if requested
        if normalize_method is not None:
            dataset = self.normalize_features(dataset, method=normalize_method, fit=True)

        return dataset


# Legacy Preprocessor class for backward compatibility
class Preprocessor:
    """Preprocess and clean MIMIC-IV data for analysis (legacy interface).

    This class is kept for backward compatibility. New code should use CBCPreprocessor.
    """

    def __init__(self):
        """Initialize Preprocessor."""
        pass

    def clean_labevents(self, df: pl.DataFrame) -> pl.DataFrame:
        """Clean laboratory events data.

        Args:
            df: Raw labevents DataFrame.

        Returns:
            Cleaned DataFrame with valid values.
        """
        raise NotImplementedError

    def handle_missing_values(
        self, df: pl.DataFrame, strategy: str = "median"
    ) -> pl.DataFrame:
        """Handle missing values in the dataset.

        Args:
            df: Input DataFrame.
            strategy: Imputation strategy ('median', 'mean', 'drop').

        Returns:
            DataFrame with missing values handled.
        """
        raise NotImplementedError

    def remove_outliers(
        self, df: pl.DataFrame, columns: List[str], method: str = "iqr"
    ) -> pl.DataFrame:
        """Remove outliers from specified columns.

        Args:
            df: Input DataFrame.
            columns: Columns to check for outliers.
            method: Outlier detection method ('iqr', 'zscore').

        Returns:
            DataFrame with outliers removed.
        """
        raise NotImplementedError
