"""Unit tests for data loader module."""

import pickle
import tempfile
from pathlib import Path

import polars as pl
import pytest

from src.data.loader import MIMICLoader, DataLoader, load_config


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config():
    """Create sample configuration for testing."""
    return {
        "diseases": {
            "rheumatoid_arthritis": {
                "name": "Rheumatoid Arthritis",
                "icd9_codes": ["714.0", "714.1", "714.2"],
                "icd10_codes": ["M05", "M06"],
                "relevant_biomarkers": ["wbc", "platelets", "hemoglobin"],
            },
            "crohns_disease": {
                "name": "Crohn's Disease",
                "icd9_codes": ["555.0", "555.1", "555.2"],
                "icd10_codes": ["K50"],
                "relevant_biomarkers": ["wbc", "platelets", "hemoglobin"],
            },
            "diabetes_type2": {
                "name": "Diabetes Type 2",
                "icd9_codes": ["250.00", "250.02"],
                "icd10_codes": ["E11"],
                "relevant_biomarkers": ["wbc"],
            },
        },
        "cbc_features": {
            "hemoglobin": {
                "itemids": [51222, 50811],
                "unit": "g/dL",
                "normal_range": [12.0, 17.0],
            },
            "wbc": {
                "itemids": [51301, 51300],
                "unit": "K/uL",
                "normal_range": [4.5, 11.0],
            },
            "platelets": {
                "itemids": [51265],
                "unit": "K/uL",
                "normal_range": [150, 400],
            },
        },
    }


@pytest.fixture
def sample_admissions(temp_data_dir):
    """Create sample admissions data."""
    data = {
        "subject_id": [1001, 1002, 1003, 1004],
        "hadm_id": [2001, 2002, 2003, 2004],
        "admittime": [
            "2020-01-01 10:00:00",
            "2020-01-02 11:00:00",
            "2020-01-03 12:00:00",
            "2020-01-04 13:00:00",
        ],
        "dischtime": [
            "2020-01-05 10:00:00",
            "2020-01-06 11:00:00",
            "2020-01-07 12:00:00",
            "2020-01-08 13:00:00",
        ],
        "admission_type": ["EMERGENCY", "ELECTIVE", "URGENT", "EMERGENCY"],
    }
    df = pl.DataFrame(data)
    parquet_path = temp_data_dir / "admissions.parquet"
    df.write_parquet(parquet_path)
    return df


@pytest.fixture
def sample_patients(temp_data_dir):
    """Create sample patients data."""
    data = {
        "subject_id": [1001, 1002, 1003, 1004],
        "gender": ["M", "F", "M", "F"],
        "anchor_age": [45, 62, 38, 71],
        "anchor_year": [2017, 2017, 2018, 2019],
    }
    df = pl.DataFrame(data)
    parquet_path = temp_data_dir / "patients.parquet"
    df.write_parquet(parquet_path)
    return df


@pytest.fixture
def sample_diagnoses(temp_data_dir):
    """Create sample diagnoses data."""
    data = {
        "subject_id": [1001, 1001, 1002, 1003, 1003, 1004],
        "hadm_id": [2001, 2001, 2002, 2003, 2003, 2004],
        "icd_code": ["714.0", "555.0", "250.00", "M05", "E11.9", "K50"],
        "icd_version": [9, 9, 9, 10, 10, 10],
        "seq_num": [1, 2, 1, 1, 2, 1],
    }
    df = pl.DataFrame(data)
    parquet_path = temp_data_dir / "diagnoses_icd.parquet"
    df.write_parquet(parquet_path)
    return df


@pytest.fixture
def sample_labevents(temp_data_dir):
    """Create sample lab events data."""
    data = {
        "subject_id": [1001, 1001, 1002, 1002, 1003, 1004, 1004],
        "hadm_id": [2001, 2001, 2002, 2002, 2003, 2004, 2004],
        "itemid": [51222, 51301, 51222, 51265, 50811, 51300, 51265],
        "charttime": [
            "2020-01-02 08:00:00",
            "2020-01-02 09:00:00",
            "2020-01-03 08:00:00",
            "2020-01-03 09:00:00",
            "2020-01-04 08:00:00",
            "2020-01-05 08:00:00",
            "2020-01-05 09:00:00",
        ],
        "valuenum": [13.5, 8.2, 11.0, 250.0, 14.2, 6.8, 180.0],
        "valueuom": ["g/dL", "K/uL", "g/dL", "K/uL", "g/dL", "K/uL", "K/uL"],
    }
    df = pl.DataFrame(data)
    parquet_path = temp_data_dir / "labevents.parquet"
    df.write_parquet(parquet_path)
    return df


@pytest.fixture
def loader(temp_data_dir, sample_config, sample_admissions, sample_patients, sample_diagnoses, sample_labevents):
    """Create a MIMICLoader instance with sample data."""
    loader = MIMICLoader(data_dir=temp_data_dir, config=sample_config)
    # Clear any cached data from previous tests to ensure clean state
    loader.clear_cache()
    return loader


class TestMIMICLoader:
    """Test cases for MIMICLoader class."""

    def test_initialization(self, temp_data_dir, sample_config):
        """Test that MIMICLoader initializes correctly."""
        loader = MIMICLoader(data_dir=temp_data_dir, config=sample_config)

        assert loader.data_dir == temp_data_dir
        assert loader.config == sample_config
        assert loader.cache_dir.exists()
        assert isinstance(loader._cache, dict)
        assert len(loader._cbc_itemids) == 5  # hemoglobin (2) + wbc (2) + platelets (1)

    def test_extract_cbc_itemids(self, loader):
        """Test CBC itemid extraction from config."""
        itemids = loader._cbc_itemids
        assert 51222 in itemids  # hemoglobin
        assert 50811 in itemids  # hemoglobin
        assert 51301 in itemids  # wbc
        assert 51300 in itemids  # wbc
        assert 51265 in itemids  # platelets
        assert len(itemids) == 5

    def test_load_admissions(self, loader, sample_admissions):
        """Test loading admissions data."""
        df = loader.load_admissions()

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 4
        assert "subject_id" in df.columns
        assert "hadm_id" in df.columns
        assert "admittime" in df.columns
        assert df["subject_id"].to_list() == [1001, 1002, 1003, 1004]

    def test_load_patients(self, loader, sample_patients):
        """Test loading patients data."""
        df = loader.load_patients()

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 4
        assert "subject_id" in df.columns
        assert "gender" in df.columns
        assert "anchor_age" in df.columns

    def test_load_diagnoses(self, loader, sample_diagnoses):
        """Test loading diagnoses data."""
        df = loader.load_diagnoses()

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 6
        assert "subject_id" in df.columns
        assert "icd_code" in df.columns
        assert "icd_version" in df.columns

    def test_load_lab_results(self, loader, sample_labevents):
        """Test loading and filtering lab results for CBC tests."""
        df = loader.load_lab_results()

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 7  # All test data has CBC itemids
        assert "itemid" in df.columns
        assert "valuenum" in df.columns

        # Verify only CBC itemids are present
        itemids = df["itemid"].unique().to_list()
        for itemid in itemids:
            assert itemid in loader._cbc_itemids

    def test_get_patients_with_disease_rheumatoid_arthritis(self, loader):
        """Test identifying patients with rheumatoid arthritis."""
        patient_ids = loader.get_patients_with_disease("rheumatoid_arthritis")

        # Patients 1001 (714.0) and 1003 (M05) have rheumatoid arthritis
        assert isinstance(patient_ids, list)
        assert 1001 in patient_ids  # ICD-9: 714.0
        assert 1003 in patient_ids  # ICD-10: M05
        assert len(patient_ids) == 2

    def test_get_patients_with_disease_crohns_disease(self, loader):
        """Test identifying patients with Crohn's disease."""
        patient_ids = loader.get_patients_with_disease("crohns_disease")

        # Patients 1001 (555.0) and 1004 (K50) have Crohn's disease
        assert 1001 in patient_ids  # ICD-9: 555.0
        assert 1004 in patient_ids  # ICD-10: K50
        assert len(patient_ids) == 2

    def test_get_patients_with_disease_diabetes(self, loader):
        """Test identifying patients with diabetes type 2."""
        patient_ids = loader.get_patients_with_disease("diabetes_type2")

        # Patients 1002 (250.00) and 1003 (E11.9) have diabetes type 2
        assert 1002 in patient_ids  # ICD-9: 250.00
        assert 1003 in patient_ids  # ICD-10: E11.9
        assert len(patient_ids) == 2

    def test_get_patients_with_disease_invalid(self, loader):
        """Test that invalid disease key raises ValueError."""
        with pytest.raises(ValueError, match="not found in configuration"):
            loader.get_patients_with_disease("invalid_disease")

    def test_caching_mechanism(self, loader):
        """Test that caching works correctly."""
        # Load admissions for the first time
        df1 = loader.load_admissions()

        # Check that it's cached in memory
        assert "admissions" in loader._cache

        # Check that cache file exists
        cache_path = loader._get_cache_path("admissions")
        assert cache_path.exists()

        # Load again - should come from cache
        df2 = loader.load_admissions()

        # Should be the same object from cache
        assert df1.equals(df2)

    def test_cache_persistence(self, temp_data_dir, sample_config, sample_admissions):
        """Test that cache persists across loader instances."""
        # Create first loader and load data
        loader1 = MIMICLoader(data_dir=temp_data_dir, config=sample_config)
        df1 = loader1.load_admissions()

        # Create second loader (new instance)
        loader2 = MIMICLoader(data_dir=temp_data_dir, config=sample_config)

        # Cache should be empty in memory initially
        assert "admissions" not in loader2._cache

        # But loading should retrieve from disk cache
        df2 = loader2.load_admissions()
        assert df1.equals(df2)

        # Now should be in memory cache
        assert "admissions" in loader2._cache

    def test_clear_cache(self, loader):
        """Test clearing cache."""
        # Load data to populate cache
        loader.load_admissions()
        loader.load_patients()

        # Verify cache exists
        assert len(loader._cache) == 2
        assert loader._get_cache_path("admissions").exists()
        assert loader._get_cache_path("patients").exists()

        # Clear cache
        loader.clear_cache()

        # Verify cache is cleared
        assert len(loader._cache) == 0
        assert not loader._get_cache_path("admissions").exists()
        assert not loader._get_cache_path("patients").exists()

    def test_missing_file_error(self, temp_data_dir, sample_config):
        """Test that missing files raise appropriate errors."""
        loader = MIMICLoader(data_dir=temp_data_dir, config=sample_config)

        # Remove all parquet files to simulate missing data
        for f in temp_data_dir.glob("*.parquet"):
            f.unlink()

        with pytest.raises(FileNotFoundError, match="not found"):
            loader.load_admissions()

        with pytest.raises(FileNotFoundError, match="not found"):
            loader.load_patients()

    def test_load_from_directory_structure(self, temp_data_dir, sample_config):
        """Test loading data from new directory structure (table/*.parquet)."""
        # Create directory structure
        admissions_dir = temp_data_dir / "admissions"
        admissions_dir.mkdir()

        # Create sample data split into multiple files
        data1 = {
            "subject_id": [1001, 1002],
            "hadm_id": [2001, 2002],
            "admittime": ["2020-01-01 10:00:00", "2020-01-02 11:00:00"],
            "dischtime": ["2020-01-05 10:00:00", "2020-01-06 11:00:00"],
            "admission_type": ["EMERGENCY", "ELECTIVE"],
        }
        data2 = {
            "subject_id": [1003, 1004],
            "hadm_id": [2003, 2004],
            "admittime": ["2020-01-03 12:00:00", "2020-01-04 13:00:00"],
            "dischtime": ["2020-01-07 12:00:00", "2020-01-08 13:00:00"],
            "admission_type": ["URGENT", "EMERGENCY"],
        }

        # Write to multiple parquet files in directory
        pl.DataFrame(data1).write_parquet(admissions_dir / "file-001.parquet")
        pl.DataFrame(data2).write_parquet(admissions_dir / "file-002.parquet")

        # Test loading
        loader = MIMICLoader(data_dir=temp_data_dir, config=sample_config)
        df = loader.load_admissions()

        # Should load all data from both files
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 4
        assert set(df["subject_id"].to_list()) == {1001, 1002, 1003, 1004}

    def test_load_lab_results_from_directory(self, temp_data_dir, sample_config):
        """Test loading lab results from directory structure with filtering."""
        # Create labevents directory
        labevents_dir = temp_data_dir / "labevents"
        labevents_dir.mkdir()

        # Create sample data with both CBC and non-CBC itemids
        data = {
            "subject_id": [1001, 1001, 1002, 1002],
            "hadm_id": [2001, 2001, 2002, 2002],
            "itemid": [51222, 99999, 51301, 88888],  # 51222 and 51301 are CBC
            "charttime": [
                "2020-01-02 08:00:00",
                "2020-01-02 09:00:00",
                "2020-01-03 08:00:00",
                "2020-01-03 09:00:00",
            ],
            "valuenum": [13.5, 100.0, 8.2, 200.0],
            "valueuom": ["g/dL", "unit", "K/uL", "unit"],
        }

        # Write to parquet file in directory
        pl.DataFrame(data).write_parquet(labevents_dir / "file-001.parquet")

        # Test loading with CBC filtering
        loader = MIMICLoader(data_dir=temp_data_dir, config=sample_config)
        df = loader.load_lab_results()

        # Should only load CBC itemids (51222 and 51301)
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 2
        assert set(df["itemid"].to_list()) == {51222, 51301}


class TestDataLoader:
    """Test cases for legacy DataLoader class."""

    def test_legacy_loader_initialization(self, temp_data_dir):
        """Test that legacy DataLoader initializes correctly."""
        loader = DataLoader(data_dir=temp_data_dir)
        assert loader.data_dir == temp_data_dir

    def test_legacy_load_admissions(self, temp_data_dir, sample_admissions):
        """Test legacy loader can load admissions."""
        loader = DataLoader(data_dir=temp_data_dir)
        df = loader.load_admissions()

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 4

    def test_legacy_load_patients(self, temp_data_dir, sample_patients):
        """Test legacy loader can load patients."""
        loader = DataLoader(data_dir=temp_data_dir)
        df = loader.load_patients()

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 4


class TestConfigLoader:
    """Test cases for configuration loading."""

    def test_load_config(self):
        """Test loading configuration from YAML files."""
        config = load_config("configs")

        # Should contain both diseases and cbc_features
        assert "diseases" in config
        assert "cbc_features" in config

        # Verify some diseases are loaded
        assert "rheumatoid_arthritis" in config["diseases"]
        assert "diabetes_type2" in config["diseases"]
        assert "crohns_disease" in config["diseases"]

        # Verify some CBC features are loaded
        assert "hemoglobin" in config["cbc_features"]
        assert "wbc" in config["cbc_features"]

    def test_load_config_missing_dir(self):
        """Test loading config from non-existent directory."""
        config = load_config("nonexistent_dir")

        # Should return empty dict if directory doesn't exist
        assert isinstance(config, dict)
