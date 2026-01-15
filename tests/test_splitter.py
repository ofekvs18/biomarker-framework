"""Unit tests for data splitter module."""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from src.data.splitter import DataSplitter


@pytest.fixture
def sample_data():
    """Create sample dataset for testing."""
    # Create a sample dataset with 100 patients
    n_samples = 100

    # Generate patient IDs
    subject_ids = list(range(1000, 1000 + n_samples))

    # Generate labels (30% positive, 70% negative for imbalanced dataset)
    labels = [1] * 30 + [0] * 70

    # Generate dates (spread over 365 days)
    base_date = datetime(2023, 1, 1)
    dates = [base_date + timedelta(days=i) for i in range(n_samples)]

    # Generate some mock features
    wbc = [5.0 + i * 0.1 for i in range(n_samples)]
    hemoglobin = [13.0 + i * 0.05 for i in range(n_samples)]

    return pl.DataFrame({
        "subject_id": subject_ids,
        "label": labels,
        "admission_date": dates,
        "wbc": wbc,
        "hemoglobin": hemoglobin
    })


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for output files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestDataSplitterInit:
    """Test DataSplitter initialization."""

    def test_valid_initialization(self):
        """Test initialization with valid parameters."""
        splitter = DataSplitter(test_size=0.2, val_size=0.1)
        assert splitter.test_size == 0.2
        assert splitter.val_size == 0.1
        assert abs(splitter.train_size - 0.7) < 1e-10  # Use tolerance for floating point comparison

    def test_invalid_test_size(self):
        """Test initialization with invalid test_size."""
        with pytest.raises(ValueError):
            DataSplitter(test_size=1.5, val_size=0.1)

        with pytest.raises(ValueError):
            DataSplitter(test_size=-0.1, val_size=0.1)

    def test_invalid_val_size(self):
        """Test initialization with invalid val_size."""
        with pytest.raises(ValueError):
            DataSplitter(test_size=0.2, val_size=1.5)

        with pytest.raises(ValueError):
            DataSplitter(test_size=0.2, val_size=-0.1)

    def test_sizes_too_large(self):
        """Test initialization when test_size + val_size >= 1."""
        with pytest.raises(ValueError):
            DataSplitter(test_size=0.6, val_size=0.5)


class TestRandomSplit:
    """Test random splitting functionality."""

    def test_random_split_basic(self, sample_data):
        """Test basic random split functionality."""
        splitter = DataSplitter(test_size=0.2, val_size=0.1)
        train, val, test = splitter.split_random(sample_data, random_state=42)

        # Check sizes
        assert len(train) == 70  # 70% of 100
        assert len(val) == 10    # 10% of 100
        assert len(test) == 20   # 20% of 100

        # Check total matches original
        assert len(train) + len(val) + len(test) == len(sample_data)

    def test_random_split_reproducibility(self, sample_data):
        """Test that random split is reproducible with same seed."""
        splitter = DataSplitter(test_size=0.2, val_size=0.1)

        train1, val1, test1 = splitter.split_random(sample_data, random_state=42)
        train2, val2, test2 = splitter.split_random(sample_data, random_state=42)

        # Check that splits are identical
        assert train1.equals(train2)
        assert val1.equals(val2)
        assert test1.equals(test2)

    def test_random_split_different_seeds(self, sample_data):
        """Test that different seeds produce different splits."""
        splitter = DataSplitter(test_size=0.2, val_size=0.1)

        train1, _, _ = splitter.split_random(sample_data, random_state=42)
        train2, _, _ = splitter.split_random(sample_data, random_state=123)

        # Check that splits are different
        assert not train1.equals(train2)

    def test_random_split_no_data_leakage(self, sample_data):
        """Test that there's no overlap between splits."""
        splitter = DataSplitter(test_size=0.2, val_size=0.1)
        train, val, test = splitter.split_random(sample_data, random_state=42)

        train_ids = set(train["subject_id"].to_list())
        val_ids = set(val["subject_id"].to_list())
        test_ids = set(test["subject_id"].to_list())

        # Check no overlap
        assert len(train_ids & val_ids) == 0
        assert len(train_ids & test_ids) == 0
        assert len(val_ids & test_ids) == 0

        # Check all samples accounted for
        assert len(train_ids | val_ids | test_ids) == len(sample_data)


class TestTemporalSplit:
    """Test temporal splitting functionality."""

    def test_temporal_split_basic(self, sample_data):
        """Test basic temporal split functionality."""
        splitter = DataSplitter(test_size=0.2, val_size=0.1)
        train, val, test = splitter.split_temporal(sample_data, date_column="admission_date")

        # Check sizes
        assert len(train) == 70
        assert len(val) == 10
        assert len(test) == 20

        # Check temporal ordering
        train_max_date = train["admission_date"].max()
        val_min_date = val["admission_date"].min()
        val_max_date = val["admission_date"].max()
        test_min_date = test["admission_date"].min()

        # Training data should be before validation
        assert train_max_date <= val_min_date
        # Validation data should be before test
        assert val_max_date <= test_min_date

    def test_temporal_split_invalid_column(self, sample_data):
        """Test temporal split with invalid date column."""
        splitter = DataSplitter(test_size=0.2, val_size=0.1)

        with pytest.raises(ValueError):
            splitter.split_temporal(sample_data, date_column="nonexistent_column")

    def test_temporal_split_no_data_leakage(self, sample_data):
        """Test that there's no overlap between temporal splits."""
        splitter = DataSplitter(test_size=0.2, val_size=0.1)
        train, val, test = splitter.split_temporal(sample_data, date_column="admission_date")

        train_ids = set(train["subject_id"].to_list())
        val_ids = set(val["subject_id"].to_list())
        test_ids = set(test["subject_id"].to_list())

        # Check no overlap
        assert len(train_ids & val_ids) == 0
        assert len(train_ids & test_ids) == 0
        assert len(val_ids & test_ids) == 0


class TestStratifiedSplit:
    """Test stratified splitting functionality."""

    def test_stratified_split_basic(self, sample_data):
        """Test basic stratified split functionality."""
        splitter = DataSplitter(test_size=0.2, val_size=0.1)
        train, val, test = splitter.split_stratified(
            sample_data,
            stratify_column="label",
            random_state=42
        )

        # Check sizes
        assert len(train) == 70
        assert len(val) == 10
        assert len(test) == 20

    def test_stratified_split_maintains_proportions(self, sample_data):
        """Test that stratified split maintains class proportions."""
        splitter = DataSplitter(test_size=0.2, val_size=0.1)
        train, val, test = splitter.split_stratified(
            sample_data,
            stratify_column="label",
            random_state=42
        )

        # Get class proportions in original data
        original_pos = sample_data["label"].sum() / len(sample_data)

        # Get class proportions in splits
        train_pos = train["label"].sum() / len(train)
        val_pos = val["label"].sum() / len(val)
        test_pos = test["label"].sum() / len(test)

        # Check proportions are similar (within 5% tolerance)
        tolerance = 0.05
        assert abs(train_pos - original_pos) < tolerance
        assert abs(val_pos - original_pos) < tolerance
        assert abs(test_pos - original_pos) < tolerance

    def test_stratified_split_invalid_column(self, sample_data):
        """Test stratified split with invalid column."""
        splitter = DataSplitter(test_size=0.2, val_size=0.1)

        with pytest.raises(ValueError):
            splitter.split_stratified(
                sample_data,
                stratify_column="nonexistent_column",
                random_state=42
            )

    def test_stratified_split_reproducibility(self, sample_data):
        """Test that stratified split is reproducible."""
        splitter = DataSplitter(test_size=0.2, val_size=0.1)

        train1, val1, test1 = splitter.split_stratified(
            sample_data,
            stratify_column="label",
            random_state=42
        )
        train2, val2, test2 = splitter.split_stratified(
            sample_data,
            stratify_column="label",
            random_state=42
        )

        # Check that splits are identical
        assert set(train1["subject_id"].to_list()) == set(train2["subject_id"].to_list())
        assert set(val1["subject_id"].to_list()) == set(val2["subject_id"].to_list())
        assert set(test1["subject_id"].to_list()) == set(test2["subject_id"].to_list())


class TestSavingAndLoading:
    """Test saving and loading splits."""

    def test_save_splits(self, sample_data, temp_output_dir):
        """Test saving splits to parquet files."""
        splitter = DataSplitter(test_size=0.2, val_size=0.1)
        train, val, test = splitter.split_random(sample_data, random_state=42)

        # Save splits
        paths = splitter.save_splits(
            train, val, test,
            disease_name="test_disease",
            output_dir=temp_output_dir
        )

        # Check that files were created
        assert paths["train"].exists()
        assert paths["val"].exists()
        assert paths["test"].exists()
        assert paths["stats"].exists()

        # Check filenames
        assert paths["train"].name == "test_disease_train.parquet"
        assert paths["val"].name == "test_disease_val.parquet"
        assert paths["test"].name == "test_disease_test.parquet"
        assert paths["stats"].name == "test_disease_split_stats.json"

    def test_load_splits(self, sample_data, temp_output_dir):
        """Test loading previously saved splits."""
        splitter = DataSplitter(test_size=0.2, val_size=0.1)
        train_orig, val_orig, test_orig = splitter.split_random(sample_data, random_state=42)

        # Save splits
        splitter.save_splits(
            train_orig, val_orig, test_orig,
            disease_name="test_disease",
            output_dir=temp_output_dir
        )

        # Load splits
        train_loaded, val_loaded, test_loaded = splitter.load_splits(
            disease_name="test_disease",
            input_dir=temp_output_dir
        )

        # Check that loaded data matches original
        assert train_loaded.equals(train_orig)
        assert val_loaded.equals(val_orig)
        assert test_loaded.equals(test_orig)

    def test_load_nonexistent_splits(self, temp_output_dir):
        """Test loading splits that don't exist."""
        splitter = DataSplitter(test_size=0.2, val_size=0.1)

        with pytest.raises(FileNotFoundError):
            splitter.load_splits(
                disease_name="nonexistent_disease",
                input_dir=temp_output_dir
            )


class TestSplitStatistics:
    """Test split statistics functionality."""

    def test_get_split_stats(self, sample_data):
        """Test getting split statistics."""
        splitter = DataSplitter(test_size=0.2, val_size=0.1)
        train, val, test = splitter.split_random(sample_data, random_state=42)

        stats = splitter.get_split_stats()

        # Check that stats are present
        assert "strategy" in stats
        assert stats["strategy"] == "random"
        assert "sizes" in stats
        assert stats["sizes"]["train"] == 70
        assert stats["sizes"]["val"] == 10
        assert stats["sizes"]["test"] == 20
        assert "class_balance" in stats

    def test_stats_with_stratified_split(self, sample_data):
        """Test statistics for stratified split."""
        splitter = DataSplitter(test_size=0.2, val_size=0.1)
        train, val, test = splitter.split_stratified(
            sample_data,
            stratify_column="label",
            random_state=42
        )

        stats = splitter.get_split_stats()

        # Check strategy and stratify column
        assert stats["strategy"] == "stratified"
        assert stats["stratify_column"] == "label"

    def test_stats_with_temporal_split(self, sample_data):
        """Test statistics for temporal split."""
        splitter = DataSplitter(test_size=0.2, val_size=0.1)
        train, val, test = splitter.split_temporal(
            sample_data,
            date_column="admission_date"
        )

        stats = splitter.get_split_stats()

        # Check strategy and temporal info
        assert stats["strategy"] == "temporal"
        assert "temporal_info" in stats
        assert stats["temporal_info"]["date_column"] == "admission_date"
        assert "train_period" in stats["temporal_info"]
        assert "val_period" in stats["temporal_info"]
        assert "test_period" in stats["temporal_info"]

    def test_stats_saved_to_json(self, sample_data, temp_output_dir):
        """Test that statistics are saved to JSON file."""
        splitter = DataSplitter(test_size=0.2, val_size=0.1)
        train, val, test = splitter.split_random(sample_data, random_state=42)

        paths = splitter.save_splits(
            train, val, test,
            disease_name="test_disease",
            output_dir=temp_output_dir
        )

        # Load and check JSON stats
        with open(paths["stats"], "r") as f:
            saved_stats = json.load(f)

        assert saved_stats["strategy"] == "random"
        assert saved_stats["sizes"]["train"] == 70
        assert "class_balance" in saved_stats


class TestPrintSummary:
    """Test print summary functionality."""

    def test_print_summary_no_split(self, capsys):
        """Test printing summary before any split."""
        splitter = DataSplitter(test_size=0.2, val_size=0.1)
        splitter.print_split_summary()

        captured = capsys.readouterr()
        assert "No split statistics available" in captured.out

    def test_print_summary_after_split(self, sample_data, capsys):
        """Test printing summary after a split."""
        splitter = DataSplitter(test_size=0.2, val_size=0.1)
        train, val, test = splitter.split_random(sample_data, random_state=42)

        splitter.print_split_summary()

        captured = capsys.readouterr()
        assert "Split Summary" in captured.out
        assert "RANDOM" in captured.out
        assert "Train:" in captured.out
        assert "Val:" in captured.out
        assert "Test:" in captured.out
