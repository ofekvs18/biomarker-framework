"""Data splitting utilities for train/validation/test splits."""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import polars as pl


class DataSplitter:
    """Split datasets into train/validation/test sets with various strategies.

    This class provides multiple splitting strategies for creating reproducible
    train/validation/test splits:
    - Random split with fixed seed for reproducibility
    - Temporal split for time-series data (train on earlier, test on later)
    - Stratified split to maintain class balance across splits

    All splits can be saved to disk and statistics are automatically documented.
    """

    def __init__(self, test_size: float = 0.2, val_size: float = 0.1):
        """Initialize DataSplitter with split proportions.

        Args:
            test_size: Proportion of data for test set (default 0.2 = 20%).
            val_size: Proportion of data for validation set (default 0.1 = 10%).
                     The training set will receive the remaining proportion.

        Raises:
            ValueError: If test_size or val_size are not in (0, 1) or if
                       test_size + val_size >= 1.
        """
        if not (0 < test_size < 1):
            raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
        if not (0 < val_size < 1):
            raise ValueError(f"val_size must be between 0 and 1, got {val_size}")
        if test_size + val_size >= 1:
            raise ValueError(
                f"test_size ({test_size}) + val_size ({val_size}) must be < 1. "
                f"Got {test_size + val_size}"
            )

        self.test_size = test_size
        self.val_size = val_size
        self.train_size = 1.0 - test_size - val_size

        # Storage for split statistics
        self._split_stats: Dict = {}

    def split_random(
        self,
        df: pl.DataFrame,
        random_state: int = 42
    ) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Split data randomly with fixed seed for reproducibility.

        Creates random train/validation/test splits while maintaining reproducibility
        through a fixed random seed. This is the simplest splitting strategy and
        works well when data is already shuffled or when temporal/group structure
        doesn't matter.

        Args:
            df: Input DataFrame to split.
            random_state: Random seed for reproducibility (default 42).

        Returns:
            Tuple of (train_df, val_df, test_df) DataFrames.
        """
        # Shuffle the dataframe with fixed seed
        shuffled = df.sample(fraction=1.0, shuffle=True, seed=random_state)

        # Calculate split indices
        n_samples = len(shuffled)
        test_idx = int(n_samples * (1 - self.test_size))
        val_idx = int(n_samples * (1 - self.test_size - self.val_size))

        # Split the data
        train_df = shuffled[:val_idx]
        val_df = shuffled[val_idx:test_idx]
        test_df = shuffled[test_idx:]

        # Compute and store statistics
        self._compute_split_stats(train_df, val_df, test_df, "random")

        return train_df, val_df, test_df

    def split_temporal(
        self,
        df: pl.DataFrame,
        date_column: str
    ) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Split data temporally (chronologically) based on a date column.

        Creates train/validation/test splits where:
        - Training set contains the earliest data
        - Validation set contains middle period data
        - Test set contains the most recent data

        This is crucial for time-series data to prevent data leakage and
        simulate real-world deployment where models predict future events.

        Args:
            df: Input DataFrame to split.
            date_column: Name of the column containing dates/timestamps.

        Returns:
            Tuple of (train_df, val_df, test_df) DataFrames.

        Raises:
            ValueError: If date_column is not in the DataFrame.
        """
        if date_column not in df.columns:
            raise ValueError(
                f"Column '{date_column}' not found in DataFrame. "
                f"Available columns: {df.columns}"
            )

        # Ensure date column is datetime type
        if df.schema[date_column] not in [pl.Datetime, pl.Date]:
            # Try to convert to datetime
            try:
                df = df.with_columns(
                    pl.col(date_column).str.to_datetime().alias(date_column)
                )
            except Exception:
                raise ValueError(
                    f"Column '{date_column}' must be datetime type or "
                    f"convertible to datetime. Got type: {df.schema[date_column]}"
                )

        # Sort by date column
        sorted_df = df.sort(date_column)

        # Calculate split indices
        n_samples = len(sorted_df)
        val_idx = int(n_samples * self.train_size)
        test_idx = int(n_samples * (self.train_size + self.val_size))

        # Split the data chronologically
        train_df = sorted_df[:val_idx]
        val_df = sorted_df[val_idx:test_idx]
        test_df = sorted_df[test_idx:]

        # Compute and store statistics
        self._compute_split_stats(
            train_df, val_df, test_df,
            "temporal",
            date_column=date_column
        )

        return train_df, val_df, test_df

    def split_stratified(
        self,
        df: pl.DataFrame,
        stratify_column: str,
        random_state: int = 42
    ) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Split data while maintaining class distribution (stratified sampling).

        Creates train/validation/test splits that preserve the proportion of
        samples for each class in the stratify_column. This is important for:
        - Imbalanced datasets (e.g., rare diseases)
        - Ensuring each split has representative samples of all classes
        - Maintaining statistical properties across splits

        Args:
            df: Input DataFrame to split.
            stratify_column: Name of the column to stratify by (e.g., 'label').
            random_state: Random seed for reproducibility (default 42).

        Returns:
            Tuple of (train_df, val_df, test_df) DataFrames.

        Raises:
            ValueError: If stratify_column is not in the DataFrame.
        """
        if stratify_column not in df.columns:
            raise ValueError(
                f"Column '{stratify_column}' not found in DataFrame. "
                f"Available columns: {df.columns}"
            )

        # Get unique classes
        classes = df.select(stratify_column).unique().to_series().to_list()

        train_dfs = []
        val_dfs = []
        test_dfs = []

        # Split each class separately to maintain proportions
        for class_value in classes:
            # Filter data for this class
            class_df = df.filter(pl.col(stratify_column) == class_value)

            # Shuffle with fixed seed
            shuffled = class_df.sample(fraction=1.0, shuffle=True, seed=random_state)

            # Calculate split indices for this class
            n_samples = len(shuffled)
            val_idx = int(n_samples * self.train_size)
            test_idx = int(n_samples * (self.train_size + self.val_size))

            # Split this class
            train_dfs.append(shuffled[:val_idx])
            val_dfs.append(shuffled[val_idx:test_idx])
            test_dfs.append(shuffled[test_idx:])

        # Concatenate all classes
        train_df = pl.concat(train_dfs)
        val_df = pl.concat(val_dfs)
        test_df = pl.concat(test_dfs)

        # Shuffle each split to mix classes
        train_df = train_df.sample(fraction=1.0, shuffle=True, seed=random_state)
        val_df = val_df.sample(fraction=1.0, shuffle=True, seed=random_state + 1)
        test_df = test_df.sample(fraction=1.0, shuffle=True, seed=random_state + 2)

        # Compute and store statistics
        self._compute_split_stats(
            train_df, val_df, test_df,
            "stratified",
            stratify_column=stratify_column
        )

        return train_df, val_df, test_df

    def _compute_split_stats(
        self,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
        test_df: pl.DataFrame,
        strategy: str,
        **kwargs
    ) -> None:
        """Compute and store statistics about the splits.

        Args:
            train_df: Training set DataFrame.
            val_df: Validation set DataFrame.
            test_df: Test set DataFrame.
            strategy: Splitting strategy used.
            **kwargs: Additional metadata (e.g., date_column, stratify_column).
        """
        stats = {
            "strategy": strategy,
            "sizes": {
                "train": len(train_df),
                "val": len(val_df),
                "test": len(test_df),
                "total": len(train_df) + len(val_df) + len(test_df)
            },
            "proportions": {
                "train": self.train_size,
                "val": self.val_size,
                "test": self.test_size
            }
        }

        # Add class balance statistics if label column exists
        if "label" in train_df.columns:
            stats["class_balance"] = {
                "train": self._get_class_distribution(train_df, "label"),
                "val": self._get_class_distribution(val_df, "label"),
                "test": self._get_class_distribution(test_df, "label")
            }

        # Add temporal information if date column was used
        if "date_column" in kwargs:
            date_col = kwargs["date_column"]
            stats["temporal_info"] = {
                "date_column": date_col,
                "train_period": {
                    "start": str(train_df[date_col].min()),
                    "end": str(train_df[date_col].max())
                },
                "val_period": {
                    "start": str(val_df[date_col].min()),
                    "end": str(val_df[date_col].max())
                },
                "test_period": {
                    "start": str(test_df[date_col].min()),
                    "end": str(test_df[date_col].max())
                }
            }

        # Add stratification column info
        if "stratify_column" in kwargs:
            stats["stratify_column"] = kwargs["stratify_column"]

        self._split_stats = stats

    def _get_class_distribution(
        self,
        df: pl.DataFrame,
        column: str
    ) -> Dict[str, Union[int, float]]:
        """Get class distribution statistics for a column.

        Args:
            df: DataFrame to analyze.
            column: Column name to compute distribution for.

        Returns:
            Dictionary with counts and percentages for each class.
        """
        total = len(df)
        if total == 0:
            return {}

        # Count values for each class
        counts = df.group_by(column).len().sort(column)

        distribution = {}
        for row in counts.iter_rows(named=True):
            class_value = row[column]
            count = row["len"]
            percentage = (count / total) * 100
            distribution[str(class_value)] = {
                "count": count,
                "percentage": round(percentage, 2)
            }

        return distribution

    def save_splits(
        self,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
        test_df: pl.DataFrame,
        disease_name: str,
        output_dir: Union[str, Path] = "data/splits"
    ) -> Dict[str, Path]:
        """Save train/validation/test splits to parquet files.

        Saves the splits to disk with standardized naming convention:
        - {disease}_train.parquet
        - {disease}_val.parquet
        - {disease}_test.parquet
        - {disease}_split_stats.json (statistics about the split)

        Args:
            train_df: Training set DataFrame.
            val_df: Validation set DataFrame.
            test_df: Test set DataFrame.
            disease_name: Name of the disease (used in filenames).
            output_dir: Directory to save splits (default 'data/splits').

        Returns:
            Dictionary mapping split names to their file paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Define file paths
        paths = {
            "train": output_dir / f"{disease_name}_train.parquet",
            "val": output_dir / f"{disease_name}_val.parquet",
            "test": output_dir / f"{disease_name}_test.parquet",
            "stats": output_dir / f"{disease_name}_split_stats.json"
        }

        # Save DataFrames to parquet
        train_df.write_parquet(paths["train"])
        val_df.write_parquet(paths["val"])
        test_df.write_parquet(paths["test"])

        # Save statistics to JSON
        if self._split_stats:
            with open(paths["stats"], "w") as f:
                json.dump(self._split_stats, f, indent=2)

        return paths

    def load_splits(
        self,
        disease_name: str,
        input_dir: Union[str, Path] = "data/splits"
    ) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Load previously saved train/validation/test splits.

        Args:
            disease_name: Name of the disease (used in filenames).
            input_dir: Directory containing the splits (default 'data/splits').

        Returns:
            Tuple of (train_df, val_df, test_df) DataFrames.

        Raises:
            FileNotFoundError: If any of the split files are not found.
        """
        input_dir = Path(input_dir)

        # Define file paths
        train_path = input_dir / f"{disease_name}_train.parquet"
        val_path = input_dir / f"{disease_name}_val.parquet"
        test_path = input_dir / f"{disease_name}_test.parquet"

        # Check if files exist
        for path in [train_path, val_path, test_path]:
            if not path.exists():
                raise FileNotFoundError(
                    f"Split file not found: {path}. "
                    f"Please run save_splits() first or check the disease_name."
                )

        # Load DataFrames
        train_df = pl.read_parquet(train_path)
        val_df = pl.read_parquet(val_path)
        test_df = pl.read_parquet(test_path)

        # Load statistics if available
        stats_path = input_dir / f"{disease_name}_split_stats.json"
        if stats_path.exists():
            with open(stats_path, "r") as f:
                self._split_stats = json.load(f)

        return train_df, val_df, test_df

    def get_split_stats(self) -> Dict:
        """Get statistics about the most recent split operation.

        Returns:
            Dictionary containing split statistics including sizes, proportions,
            and class balance information.
        """
        return self._split_stats.copy() if self._split_stats else {}

    def print_split_summary(self) -> None:
        """Print a formatted summary of the split statistics."""
        if not self._split_stats:
            print("No split statistics available. Perform a split first.")
            return

        stats = self._split_stats

        print("\n" + "="*60)
        print(f"Split Summary - Strategy: {stats['strategy'].upper()}")
        print("="*60)

        # Print sizes
        print("\nDataset Sizes:")
        sizes = stats["sizes"]
        print(f"  Train: {sizes['train']:>6} samples ({self.train_size*100:.1f}%)")
        print(f"  Val:   {sizes['val']:>6} samples ({self.val_size*100:.1f}%)")
        print(f"  Test:  {sizes['test']:>6} samples ({self.test_size*100:.1f}%)")
        print(f"  Total: {sizes['total']:>6} samples")

        # Print class balance if available
        if "class_balance" in stats:
            print("\nClass Balance:")
            for split_name in ["train", "val", "test"]:
                balance = stats["class_balance"][split_name]
                print(f"\n  {split_name.capitalize()}:")
                for class_name, class_stats in balance.items():
                    count = class_stats["count"]
                    pct = class_stats["percentage"]
                    print(f"    Class {class_name}: {count:>5} samples ({pct:>5.1f}%)")

        # Print temporal info if available
        if "temporal_info" in stats:
            print("\nTemporal Information:")
            temp_info = stats["temporal_info"]
            print(f"  Date column: {temp_info['date_column']}")
            print(f"  Train period: {temp_info['train_period']['start']} to "
                  f"{temp_info['train_period']['end']}")
            print(f"  Val period:   {temp_info['val_period']['start']} to "
                  f"{temp_info['val_period']['end']}")
            print(f"  Test period:  {temp_info['test_period']['start']} to "
                  f"{temp_info['test_period']['end']}")

        # Print stratification info if available
        if "stratify_column" in stats:
            print(f"\nStratified by column: {stats['stratify_column']}")

        print("\n" + "="*60 + "\n")
