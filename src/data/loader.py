"""Data loading utilities for MIMIC-IV datasets."""

from pathlib import Path
from typing import Optional, Union

import pandas as pd


class DataLoader:
    """Load and manage MIMIC-IV data files."""

    def __init__(self, data_dir: Union[str, Path] = "data/raw/mimic_data"):
        """Initialize DataLoader with data directory path.

        Args:
            data_dir: Path to the raw MIMIC data directory.
        """
        self.data_dir = Path(data_dir)

    def load_labevents(self) -> pd.DataFrame:
        """Load laboratory events data."""
        raise NotImplementedError

    def load_patients(self) -> pd.DataFrame:
        """Load patient demographics data."""
        raise NotImplementedError

    def load_admissions(self) -> pd.DataFrame:
        """Load hospital admissions data."""
        raise NotImplementedError

    def load_diagnoses(self) -> pd.DataFrame:
        """Load diagnosis codes data."""
        raise NotImplementedError
