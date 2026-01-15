#!/usr/bin/env python3
"""
Data Setup Script
=================

Checks if MIMIC-IV data exists in biomarker-framework/data/raw/ and downloads
it from BigQuery if not.

Usage:
    python scripts/setup_data.py

    # Force re-download even if data exists
    python scripts/setup_data.py --force

    # Check data status only (no download)
    python scripts/setup_data.py --check-only

    # Download specific tables only
    python scripts/setup_data.py --table labevents
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent

# Data directory inside biomarker-framework
DATA_DIR = PROJECT_DIR / "data" / "raw"

# Required parquet files
REQUIRED_FILES = [
    "patients.parquet",
    "admissions.parquet",
    "icustays.parquet",
    "diagnoses_icd.parquet",
    "d_icd_diagnoses.parquet",
    "labevents.parquet",
    "d_labitems.parquet",
]


def check_data_exists(data_dir: Path) -> dict:
    """
    Check which data files exist.

    Returns:
        dict with file status: {filename: exists_bool}
    """
    status = {}
    for filename in REQUIRED_FILES:
        filepath = data_dir / filename
        status[filename] = filepath.exists()
    return status


def print_status(status: dict, data_dir: Path):
    """Print data status in a nice format."""
    print(f"\nData directory: {data_dir}")
    print(f"Directory exists: {data_dir.exists()}\n")

    if not data_dir.exists():
        print("Data directory not found! Will be created on download.")
        return False

    print("File status:")
    all_exist = True
    for filename, exists in status.items():
        icon = "[OK]" if exists else "[MISSING]"
        print(f"  {icon} {filename}")
        if not exists:
            all_exist = False

    if all_exist:
        print("\nAll required data files are present!")
    else:
        print("\nSome data files are missing.")

    return all_exist


def get_missing_files(status: dict) -> list:
    """Get list of missing files."""
    return [f for f, exists in status.items() if not exists]


# Map filenames to download methods
FILE_TO_TABLE = {
    "patients.parquet": "patients",
    "admissions.parquet": "admissions",
    "icustays.parquet": "icustays",
    "diagnoses_icd.parquet": "diagnoses_icd",
    "d_icd_diagnoses.parquet": "d_icd_diagnoses",
    "labevents.parquet": "labevents",
    "d_labitems.parquet": "d_labitems",
}


def download_data(output_dir: Path, table: str = "all", **kwargs):
    """
    Download MIMIC-IV data from BigQuery.

    Args:
        output_dir: Directory to save parquet files
        table: Specific table to download, or "all"
        **kwargs: Additional arguments passed to downloader
    """
    # Import the download module
    sys.path.insert(0, str(SCRIPT_DIR))

    try:
        from download_mimic_from_bigquery import MIMICDataDownloader
    except ImportError:
        print("Error: download_mimic_from_bigquery.py not found in scripts/")
        print("Make sure the download script is in the scripts directory.")
        return False

    try:
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nDownloading MIMIC-IV data to: {output_dir}")
        print("This may take 30-60 minutes depending on your connection...\n")

        downloader = MIMICDataDownloader(str(output_dir))

        if table == "all":
            downloader.download_all(**kwargs)
        elif table == "patients":
            downloader.download_patients()
        elif table == "admissions":
            downloader.download_admissions(**kwargs)
        elif table == "icustays":
            downloader.download_icustays()
        elif table == "diagnoses" or table == "diagnoses_icd":
            downloader.download_diagnoses_icd(**kwargs)
        elif table == "d_icd_diagnoses":
            downloader.download_d_icd_diagnoses()
        elif table == "labevents":
            downloader.download_labevents(**kwargs)
        elif table == "d_labitems":
            downloader.download_d_labitems()
        else:
            print(f"Unknown table: {table}")
            return False

        print("\nDownload complete!")
        return True

    except Exception as e:
        print(f"\nDownload failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Run: gcloud auth application-default login")
        print("  2. Ensure you have BigQuery access to physionet-data.mimiciv_3_1_hosp")
        print("  3. Check your internet connection")
        return False


def download_missing_only(output_dir: Path, missing_files: list, **kwargs):
    """
    Download only the missing files.

    Args:
        output_dir: Directory to save parquet files
        missing_files: List of missing filenames
        **kwargs: Additional arguments passed to downloader
    """
    # Import the download module
    sys.path.insert(0, str(SCRIPT_DIR))

    try:
        from download_mimic_from_bigquery import MIMICDataDownloader
    except ImportError:
        print("Error: download_mimic_from_bigquery.py not found in scripts/")
        print("Make sure the download script is in the scripts directory.")
        return False

    try:
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nDownloading missing data to: {output_dir}")
        print(f"Files to download: {', '.join(missing_files)}\n")

        downloader = MIMICDataDownloader(str(output_dir))

        for filename in missing_files:
            table = FILE_TO_TABLE.get(filename)
            if not table:
                print(f"Warning: Unknown file {filename}, skipping...")
                continue

            print(f"\n--- Downloading {filename} ---")
            if table == "patients":
                downloader.download_patients()
            elif table == "admissions":
                downloader.download_admissions(**kwargs)
            elif table == "icustays":
                downloader.download_icustays()
            elif table == "diagnoses_icd":
                downloader.download_diagnoses_icd(**kwargs)
            elif table == "d_icd_diagnoses":
                downloader.download_d_icd_diagnoses()
            elif table == "labevents":
                downloader.download_labevents(**kwargs)
            elif table == "d_labitems":
                downloader.download_d_labitems()

        print("\nDownload complete!")
        return True

    except Exception as e:
        print(f"\nDownload failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Run: gcloud auth application-default login")
        print("  2. Ensure you have BigQuery access to physionet-data.mimiciv_3_1_hosp")
        print("  3. Check your internet connection")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Check and setup MIMIC-IV data for the biomarker framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check data status
  python scripts/setup_data.py --check-only

  # Download only missing files (default behavior)
  python scripts/setup_data.py

  # Download all tables (even if some exist)
  python scripts/setup_data.py --all

  # Download specific table only
  python scripts/setup_data.py --table labevents

  # Force re-download everything
  python scripts/setup_data.py --force

Note: Requires BigQuery access to physionet-data.mimiciv_3_1_hosp
      Run 'gcloud auth application-default login' first
        """
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download all files even if they exist"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="download_all",
        help="Download all tables (not just missing ones)"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check data status, don't download"
    )
    parser.add_argument(
        "--table",
        type=str,
        choices=["patients", "admissions", "icustays", "diagnoses",
                 "labevents", "d_labitems", "d_icd_diagnoses"],
        default=None,
        help="Download specific table only"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Filter data after this date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Filter data before this date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Auto-confirm download without prompting"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("MIMIC-IV Data Setup")
    print("=" * 60)

    # Check current status
    status = check_data_exists(DATA_DIR)
    all_exist = print_status(status, DATA_DIR)

    # If check-only mode, exit here
    if args.check_only:
        sys.exit(0 if all_exist else 1)

    # Get missing files
    missing = get_missing_files(status)

    # Determine what to download
    if args.force:
        # Force: re-download everything
        download_mode = "all"
        print("\n--force flag set, will re-download all data...")
    elif args.table:
        # Specific table requested
        download_mode = "table"
        print(f"\nWill download specific table: {args.table}")
    elif args.download_all:
        # --all flag: download all tables
        download_mode = "all"
        print("\n--all flag set, will download all tables...")
    elif missing:
        # Default: only download missing files
        download_mode = "missing"
        print(f"\nWill download only missing files: {', '.join(missing)}")
    else:
        # Nothing to do
        print("\nAll data files already present. Use --force to re-download.")
        print("\n" + "=" * 60)
        print("Setup complete!")
        print("=" * 60)
        return

    # Confirm download
    if args.yes:
        do_download = True
    else:
        response = input("\nDo you want to download the data from BigQuery? (yes/no): ").strip().lower()
        do_download = response in ['yes', 'y']

    if do_download:
        # Build kwargs for downloader
        kwargs = {}
        if args.start_date:
            kwargs['start_date'] = args.start_date
        if args.end_date:
            kwargs['end_date'] = args.end_date

        if download_mode == "missing":
            success = download_missing_only(DATA_DIR, missing, **kwargs)
        elif download_mode == "table":
            success = download_data(DATA_DIR, table=args.table, **kwargs)
        else:  # "all"
            success = download_data(DATA_DIR, table="all", **kwargs)

        if not success:
            sys.exit(1)

        # Re-check status after download
        print("\n" + "-" * 60)
        status = check_data_exists(DATA_DIR)
        print_status(status, DATA_DIR)
    else:
        print("\nDownload skipped.")
        if missing:
            print("Warning: Some data files are missing. The framework may not work correctly.")

    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)

    # Final instructions
    if all(check_data_exists(DATA_DIR).values()):
        print("\nNext steps:")
        print("  1. cd biomarker-framework")
        print("  2. pip install -e .")
        print("  3. jupyter notebook notebooks/01_eda.ipynb")


if __name__ == "__main__":
    main()
