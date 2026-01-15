#!/usr/bin/env python3
"""
Download MIMIC-IV Data from BigQuery to Parquet Files
======================================================

This script downloads essential MIMIC-IV tables from BigQuery (physionet-data)
and saves them as Parquet files for local analysis.

Required tables for biomarker discovery pipeline:
- patients: Patient demographics
- admissions: Hospital admissions (for temporal anchoring)
- icustays: ICU stays
- diagnoses_icd: ICD-9 and ICD-10 diagnoses
- d_icd_diagnoses: Diagnosis code descriptions
- labevents: Laboratory test results (blood tests)
- d_labitems: Lab test definitions and descriptions

Prerequisites:
1. Google Cloud authentication: gcloud auth application-default login
2. Access to physionet-data.mimiciv_3_1_hosp on BigQuery
3. Required packages: google-cloud-bigquery, pandas, pyarrow

Usage:
    python download_mimic_from_bigquery.py --output-dir ./mimic_data

    # With filtering:
    python download_mimic_from_bigquery.py --output-dir ./mimic_data \
        --start-date 2015-01-01 --end-date 2019-12-31 \
        --filter-labs "Hemoglobin,Glucose,Creatinine"
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from typing import Optional, List
import json

try:
    from google.cloud import bigquery
    import pandas as pd
    import pyarrow.parquet as pq
    from google.auth.exceptions import DefaultCredentialsError
    import google.auth
except ImportError as e:
    print(f"Missing required package: {e}")
    print("\nInstall with: pip install google-cloud-bigquery pandas pyarrow")
    sys.exit(1)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def authenticate_bigquery():
    """
    Guide user through BigQuery authentication process

    Returns:
        True if user wants to proceed with authentication, False otherwise
    """
    print("\n" + "="*80)
    print("GOOGLE CLOUD AUTHENTICATION REQUIRED")
    print("="*80)
    print("\nYou need to authenticate with Google Cloud to access BigQuery.")
    print("\nPlease follow these steps:")
    print("\n1. The authentication page will open in your default browser")
    print("2. If it doesn't open automatically, copy and paste the URL below into Chrome")
    print("3. Sign in with your Google account that has access to BigQuery")
    print("4. Grant the requested permissions")
    print("5. Return here after completing authentication")
    print("\nAuthentication command:")
    print("  gcloud auth application-default login")
    print("\nNote: If you don't have gcloud installed, visit:")
    print("  https://cloud.google.com/sdk/docs/install")
    print("\n" + "="*80)

    response = input("\nDo you want to run the authentication command now? (yes/no): ").strip().lower()

    if response in ['yes', 'y']:
        print("\nLaunching authentication...")
        print("\nIMPORTANT: Please open this authentication link in Chrome:")
        print("-" * 80)

        try:
            import subprocess
            # Run gcloud auth command
            result = subprocess.run(
                ['gcloud', 'auth', 'application-default', 'login'],
                capture_output=False,
                text=True
            )

            if result.returncode == 0:
                print("\n" + "="*80)
                print("Authentication successful!")
                print("="*80)
                return True
            else:
                print("\nAuthentication failed. Please try again.")
                return False

        except FileNotFoundError:
            print("\nERROR: 'gcloud' command not found.")
            print("\nPlease install Google Cloud SDK from:")
            print("  https://cloud.google.com/sdk/docs/install")
            print("\nAfter installation, run:")
            print("  gcloud auth application-default login")
            return False
        except Exception as e:
            print(f"\nError during authentication: {e}")
            return False
    else:
        print("\nAuthentication cancelled.")
        print("\nTo authenticate later, run:")
        print("  gcloud auth application-default login")
        return False


class MIMICDataDownloader:
    """Download MIMIC-IV data from BigQuery to Parquet files"""

    def __init__(self, output_dir: str, project_id: Optional[str] = None):
        """
        Initialize downloader

        Args:
            output_dir: Directory to save Parquet files
            project_id: GCP project ID (uses default if None)
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Initialize BigQuery client
        try:
            self.client = bigquery.Client(project=project_id)
            logger.info(f"Connected to BigQuery (project: {self.client.project})")
        except DefaultCredentialsError as e:
            logger.error(f"Failed to initialize BigQuery client: {e}")

            # Prompt user to authenticate
            if authenticate_bigquery():
                # Try again after authentication
                try:
                    self.client = bigquery.Client(project=project_id)
                    logger.info(f"Connected to BigQuery (project: {self.client.project})")
                except Exception as retry_error:
                    logger.error(f"Still failed to connect after authentication: {retry_error}")
                    raise
            else:
                raise
        except Exception as e:
            logger.error(f"Failed to initialize BigQuery client: {e}")
            logger.error("Run: gcloud auth application-default login")
            raise

        # Track download metadata
        self.metadata = {
            'download_timestamp': datetime.now().isoformat(),
            'project_id': self.client.project,
            'tables': {}
        }

    def _execute_query(self, query: str, table_name: str,
                       description: str = "") -> pd.DataFrame:
        """
        Execute BigQuery query and return DataFrame

        Args:
            query: SQL query string
            table_name: Name for logging/metadata
            description: Human-readable description

        Returns:
            DataFrame with query results
        """
        logger.info(f"Downloading {table_name}... {description}")

        try:
            # Execute query
            query_job = self.client.query(query)

            # Get results
            df = query_job.result().to_dataframe()

            # Store metadata
            self.metadata['tables'][table_name] = {
                'rows': len(df),
                'columns': list(df.columns),
                'bytes_processed': query_job.total_bytes_processed,
                'description': description
            }

            logger.info(f"  ✓ Downloaded {len(df):,} rows, "
                       f"{len(df.columns)} columns, "
                       f"{query_job.total_bytes_processed / 1e9:.2f} GB processed")

            return df

        except Exception as e:
            logger.error(f"Failed to download {table_name}: {e}")
            raise

    def _save_parquet(self, df: pd.DataFrame, filename: str):
        """Save DataFrame to Parquet file"""
        filepath = os.path.join(self.output_dir, filename)
        df.to_parquet(filepath, index=False, engine='pyarrow', compression='snappy')

        file_size_mb = os.path.getsize(filepath) / (1024 ** 2)
        logger.info(f"  ✓ Saved to {filename} ({file_size_mb:.1f} MB)")

    def download_patients(self):
        """Download patient demographics"""
        query = """
        SELECT
            subject_id,
            gender,
            anchor_age,
            anchor_year,
            anchor_year_group,
            dod  -- Date of death (if applicable)
        FROM `physionet-data.mimiciv_3_1_hosp.patients`
        """

        df = self._execute_query(
            query,
            'patients',
            'Patient demographics and death dates'
        )
        self._save_parquet(df, 'patients.parquet')
        return df

    def download_admissions(self, start_date: Optional[str] = None,
                           end_date: Optional[str] = None):
        """
        Download hospital admissions

        Args:
            start_date: Filter admissions after this date (YYYY-MM-DD)
            end_date: Filter admissions before this date (YYYY-MM-DD)
        """
        date_filter = ""
        if start_date:
            date_filter += f"\n    AND admittime >= '{start_date}'"
        if end_date:
            date_filter += f"\n    AND admittime <= '{end_date}'"

        query = f"""
        SELECT
            subject_id,
            hadm_id,
            admittime,
            dischtime,
            deathtime,
            admission_type,
            admission_location,
            discharge_location,
            insurance,
            language,
            marital_status,
            race,
            hospital_expire_flag
        FROM `physionet-data.mimiciv_3_1_hosp.admissions`
        WHERE 1=1
        {date_filter}
        """

        desc = "Hospital admissions"
        if start_date or end_date:
            desc += f" ({start_date or 'start'} to {end_date or 'end'})"

        df = self._execute_query(query, 'admissions', desc)
        self._save_parquet(df, 'admissions.parquet')
        return df

    def download_icustays(self):
        """Download ICU stays"""
        query = """
        SELECT
            subject_id,
            hadm_id,
            stay_id,
            first_careunit,
            last_careunit,
            intime,
            outtime,
            los  -- Length of stay in days
        FROM `physionet-data.mimiciv_3_1_icu.icustays`
        """

        df = self._execute_query(
            query,
            'icustays',
            'ICU stays with care units and duration'
        )
        self._save_parquet(df, 'icustays.parquet')
        return df

    def download_diagnoses_icd(self, icd_version: Optional[int] = None):
        """
        Download ICD diagnoses

        Args:
            icd_version: Filter by ICD version (9 or 10). None = both.
        """
        version_filter = ""
        if icd_version in [9, 10]:
            version_filter = f"\n    AND icd_version = {icd_version}"

        query = f"""
        SELECT
            subject_id,
            hadm_id,
            seq_num,  -- Diagnosis sequence number
            icd_code,
            icd_version
        FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd`
        WHERE 1=1
        {version_filter}
        ORDER BY subject_id, hadm_id, seq_num
        """

        desc = "ICD diagnoses"
        if icd_version:
            desc += f" (ICD-{icd_version} only)"
        else:
            desc += " (ICD-9 and ICD-10)"

        df = self._execute_query(query, 'diagnoses_icd', desc)
        self._save_parquet(df, 'diagnoses_icd.parquet')
        return df

    def download_d_icd_diagnoses(self):
        """Download ICD diagnosis code descriptions"""
        query = """
        SELECT
            icd_code,
            icd_version,
            long_title  -- Full description of the diagnosis
        FROM `physionet-data.mimiciv_3_1_hosp.d_icd_diagnoses`
        ORDER BY icd_version, icd_code
        """

        df = self._execute_query(
            query,
            'd_icd_diagnoses',
            'ICD code descriptions (lookup table)'
        )
        self._save_parquet(df, 'd_icd_diagnoses.parquet')
        return df

    def download_d_labitems(self):
        """Download lab item definitions (small reference table)"""
        query = """
        SELECT
            itemid,
            label,
            fluid,  -- e.g., 'Blood', 'Urine'
            category  -- e.g., 'Hematology', 'Chemistry'
        FROM `physionet-data.mimiciv_3_1_hosp.d_labitems`
        ORDER BY itemid
        """

        df = self._execute_query(
            query,
            'd_labitems',
            'Lab test definitions (lookup table)'
        )
        self._save_parquet(df, 'd_labitems.parquet')
        return df

    def download_labevents(self,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None,
                          lab_names: Optional[List[str]] = None,
                          chunk_size: int = 5000000):
        """
        Download lab events (blood tests) - LARGE TABLE

        Args:
            start_date: Filter labs after this date (YYYY-MM-DD)
            end_date: Filter labs before this date (YYYY-MM-DD)
            lab_names: Filter by specific lab test names (e.g., ['Hemoglobin', 'Glucose'])
            chunk_size: Download in chunks to avoid memory issues
        """
        # Build filters
        filters = []

        if start_date:
            filters.append(f"charttime >= '{start_date}'")
        if end_date:
            filters.append(f"charttime <= '{end_date}'")

        # Lab name filter requires subquery
        lab_filter = ""
        if lab_names:
            lab_names_str = "', '".join(lab_names)
            lab_filter = f"""
            AND itemid IN (
                SELECT itemid
                FROM `physionet-data.mimiciv_3_1_hosp.d_labitems`
                WHERE label IN ('{lab_names_str}')
            )
            """

        where_clause = " AND ".join(filters) if filters else "1=1"

        query = f"""
        SELECT
            labevent_id,
            subject_id,
            hadm_id,
            specimen_id,
            itemid,
            charttime,
            storetime,
            value,
            valuenum,
            valueuom,  -- Unit of measurement
            ref_range_lower,
            ref_range_upper,
            flag,  -- 'abnormal' flag
            priority,
            comments
        FROM `physionet-data.mimiciv_3_1_hosp.labevents`
        WHERE {where_clause}
        {lab_filter}
        AND valuenum IS NOT NULL  -- Only numeric values
        ORDER BY subject_id, charttime
        """

        desc = "Lab events (blood tests)"
        if lab_names:
            desc += f" - {len(lab_names)} specific tests"
        if start_date or end_date:
            desc += f" ({start_date or 'start'} to {end_date or 'end'})"

        logger.info(f"Downloading {desc}...")
        logger.warning("This is a LARGE table and may take 10-30 minutes!")

        try:
            # Execute query
            query_job = self.client.query(query)

            # Download in chunks to avoid memory issues
            df_chunks = []
            total_rows = 0

            for chunk in query_job.result(page_size=chunk_size):
                chunk_df = chunk.to_dataframe()
                df_chunks.append(chunk_df)
                total_rows += len(chunk_df)
                logger.info(f"  Downloaded {total_rows:,} rows so far...")

            # Combine chunks
            df = pd.concat(df_chunks, ignore_index=True)

            # Store metadata
            self.metadata['tables']['labevents'] = {
                'rows': len(df),
                'columns': list(df.columns),
                'bytes_processed': query_job.total_bytes_processed,
                'description': desc
            }

            logger.info(f"  ✓ Downloaded {len(df):,} total rows, "
                       f"{query_job.total_bytes_processed / 1e9:.2f} GB processed")

            self._save_parquet(df, 'labevents.parquet')
            return df

        except Exception as e:
            logger.error(f"Failed to download labevents: {e}")
            raise

    def download_all(self,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None,
                    lab_names: Optional[List[str]] = None,
                    icd_version: Optional[int] = None):
        """
        Download all tables in optimal order

        Args:
            start_date: Filter time-based tables after this date
            end_date: Filter time-based tables before this date
            lab_names: Filter labevents by specific lab test names
            icd_version: Filter diagnoses by ICD version (9 or 10)
        """
        logger.info("="*80)
        logger.info("Starting MIMIC-IV data download from BigQuery")
        logger.info("="*80)

        # Download small reference tables first
        logger.info("\n[1/7] Downloading reference tables...")
        self.download_d_labitems()
        self.download_d_icd_diagnoses()

        # Download patient demographics
        logger.info("\n[2/7] Downloading patient demographics...")
        self.download_patients()

        # Download admissions
        logger.info("\n[3/7] Downloading hospital admissions...")
        self.download_admissions(start_date, end_date)

        # Download ICU stays
        logger.info("\n[4/7] Downloading ICU stays...")
        self.download_icustays()

        # Download diagnoses
        logger.info("\n[5/7] Downloading ICD diagnoses...")
        self.download_diagnoses_icd(icd_version)

        # Download lab events (largest table - do last)
        logger.info("\n[6/7] Downloading lab events (this will take a while)...")
        self.download_labevents(start_date, end_date, lab_names)

        # Save metadata
        logger.info("\n[7/7] Saving download metadata...")
        self.save_metadata()

        # Summary
        self.print_summary()

    def save_metadata(self):
        """Save download metadata as JSON"""
        metadata_path = os.path.join(self.output_dir, 'download_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        logger.info(f"  ✓ Saved metadata to download_metadata.json")

    def print_summary(self):
        """Print download summary"""
        logger.info("\n" + "="*80)
        logger.info("DOWNLOAD COMPLETE!")
        logger.info("="*80)

        total_rows = sum(t['rows'] for t in self.metadata['tables'].values())
        total_bytes = sum(t.get('bytes_processed', 0) for t in self.metadata['tables'].values())

        logger.info(f"\nTables downloaded: {len(self.metadata['tables'])}")
        logger.info(f"Total rows: {total_rows:,}")
        logger.info(f"Total bytes processed: {total_bytes / 1e9:.2f} GB")
        logger.info(f"Estimated BigQuery cost: ${total_bytes / 1e12 * 5:.2f}")

        # Calculate total file size
        total_size = sum(
            os.path.getsize(os.path.join(self.output_dir, f))
            for f in os.listdir(self.output_dir)
            if f.endswith('.parquet')
        )
        logger.info(f"Total disk space used: {total_size / 1e9:.2f} GB")

        logger.info(f"\nFiles saved to: {os.path.abspath(self.output_dir)}")
        logger.info("\nTable breakdown:")
        for table_name, info in self.metadata['tables'].items():
            logger.info(f"  {table_name:20s} {info['rows']:>12,} rows")


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description='Download MIMIC-IV data from BigQuery to Parquet files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all tables (full dataset)
  python download_mimic_from_bigquery.py --output-dir ./mimic_data

  # Download only recent data (2015 onwards)
  python download_mimic_from_bigquery.py --output-dir ./mimic_data \\
      --start-date 2015-01-01

  # Download specific lab tests only
  python download_mimic_from_bigquery.py --output-dir ./mimic_data \\
      --filter-labs "Hemoglobin,Glucose,Creatinine,Sodium"

  # Download only ICD-9 diagnoses
  python download_mimic_from_bigquery.py --output-dir ./mimic_data \\
      --icd-version 9

Note: Requires BigQuery access to physionet-data.mimiciv_3_1_hosp
      Run 'gcloud auth application-default login' first
        """
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./mimic_data',
        help='Directory to save Parquet files (default: ./mimic_data)'
    )

    parser.add_argument(
        '--project-id',
        type=str,
        default=None,
        help='GCP project ID (uses default if not specified)'
    )

    parser.add_argument(
        '--start-date',
        type=str,
        default=None,
        help='Filter data after this date (YYYY-MM-DD), e.g., 2015-01-01'
    )

    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='Filter data before this date (YYYY-MM-DD), e.g., 2019-12-31'
    )

    parser.add_argument(
        '--filter-labs',
        type=str,
        default=None,
        help='Comma-separated lab test names to include, e.g., "Hemoglobin,Glucose,Creatinine"'
    )

    parser.add_argument(
        '--icd-version',
        type=int,
        choices=[9, 10],
        default=None,
        help='Download only ICD-9 or ICD-10 diagnoses (default: both)'
    )

    parser.add_argument(
        '--table',
        type=str,
        choices=['patients', 'admissions', 'icustays', 'diagnoses', 'labevents', 'all'],
        default='all',
        help='Download specific table only (default: all)'
    )

    args = parser.parse_args()

    # Parse lab names
    lab_names = None
    if args.filter_labs:
        lab_names = [name.strip() for name in args.filter_labs.split(',')]
        logger.info(f"Will filter for {len(lab_names)} specific lab tests")

    # Initialize downloader
    try:
        downloader = MIMICDataDownloader(args.output_dir, args.project_id)
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        return 1

    # Download requested tables
    try:
        if args.table == 'all':
            downloader.download_all(
                start_date=args.start_date,
                end_date=args.end_date,
                lab_names=lab_names,
                icd_version=args.icd_version
            )
        elif args.table == 'patients':
            downloader.download_patients()
        elif args.table == 'admissions':
            downloader.download_admissions(args.start_date, args.end_date)
        elif args.table == 'icustays':
            downloader.download_icustays()
        elif args.table == 'diagnoses':
            downloader.download_diagnoses_icd(args.icd_version)
            downloader.download_d_icd_diagnoses()
        elif args.table == 'labevents':
            downloader.download_d_labitems()
            downloader.download_labevents(args.start_date, args.end_date, lab_names)

        return 0

    except Exception as e:
        logger.error(f"Download failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
