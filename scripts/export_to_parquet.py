#!/usr/bin/env python3
"""
Generate BigQuery EXPORT DATA commands for MIMIC-IV tables based on configuration files.
This creates filtered exports to GCS to reduce download time and storage costs.

Usage:
    python export_mimic_to_gcs.py --bucket biomarker-temp-data --config configs/
"""

import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Set


class BigQueryExportGenerator:
    """Generate optimized BigQuery export commands for MIMIC-IV data."""
    
    def __init__(self, bucket_name: str, project: str = "physionet-data", 
                 dataset: str = "mimiciv_3_1_hosp"):
        self.bucket_name = bucket_name
        self.project = project
        self.dataset = dataset
        
    def load_disease_config(self, config_path: Path) -> Dict:
        """Load disease configuration with ICD-9 codes."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_cbc_config(self, config_path: Path) -> Dict:
        """Load CBC features configuration with item IDs."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get_all_icd9_codes(self, disease_config: Dict) -> List[str]:
        """Extract all ICD-9 codes from disease config."""
        icd9_codes = []
        for disease_key, disease_info in disease_config.get('diseases', {}).items():
            codes = disease_info.get('icd9_codes', [])
            icd9_codes.extend(codes)
        return icd9_codes
    
    def get_all_lab_itemids(self, cbc_config: Dict) -> Set[int]:
        """Extract all lab item IDs from CBC config."""
        itemids = set()
        for feature_name, feature_info in cbc_config.get('cbc_features', {}).items():
            ids = feature_info.get('itemids', [])
            itemids.update(ids)
        return itemids
    
    def generate_labevents_export(self, lab_itemids: Set[int], 
                                  filter_data: bool = True) -> str:
        """
        Generate export command for labevents table.
        
        Args:
            lab_itemids: Set of itemid values to filter for CBC tests
            filter_data: If True, filter by itemids. If False, export all.
        """
        uri = f"gs://{self.bucket_name}/labevents/file-*.parquet"
        table = f"`{self.project}.{self.dataset}.labevents`"
        
        if filter_data and lab_itemids:
            # Filter to only CBC-related lab tests
            itemid_list = ', '.join(map(str, sorted(lab_itemids)))
            query = f"""
EXPORT DATA OPTIONS(
  uri='{uri}',
  format='PARQUET',
  compression='SNAPPY',
  overwrite=true) AS
SELECT * FROM {table}
WHERE itemid IN ({itemid_list})"""
        else:
            # Export all labevents (WARNING: very large!)
            query = f"""
EXPORT DATA OPTIONS(
  uri='{uri}',
  format='PARQUET',
  compression='SNAPPY',
  overwrite=true) AS
SELECT * FROM {table}"""
        
        return query
    
    def generate_diagnoses_export(self, icd9_codes: List[str], 
                                   filter_data: bool = True) -> str:
        """
        Generate export command for diagnoses_icd table.
        
        Args:
            icd9_codes: List of ICD-9 codes to filter for target diseases
            filter_data: If True, filter by ICD-9 codes. If False, export all.
        """
        uri = f"gs://{self.bucket_name}/diagnoses_icd/file-*.parquet"
        table = f"`{self.project}.{self.dataset}.diagnoses_icd`"
        
        if filter_data and icd9_codes:
            # Create LIKE patterns for each ICD-9 code (handle wildcards)
            like_conditions = []
            for code in icd9_codes:
                # Remove trailing 'x' or wildcards and create pattern
                base_code = code.rstrip('x').rstrip('.')
                like_conditions.append(f"icd_code LIKE '{base_code}%'")
            
            where_clause = ' OR '.join(like_conditions)
            
            query = f"""
EXPORT DATA OPTIONS(
  uri='{uri}',
  format='PARQUET',
  compression='SNAPPY',
  overwrite=true) AS
SELECT * FROM {table}
WHERE {where_clause}"""
        else:
            # Export all diagnoses
            query = f"""
EXPORT DATA OPTIONS(
  uri='{uri}',
  format='PARQUET',
  compression='SNAPPY',
  overwrite=true) AS
SELECT * FROM {table}"""
        
        return query
    
    def generate_admissions_export(self, filter_data: bool = False) -> str:
        """
        Generate export command for admissions table.
        
        Note: Usually we want all admissions for context, so filter_data=False by default.
        """
        uri = f"gs://{self.bucket_name}/admissions/file-*.parquet"
        table = f"`{self.project}.{self.dataset}.admissions`"
        
        query = f"""
EXPORT DATA OPTIONS(
  uri='{uri}',
  format='PARQUET',
  compression='SNAPPY',
  overwrite=true) AS
SELECT * FROM {table}"""
        
        return query
    
    def generate_patients_export(self) -> str:
        """Generate export command for patients table."""
        uri = f"gs://{self.bucket_name}/patients/file-*.parquet"
        table = f"`{self.project}.{self.dataset}.patients`"
        
        query = f"""
EXPORT DATA OPTIONS(
  uri='{uri}',
  format='PARQUET',
  compression='SNAPPY',
  overwrite=true) AS
SELECT * FROM {table}"""
        
        return query
    
    def generate_d_labitems_export(self) -> str:
        """Generate export command for d_labitems reference table."""
        uri = f"gs://{self.bucket_name}/d_labitems/file-*.parquet"
        table = f"`{self.project}.{self.dataset}.d_labitems`"
        
        query = f"""
EXPORT DATA OPTIONS(
  uri='{uri}',
  format='PARQUET',
  compression='SNAPPY',
  overwrite=true) AS
SELECT * FROM {table}"""
        
        return query
    
    def generate_d_icd_diagnoses_export(self) -> str:
        """Generate export command for d_icd_diagnoses reference table."""
        uri = f"gs://{self.bucket_name}/d_icd_diagnoses/file-*.parquet"
        table = f"`{self.project}.{self.dataset}.d_icd_diagnoses`"
        
        query = f"""
EXPORT DATA OPTIONS(
  uri='{uri}',
  format='PARQUET',
  compression='SNAPPY',
  overwrite=true) AS
SELECT * FROM {table}"""
        
        return query
    
    def generate_icustays_export(self) -> str:
        """Generate export command for icustays table."""
        uri = f"gs://{self.bucket_name}/icustays/file-*.parquet"
        
        # ICU stays are in the icu dataset, not hosp
        table = f"`{self.project}.mimiciv_3_1_icu.icustays`"
        
        query = f"""
EXPORT DATA OPTIONS(
  uri='{uri}',
  format='PARQUET',
  compression='SNAPPY',
  overwrite=true) AS
SELECT * FROM {table}"""
        
        return query
    
    def generate_all_exports(self, disease_config_path: Path, 
                            cbc_config_path: Path,
                            filter_large_tables: bool = True) -> Dict[str, str]:
        """
        Generate all export commands needed for the project.
        
        Args:
            disease_config_path: Path to diseases.yaml
            cbc_config_path: Path to cbc_features.yaml
            filter_large_tables: If True, filter labevents and diagnoses
        
        Returns:
            Dictionary mapping table name to export SQL
        """
        # Load configurations
        disease_config = self.load_disease_config(disease_config_path)
        cbc_config = self.load_cbc_config(cbc_config_path)
        
        # Extract filter criteria
        icd9_codes = self.get_all_icd9_codes(disease_config)
        lab_itemids = self.get_all_lab_itemids(cbc_config)
        
        # Generate exports
        exports = {
            'labevents': self.generate_labevents_export(lab_itemids, filter_large_tables),
            'diagnoses_icd': self.generate_diagnoses_export(icd9_codes, filter_large_tables),
            'admissions': self.generate_admissions_export(False),  # Usually want all
            'patients': self.generate_patients_export(),
            'd_labitems': self.generate_d_labitems_export(),
            'd_icd_diagnoses': self.generate_d_icd_diagnoses_export(),
            'icustays': self.generate_icustays_export(),
        }
        
        return exports
    
    def write_sql_file(self, exports: Dict[str, str], output_file: Path):
        """
        Write only SQL commands to file (no comments or instructions).
        Each command ends with a semicolon.
        
        Args:
            exports: Dictionary of table name to SQL command
            output_file: File to write SQL commands to
        """
        sql_lines = []
        
        for table_name, sql_command in exports.items():
            # Add the SQL command and ensure it ends with semicolon
            sql_lines.append(sql_command.strip())
            if not sql_command.strip().endswith(';'):
                sql_lines.append(';')
            sql_lines.append('')  # Blank line between commands
        
        sql_text = '\n'.join(sql_lines)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(sql_text)
        print(f"SQL file written to: {output_file}")
    
    def write_instructions_file(self, output_file: Path):
        """
        Write execution instructions to a separate file.
        
        Args:
            output_file: File to write instructions to
        """
        instructions = f"""BigQuery EXPORT DATA Commands for MIMIC-IV
Target bucket: gs://{self.bucket_name}/

================================================================================
EXECUTION INSTRUCTIONS:
================================================================================

1. Create GCS bucket if it doesn't exist:
   gcloud storage buckets create gs://{self.bucket_name} --location=us

2. Run all export commands in BigQuery console or bq CLI:
   bq query --use_legacy_sql=false < export_commands.sql

3. Download from GCS to local:
   gcloud storage cp -r gs://{self.bucket_name}/* ./data/raw/

4. Estimated data sizes after filtering:
   - labevents (filtered CBC only): ~500MB - 2GB
   - diagnoses_icd (filtered): ~50MB - 200MB
   - admissions: ~100MB
   - patients: ~10MB
   - d_labitems: <1MB
   - d_icd_diagnoses: ~10MB
   - icustays: ~50MB

5. Alternatively, use the provided download script:
   bash download_from_gcs.sh
"""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(instructions)
        print(f"Instructions written to: {output_file}")
    
    def generate_download_script(self, output_file: Path):
        """Generate a shell script to download all exported data from GCS."""
        script = f"""#!/bin/bash
# Download MIMIC-IV data from GCS
# Generated automatically by export_mimic_to_gcs.py

set -e  # Exit on error

BUCKET="gs://{self.bucket_name}"
LOCAL_DIR="./mimic_data"

echo "Downloading MIMIC-IV data from $BUCKET to $LOCAL_DIR"
echo "========================================================================"

# Create local directory structure
mkdir -p "$LOCAL_DIR"

# Download each table
tables=("labevents" "diagnoses_icd" "admissions" "patients" "d_labitems" "d_icd_diagnoses" "icustays")

for table in "${{tables[@]}}"; do
    echo ""
    echo "Downloading $table..."
    gcloud storage cp -r "$BUCKET/$table/" "$LOCAL_DIR/$table/"
    
    # Merge parquet files if needed
    echo "Files downloaded for $table:"
    ls -lh "$LOCAL_DIR/$table/"
done

echo ""
echo "========================================================================"
echo "Download complete! Data saved to $LOCAL_DIR/"
echo ""
echo "To verify data:"
echo "  python -c 'import polars as pl; print(pl.scan_parquet(\"$LOCAL_DIR/labevents/*.parquet\").select(pl.count()).collect())'"
"""
        
        output_file.write_text(script)
        output_file.chmod(0o755)  # Make executable
        print(f"Download script created: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate BigQuery export commands for MIMIC-IV data based on project configs"
    )
    parser.add_argument(
        '--bucket',
        type=str,
        required=True,
        help='GCS bucket name (e.g., biomarker-temp-data)'
    )
    parser.add_argument(
        '--config-dir',
        type=Path,
        default=Path('configs'),
        help='Directory containing config files (default: configs/)'
    )
    parser.add_argument(
        '--disease-config',
        type=str,
        default='diseases.yaml',
        help='Disease config filename (default: diseases.yaml)'
    )
    parser.add_argument(
        '--cbc-config',
        type=str,
        default='cbc_features.yaml',
        help='CBC features config filename (default: cbc_features.yaml)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('output'),
        help='Output directory for all generated files (default: output/)'
    )
    parser.add_argument(
        '--no-filter',
        action='store_true',
        help='Export all data without filtering (WARNING: very large!)'
    )
    
    args = parser.parse_args()
    
    # Construct config paths
    disease_config_path = args.config_dir / args.disease_config
    cbc_config_path = args.config_dir / args.cbc_config
    
    # Check if config files exist
    if not disease_config_path.exists():
        print(f"ERROR: Disease config not found: {disease_config_path}")
        print(f"Please create {args.disease_config} in {args.config_dir}/")
        return 1
    
    if not cbc_config_path.exists():
        print(f"ERROR: CBC config not found: {cbc_config_path}")
        print(f"Please create {args.cbc_config} in {args.config_dir}/")
        return 1
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate exports
    generator = BigQueryExportGenerator(bucket_name=args.bucket)
    
    filter_data = not args.no_filter
    if not filter_data:
        print("WARNING: --no-filter specified. This will export ALL data (very large!)")
        print("Press Ctrl+C to cancel, or wait 5 seconds to continue...")
        import time
        time.sleep(5)
    
    exports = generator.generate_all_exports(
        disease_config_path=disease_config_path,
        cbc_config_path=cbc_config_path,
        filter_large_tables=filter_data
    )
    
    # Output files
    sql_file = args.output_dir / 'export_commands.sql'
    instructions_file = args.output_dir / 'INSTRUCTIONS.txt'
    download_script = args.output_dir / 'download_from_gcs.sh'
    
    generator.write_sql_file(exports, output_file=sql_file)
    generator.write_instructions_file(output_file=instructions_file)
    generator.generate_download_script(output_file=download_script)
    
    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print(f"1. Review the export commands in: {sql_file}")
    print(f"2. Review instructions in: {instructions_file}")
    print(f"3. Create GCS bucket: gcloud storage buckets create gs://{args.bucket} --location=us")
    print(f"4. Run the export commands in BigQuery console:")
    print(f"   bq query --use_legacy_sql=false < {sql_file}")
    print(f"5. Download data using: bash {download_script}")
    print("=" * 80)
    
    return 0


if __name__ == '__main__':
    exit(main())