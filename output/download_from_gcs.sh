#!/bin/bash
# Download MIMIC-IV data from GCS
# Generated automatically by export_mimic_to_gcs.py

set -e  # Exit on error

BUCKET="gs://biomarker-temp-data"
LOCAL_DIR="./data/raw"

echo "Downloading MIMIC-IV data from $BUCKET to $LOCAL_DIR"
echo "========================================================================"

# Create local directory structure
mkdir -p "$LOCAL_DIR"

# Download each table
tables=("labevents" "diagnoses_icd" "admissions" "patients" "d_labitems" "d_icd_diagnoses" "icustays")

for table in "${tables[@]}"; do
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
echo "  python -c 'import polars as pl; print(pl.scan_parquet("$LOCAL_DIR/labevents/*.parquet").select(pl.count()).collect())'"
