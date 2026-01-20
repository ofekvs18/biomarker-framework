import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
import pandas as pd
import gcsfs

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(f"--- Building Features: {cfg.features.name} ---")
    
    # 1. Resolve Authentication Key
    key_path = to_absolute_path(cfg.storage.key_path)
    
    # 2. Construct Dynamic Paths
    # We need to load the SPECIFIC cohort file we generated earlier
    # Logic: "cohort_Type1Diabetes_30d.parquet"
    safe_disease_name = cfg.cohort.disease_name.replace(" ", "")
    cohort_filename = f"cohort_{safe_disease_name}_{cfg.gap_days}d.parquet"
    cohort_path = f"gs://{cfg.storage.bucket}/{cfg.storage.output_path}/{cohort_filename}"
    
    print(f"Loading Cohort Rules from: {cohort_path}")
    try:
        cohort = pd.read_parquet(cohort_path, storage_options={"token": key_path})
    except Exception as e:
        print(f"CRITICAL ERROR: Could not find cohort file.\nDid you run cohort_gen.py with the same settings?\n{e}")
        return

    # 3. Load Lab Events (Optimized)
    # We only read the specific columns we need to save memory
    labs_path = f"gs://{cfg.storage.bucket}/{cfg.storage.raw_path}labevents/"
    print(f"Loading Lab Events from: {labs_path}")
    
    # Check if we are filtering for specific IDs to speed up logic later
    target_items = set(cfg.features.lab_ids)
    print(f"Targeting {len(target_items)} specific Lab Item IDs (CBC)...")

    # Note: If your data is massive, consider using Dask here. 
    # For now, we read with Pandas but only keep essential columns.
    labs = pd.read_parquet(
        labs_path, 
        columns=['subject_id', 'itemid', 'valuenum', 'charttime'],
        storage_options={"token": key_path}
    )
    
    # 4. Filter for CBC Items Only
    # This drops millions of irrelevant rows (like chemistry or urine tests) immediately
    labs = labs[labs['itemid'].isin(target_items)]
    
    if labs.empty:
        print("WARNING: No rows matched your Lab IDs. Check if ItemIDs are correct for MIMIC-IV.")
        return

    # 5. Merge & Censor (Leakage Prevention)
    print("Merging Labs with Cohort Rules...")
    # We merge to attach the 'cutoff_date' to every single lab result
    merged = labs.merge(cohort[['subject_id', 'cutoff_date']], on='subject_id', how='inner')
    
    # Ensure charttime is datetime for comparison
    merged['charttime'] = pd.to_datetime(merged['charttime'])
    
    print("Censoring future data...")
    rows_before = len(merged)
    
    # THE CRITICAL LINE: Keep only data that happened BEFORE the index cutoff
    valid_data = merged[merged['charttime'] < merged['cutoff_date']]
    rows_after = len(valid_data)
    
    print(f"Dropped {rows_before - rows_after} rows that occurred after the diagnosis/cutoff.")

    # 6. Pivot / Aggregate (Expanded)
    print(f"Aggregating features using: {cfg.features.aggregation}...")
    
    # Convert OmegaConf list to standard python list if needed
    agg_funcs = list(cfg.features.aggregation)
    
    # Group by Patient + ItemID -> Calculate ALL statistics at once
    # Result is a DataFrame with columns: ['mean', 'max', 'min', 'std', 'count']
    grouped = valid_data.groupby(['subject_id', 'itemid'])['valuenum'].agg(agg_funcs)
    
    # Unstack moves 'itemid' to columns, creating a MultiIndex
    # Structure: (Aggregation, ItemID) -> ('mean', 51221), ('max', 51221)...
    X_features = grouped.unstack(fill_value=0)
    
    # FLATTEN THE COLUMNS
    # We want names like: "lab_51221_mean", "lab_51221_max"
    new_columns = []
    for agg_name, item_id in X_features.columns:
        new_columns.append(f"lab_{item_id}_{agg_name}")
        
    X_features.columns = new_columns
    
    print(f"Generated {len(X_features.columns)} features (was {len(cfg.features.lab_ids)} labs * {len(agg_funcs)} metrics).")

    # 7. Final Assembly (Same as before)
    final_dataset = cohort[['subject_id', 'label']].merge(X_features, on='subject_id', how='left')
    
    # 8. Save Final Dataset
    # Naming: "dataset_Type1Diabetes_30d_cbc.parquet"
    output_filename = f"dataset_{safe_disease_name}_{cfg.gap_days}d_{cfg.features.name}.parquet"
    save_path = f"gs://{cfg.storage.bucket}/{cfg.storage.output_path}/{output_filename}"
    
    print(f"Saving Final Dataset ({final_dataset.shape}) to {save_path}...")
    final_dataset.to_parquet(save_path, storage_options={"token": key_path})
    print("Done! Dataset is ready for training.")

if __name__ == "__main__":
    main()