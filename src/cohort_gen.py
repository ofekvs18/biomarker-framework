import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
import pandas as pd
import gcsfs

def load_table(cfg, key_path, table_name):
    # Construct path: gs://bucket/raw_data/table_name/
    path = f"gs://{cfg.storage.bucket}/{cfg.storage.raw_path}{table_name}/"
    print(f"Loading {table_name} from {path}...")
    return pd.read_parquet(path, storage_options={"token": key_path})

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(f"--- Generating Cohort: {cfg.cohort.disease_name} ---")
    
    key_path = to_absolute_path(cfg.storage.key_path)
    
    # 1. Load Data
    try:
        df_diag = load_table(cfg, key_path, "diagnoses_icd")
        df_adm = load_table(cfg, key_path, "admissions")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not read data.\n{e}")
        return
    
    # 2. Filter Cases
    print("Filtering for specific ICD codes...")
    target_codes = [str(c) for c in cfg.cohort.icd_codes]
    mask = df_diag['icd_code'].str.startswith(tuple(target_codes), na=False)
    cases_diag = df_diag[mask].copy()
    
    # 3. Find Dates
    print("Finding Index Dates...")
    cases_diag['hadm_id'] = cases_diag['hadm_id'].astype(int)
    df_adm['hadm_id'] = df_adm['hadm_id'].astype(int)
    
    cases_merged = cases_diag.merge(df_adm[['subject_id', 'hadm_id', 'admittime']], 
                                    on=['subject_id', 'hadm_id'], how='inner')
    cases_merged['admittime'] = pd.to_datetime(cases_merged['admittime'])
    
    index_dates = cases_merged.groupby('subject_id')['admittime'].min().reset_index()
    index_dates.rename(columns={'admittime': 'diagnosis_date'}, inplace=True)
    
    # 4. Apply Censoring
    print(f"Applying {cfg.cohort.gap_days} day gap...")
    gap = pd.Timedelta(days=cfg.cohort.gap_days)
    index_dates['cutoff_date'] = index_dates['diagnosis_date'] - gap
    index_dates['label'] = 1
    
    # 5. Handle Controls
    all_subjects = df_adm['subject_id'].unique()
    case_ids = set(index_dates['subject_id'].unique())
    control_ids = list(set(all_subjects) - case_ids)
    
    controls_df = pd.DataFrame({'subject_id': control_ids})
    controls_df['label'] = 0
    controls_df['cutoff_date'] = pd.Timestamp.max 
    
    master_cohort = pd.concat([index_dates[['subject_id', 'label', 'cutoff_date']], 
                               controls_df], ignore_index=True)
    
    # 6. Save with DYNAMIC Name (The Fix)
    safe_name = cfg.cohort.disease_name.replace(" ", "")
    filename = f"cohort_{safe_name}_{cfg.cohort.gap_days}d.parquet"
    
    # Note: We added a '/' manually between output_path and filename
    save_path = f"gs://{cfg.storage.bucket}/{cfg.storage.output_path}/{filename}"
    
    print(f"Saving Master Cohort to {save_path}...")
    master_cohort.to_parquet(save_path, storage_options={"token": key_path})
    print("Done!")

if __name__ == "__main__":
    main()