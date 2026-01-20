import hydra
from omegaconf import DictConfig, ListConfig
from hydra.utils import to_absolute_path
import pandas as pd
import gcsfs

def load_table(cfg, key_path, table_name):
    path = f"gs://{cfg.storage.bucket}/{cfg.storage.raw_path}{table_name}/"
    print(f"Loading {table_name} from {path}...")
    return pd.read_parquet(path, storage_options={"token": key_path})

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(f"--- Generating Cohorts for: {cfg.cohort.disease_name} ---")
    
    key_path = to_absolute_path(cfg.storage.key_path)
    
    # --- PHASE 1: HEAVY LIFTING (Do this ONCE) ---
    try:
        df_diag = load_table(cfg, key_path, "diagnoses_icd")
        df_adm = load_table(cfg, key_path, "admissions")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not read data.\n{e}")
        return
    
    # Filter Cases
    print("Filtering ICD codes...")
    target_codes = [str(c) for c in cfg.cohort.icd_codes]
    mask = df_diag['icd_code'].str.startswith(tuple(target_codes), na=False)
    cases_diag = df_diag[mask].copy()
    
    # Find Diagnosis Dates
    print("Finding Diagnosis Dates...")
    cases_diag['hadm_id'] = cases_diag['hadm_id'].astype(int)
    df_adm['hadm_id'] = df_adm['hadm_id'].astype(int)
    
    cases_merged = cases_diag.merge(df_adm[['subject_id', 'hadm_id', 'admittime']], 
                                    on=['subject_id', 'hadm_id'], how='inner')
    cases_merged['admittime'] = pd.to_datetime(cases_merged['admittime'])
    
    # This 'base_cohort' has the Diagnosis Date but NO cutoff calculated yet
    base_cases = cases_merged.groupby('subject_id')['admittime'].min().reset_index()
    base_cases.rename(columns={'admittime': 'diagnosis_date'}, inplace=True)
    
    # Identify Controls (Subjects who are NOT in the case list)
    all_subjects = df_adm['subject_id'].unique()
    case_ids = set(base_cases['subject_id'].unique())
    control_ids = list(set(all_subjects) - case_ids)
    
    print(f"Found {len(case_ids)} Cases and {len(control_ids)} Controls.")

    # --- PHASE 2: THE LOOP (Fast Logic) ---
    # We iterate through the list of windows defined in config
    windows = cfg.study_params.gap_windows
    print(f"Generatng cohorts for gaps: {windows}")

    for gap_days in windows:
        print(f"\nProcessing {gap_days}-day gap...")
        
        # 1. Calculate Cutoff for Cases
        current_cases = base_cases.copy()
        gap = pd.Timedelta(days=gap_days)
        current_cases['cutoff_date'] = current_cases['diagnosis_date'] - gap
        current_cases['label'] = 1
        
        # 2. Setup Controls
        current_controls = pd.DataFrame({'subject_id': control_ids})
        current_controls['label'] = 0
        current_controls['cutoff_date'] = pd.Timestamp.max 
        
        # 3. Merge
        final_cohort = pd.concat([current_cases[['subject_id', 'label', 'cutoff_date']], 
                                  current_controls], ignore_index=True)
        
        # 4. Save with Dynamic Name
        safe_name = cfg.cohort.disease_name.replace(" ", "")
        filename = f"cohort_{safe_name}_{gap_days}d.parquet"
        save_path = f"gs://{cfg.storage.bucket}/{cfg.storage.output_path}/{filename}"
        
        print(f"Saving to {save_path}...")
        final_cohort.to_parquet(save_path, storage_options={"token": key_path})

    print("\nAll cohorts generated successfully!")

if __name__ == "__main__":
    main()