import os
import subprocess

diseases = ["type1_diabetes", "diabetes_type2", "rheumatoid_arthritis", "crohns_disease", "psoriasis"]
gaps = [30, 90, 180, 365]

print("--- STARTING NIGHTLY BATCH RUN ---")

# 1. Generate Cohorts
for disease in diseases:
    print(f"\n[Cohort Gen] Processing: {disease}")
    # cohort_gen.py handles all gap windows internally
    subprocess.run(["python", "src/cohort_gen.py", f"cohort={disease}"], check=True)

# 2. Build Features
for disease in diseases:
    for gap in gaps:
        print(f"\n[Feature Build] Processing: {disease} | Gap: {gap}")
        cmd = [
            "python", "src/build_features.py",
            f"cohort={disease}",
            f"gap_days={gap}",
            "features=cbc_expanded" # Make sure this matches your new YAML name
        ]
        subprocess.run(cmd, check=True)

print("\n--- ALL TASKS FINISHED ---")