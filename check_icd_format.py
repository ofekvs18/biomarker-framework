import pandas as pd
from pathlib import Path

# Load ALL diagnoses files
data_dir = Path('data/raw/diagnoses_icd')
parquet_files = list(data_dir.glob("*.parquet"))
dfs = [pd.read_parquet(f) for f in parquet_files]
df = pd.concat(dfs, ignore_index=True)
print(f"Loaded {len(parquet_files)} files, {len(df):,} total rows")

print("=== ICD Code Format in MIMIC Data ===\n")
print("Columns:", df.columns.tolist())
print("\nSample rows:")
print(df.head(20).to_string())

print("\n\n=== Unique ICD versions ===")
print(df['icd_version'].value_counts())

print("\n\n=== Sample ICD-9 codes (version 9) ===")
icd9_samples = df[df['icd_version'] == 9]['icd_code'].head(30).tolist()
print(icd9_samples)

print("\n\n=== Sample ICD-10 codes (version 10) ===")
icd10_samples = df[df['icd_version'] == 10]['icd_code'].head(30).tolist()
print(icd10_samples)

# Check for codes that might match our diseases
print("\n\n=== Looking for Rheumatoid Arthritis codes ===")
# Config uses: 714.0, 714.1, etc. and M05, M06
# But MIMIC has NO DOTS: 7140, 7141, etc.
ra_matches_with_dot = df[df['icd_code'].str.startswith(('714.', 'M05', 'M06'))]
ra_matches_no_dot = df[df['icd_code'].str.startswith(('714', 'M05', 'M06'))]
print(f"Matches with '714.' prefix: {len(ra_matches_with_dot)}")
print(f"Matches with '714' prefix (no dot): {len(ra_matches_no_dot)}")
print("\nSample matching codes:")
print(ra_matches_no_dot['icd_code'].value_counts().head(20))

# Check diabetes codes too
print("\n\n=== Looking for Diabetes codes ===")
# Config: 250.01, 250.00, E10, E11 -> should be 25001, 25000, E10, E11
diabetes_matches = df[df['icd_code'].str.startswith(('250', 'E10', 'E11'))]
print(f"Diabetes matches: {len(diabetes_matches)}")
print(diabetes_matches['icd_code'].value_counts().head(20))

print("\n\n=== FIX NEEDED ===")
print("The diseases.yaml uses codes WITH dots (714.0, 250.01)")
print("But MIMIC data has codes WITHOUT dots (7140, 25001)")
print("Solution: Remove dots from ICD codes in diseases.yaml OR strip dots in the matching code")

print("\n\n=== What code ranges exist in data? ===")
print("First 3 chars of codes (to see ranges):")
df['code_prefix'] = df['icd_code'].str[:3]
print(df['code_prefix'].value_counts().head(30))

print("\n\nAll unique 3-char prefixes:")
print(sorted(df['code_prefix'].unique()))
