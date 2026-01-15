EXPORT DATA OPTIONS(
  uri='gs://biomarker-temp-data/labevents/file-*.parquet',
  format='PARQUET',
  compression='SNAPPY',
  overwrite=true) AS
SELECT * FROM `physionet-data.mimiciv_3_1_hosp.labevents`
WHERE itemid IN (50810, 50811, 51146, 51200, 51221, 51222, 51244, 51245, 51248, 51249, 51250, 51254, 51256, 51264, 51265, 51277, 51279, 51300, 51301)
;

EXPORT DATA OPTIONS(
  uri='gs://biomarker-temp-data/diagnoses_icd/file-*.parquet',
  format='PARQUET',
  compression='SNAPPY',
  overwrite=true) AS
SELECT * FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd`
WHERE icd_code LIKE '714.0%' OR icd_code LIKE '714.1%' OR icd_code LIKE '714.2%' OR icd_code LIKE '714.3%' OR icd_code LIKE '714.4%' OR icd_code LIKE '714.8%' OR icd_code LIKE '714.9%' OR icd_code LIKE '250.01%' OR icd_code LIKE '250.03%' OR icd_code LIKE '250.11%' OR icd_code LIKE '250.13%' OR icd_code LIKE '250.31%' OR icd_code LIKE '250.33%' OR icd_code LIKE '250.00%' OR icd_code LIKE '250.02%' OR icd_code LIKE '250.10%' OR icd_code LIKE '250.12%' OR icd_code LIKE '250.30%' OR icd_code LIKE '250.32%' OR icd_code LIKE '555.0%' OR icd_code LIKE '555.1%' OR icd_code LIKE '555.2%' OR icd_code LIKE '555.9%' OR icd_code LIKE '696.1%' OR icd_code LIKE '280%' OR icd_code LIKE '281%' OR icd_code LIKE '282%' OR icd_code LIKE '283%' OR icd_code LIKE '284%' OR icd_code LIKE '285%' OR icd_code LIKE '038%' OR icd_code LIKE '995.91%' OR icd_code LIKE '995.92%' OR icd_code LIKE '287.3%' OR icd_code LIKE '287.4%' OR icd_code LIKE '287.5%'
;

EXPORT DATA OPTIONS(
  uri='gs://biomarker-temp-data/admissions/file-*.parquet',
  format='PARQUET',
  compression='SNAPPY',
  overwrite=true) AS
SELECT * FROM `physionet-data.mimiciv_3_1_hosp.admissions`
;

EXPORT DATA OPTIONS(
  uri='gs://biomarker-temp-data/patients/file-*.parquet',
  format='PARQUET',
  compression='SNAPPY',
  overwrite=true) AS
SELECT * FROM `physionet-data.mimiciv_3_1_hosp.patients`
;

EXPORT DATA OPTIONS(
  uri='gs://biomarker-temp-data/d_labitems/file-*.parquet',
  format='PARQUET',
  compression='SNAPPY',
  overwrite=true) AS
SELECT * FROM `physionet-data.mimiciv_3_1_hosp.d_labitems`
;

EXPORT DATA OPTIONS(
  uri='gs://biomarker-temp-data/d_icd_diagnoses/file-*.parquet',
  format='PARQUET',
  compression='SNAPPY',
  overwrite=true) AS
SELECT * FROM `physionet-data.mimiciv_3_1_hosp.d_icd_diagnoses`
;

EXPORT DATA OPTIONS(
  uri='gs://biomarker-temp-data/icustays/file-*.parquet',
  format='PARQUET',
  compression='SNAPPY',
  overwrite=true) AS
SELECT * FROM `physionet-data.mimiciv_3_1_icu.icustays`
;
