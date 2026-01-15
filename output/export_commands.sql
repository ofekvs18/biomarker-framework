EXPORT DATA OPTIONS(
  uri='gs://biomarker-temp-data/labevents/file-*.parquet',
  format='PARQUET',
  compression='SNAPPY',
  overwrite=true) AS
SELECT * FROM `physionet-data.mimiciv_3_1_hosp.labevents`
;

EXPORT DATA OPTIONS(
  uri='gs://biomarker-temp-data/diagnoses_icd/file-*.parquet',
  format='PARQUET',
  compression='SNAPPY',
  overwrite=true) AS
SELECT * FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd`
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
