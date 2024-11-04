# Medical Billing Data Engineering Pipeline

> **A production-grade Medallion Architecture pipeline for healthcare medical billing data — built with PySpark, Delta Lake, Apache Airflow, and Docker on Windows.**

---

## Table of Contents

1. [Project Purpose](#1-project-purpose)
2. [Architecture Overview](#2-architecture-overview)
3. [Project Structure](#3-project-structure)
4. [Technology Stack & Why Each Was Chosen](#4-technology-stack--why-each-was-chosen)
5. [Data Model — All 20 Tables](#5-data-model--all-20-tables)
6. [Data Generation](#6-data-generation)
7. [Data Quality Injection](#7-data-quality-injection)
8. [Medallion Architecture — Bronze / Silver / Gold](#8-medallion-architecture--bronze--silver--gold)
9. [Load Modes — FULL / INCREMENTAL / CDC / SCD2 / WINDOW / FDC](#9-load-modes)
10. [Pipeline Files — Detailed Walkthrough](#10-pipeline-files--detailed-walkthrough)
11. [Gold Layer — Dimensions, Facts, KPIs](#11-gold-layer--dimensions-facts-kpis)
12. [Audit Logs & Watermarks](#12-audit-logs--watermarks)
13. [Apache Airflow — DAGs & Orchestration](#13-apache-airflow--dags--orchestration)
14. [Running Airflow on Windows — Issues & Solutions](#14-running-airflow-on-windows--issues--solutions)
15. [Docker Setup](#15-docker-setup)
16. [Technical Issues Diagnosed & Resolved](#16-technical-issues-diagnosed--resolved)
17. [Pipeline Optimizations](#17-pipeline-optimizations)
18. [Running Everything — All Commands](#18-running-everything--all-commands)
19. [Best Practices & Tradeoffs](#19-best-practices--tradeoffs)
20. [Glossary](#20-glossary)

---

## 1. Project Purpose

Medical billing is one of the most data-intensive domains in healthcare. A single patient visit generates data across a dozen systems: scheduling, clinical documentation, lab orders, pharmacy, coding, claims submission, payer adjudication, remittance, and collections. This project simulates that entire lifecycle end-to-end.

**Goals:**

- Ingest raw, messy, multi-format healthcare data from multiple source systems
- Apply a Medallion (Bronze → Silver → Gold) architecture to progressively clean and enrich it
- Produce analytics-ready Gold tables: dimensions, a fact table, and 10 KPI aggregations
- Orchestrate the pipeline with Apache Airflow running in Docker
- Demonstrate production data engineering patterns: CDC, SCD2, watermarks, audit logs, DQ flagging, schema evolution, and disk-spill-safe Spark configuration

**Business questions the Gold layer answers:**

- What is our monthly revenue, collection rate, and denial rate?
- Which payers pay the most and deny the most?
- Which providers generate the most revenue and have the lowest denial rates?
- What is our AR aging — how much is outstanding past 90 days?
- Which ICD-10 diagnoses are most prevalent and which patients are highest risk?
- What is our appointment completion rate and no-show rate by month?
- Which lab tests have the highest abnormal rates?

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        SOURCE DATA                              │
│  raw_data/          (clean CSVs + Parquets, chunked per table)  │
│  raw_data_with_issues/  (same + 22 injected DQ issue types)     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BRONZE LAYER                                 │
│  delta/bronze/<table>/                                          │
│  • All columns as StringType (schema-agnostic ingestion)        │
│  • Both clean + dirty sources unioned                           │
│  • Metadata: _ingested_at, _ingestion_date, _row_hash, _source  │
│  • Load modes: FULL | INCREMENTAL | WINDOW | CDC | FDC          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SILVER LAYER                                 │
│  delta/silver/<table>/                                          │
│  • Typed columns (DateType, DoubleType, IntegerType, Boolean)   │
│  • Trimmed strings, standardised case, deduplication            │
│  • 20+ DQ checks: format, range, impossible values, sequences   │
│  • _dq_flag, _dq_issues columns on every row                    │
│  • Load modes: FULL | INCREMENTAL | WINDOW | CDC | SCD2 | FDC   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     GOLD LAYER                                  │
│  delta/gold/<table>/                                            │
│  • 4 Dimensions: dim_patients, dim_providers,                   │
│                  dim_clinics, dim_payers                        │
│  • 1 Fact:       fact_claims                                    │
│  • 10 KPIs:      kpi_revenue_summary, kpi_denial_analysis,      │
│                  kpi_ar_aging, kpi_provider_scorecard,          │
│                  kpi_payer_mix, kpi_procedure_revenue,          │
│                  kpi_diagnosis_burden, kpi_appointment_ops,     │
│                  kpi_patient_risk, kpi_lab_abnormals            │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ORCHESTRATION                                 │
│  Apache Airflow 2.9.3 in Docker                                 │
│  • med_billing_weekly_full_reload   (Sunday 01:00)              │
│  • med_billing_daily_incremental    (nightly 02:00)             │
│  • med_billing_backfill             (manual trigger)            │
│  • med_billing_data_quality_check   (nightly 06:00)             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Project Structure

```
medical_billing_project_v2/
│
├── data_generator.py            # Generates 2+ GB of realistic raw data
├── data_quality_injector.py     # Injects 22 DQ issue types into raw data
│
├── raw_data/                    # Clean source data (chunked per table)
│   ├── appointments/            #   appointments_part_000.csv, _001.parquet ...
│   ├── claims/
│   ├── claim_lines/
│   └── ... (15 tables total)
│
├── raw_data_with_issues/        # Dirty source data (same structure)
│   └── ... (same 15 tables)
│
├── delta/                       # Delta Lake storage (written by PySpark)
│   ├── bronze/                  #   <table>/_delta_log/ + *.parquet
│   ├── silver/
│   └── gold/
│
├── checkpoints/                 # Spark checkpoint dir (disk spill)
│   ├── bronze/
│   ├── cdc/
│   └── spark_ckpt/
│
├── audit_logs/                  # JSON watermark + run history per table
│   ├── bronze/
│   └── silver/
│
├── pipelines/
│   ├── pipeline_config.py       # Central config: paths, Spark conf, table metadata
│   ├── pipeline_utils.py        # SparkSession factory, write strategies, audit helpers
│   ├── 01_bronze_ingestion.py   # Bronze layer
│   ├── 02_silver_transform.py   # Silver layer
│   ├── 03_gold_aggregations.py  # Gold layer
│   └── run_pipeline.py          # Master orchestrator (CLI entry point)
│
└── airflow/
    ├── Dockerfile               # apache/airflow:2.9.3 + Java 17 + PySpark + Delta
    ├── docker-compose.yml       # Single-container Airflow (SequentialExecutor)
    ├── airflow.cfg              # Airflow configuration
    ├── airflow.db               # SQLite metadata database
    ├── dags/
    │   ├── dag_daily_incremental.py
    │   ├── dag_weekly_full_reload.py
    │   ├── dag_backfill.py
    │   └── dag_data_quality_check.py
    ├── plugins/
    │   └── med_billing_operator.py   # Custom PythonOperator wrapper
    ├── setup_airflow.ps1
    ├── start_airflow.ps1
    └── stop_airflow.ps1
```

---

## 4. Technology Stack & Why Each Was Chosen

### PySpark 3.5.1
The dataset is 2+ GB of raw data expanding to ~5 GB across all layers. Pandas would load everything into RAM and OOM on a laptop. PySpark processes data in partitions, spills to disk when memory is exhausted, and writes directly to storage without materialising the full dataset in the driver. It also provides the Delta Lake integration needed for ACID writes, schema evolution, and time travel.

### Delta Lake 3.2.0
Delta Lake adds ACID transactions, schema enforcement, and time travel on top of plain Parquet files. Key reasons for choosing it here:

- **ACID writes** — if a Spark job crashes mid-write, the Delta log ensures the table is not left in a corrupt state. The transaction either commits or rolls back.
- **Schema evolution** — `autoMerge.enabled=true` allows new columns to be added to source data without breaking existing tables.
- **MERGE (upsert)** — CDC mode uses `DeltaTable.merge()` to upsert rows by primary key, which is impossible with plain Parquet.
- **Time travel** — every write creates a versioned snapshot. You can query `VERSION AS OF 0` to see the original data.
- **Optimise writes** — `optimizeWrite.enabled=true` automatically coalesces small files into larger ones, preventing the "small files problem" that kills query performance.

### Apache Airflow 2.9.3
Airflow provides DAG-based scheduling, dependency management, retry logic, task-level logging, and a web UI. The four DAGs cover the full operational lifecycle: weekly full reload, nightly incremental, manual backfill, and nightly DQ checks. Each task maps to one pipeline layer, making failures easy to isolate and retry.

### Docker
Airflow does not officially support Windows. Running it in a Linux container (via Docker Desktop + WSL2) gives a proper POSIX environment. The container mounts the project directory at `/opt/project`, so the pipeline code runs against the same Delta files on the Windows filesystem.

### Python 3.11 / C:\pyspark_env
A dedicated virtual environment at `C:\pyspark_env` isolates PySpark, Delta, Pandas, Faker, and Airflow from the system Python. This avoids version conflicts and makes the environment reproducible.

### SQLite (Airflow metadata)
For a single-developer local setup, SQLite is sufficient. The `SequentialExecutor` runs one task at a time, which is appropriate for a laptop where Spark already consumes most available memory.

---

## 5. Data Model — All 20 Tables

### Reference Tables (small, full-reload)

| Table | PK | Key Columns | Rows |
|---|---|---|---|
| `clinics` | clinic_id | npi, clinic_type, network_status, bed_count | 75 |
| `providers` | provider_id | npi, specialty, credential, board_certified, accepting_new_patients | 500 |
| `payers` | payer_id | payer_name, payer_type, timely_filing_limit_days | 16 |

### Patient / Insurance Tables

| Table | PK | Key Columns | Rows |
|---|---|---|---|
| `patients` | patient_id | mrn, dob, age, gender, race, zip_code, smoking_status, bmi_category | 408,000 |
| `patient_insurance` | patient_insurance_id | patient_id, payer_id, plan_type, effective_date, deductible, copay | 530,844 |

### Clinical Tables

| Table | PK | Watermark | Rows |
|---|---|---|---|
| `appointments` | appointment_id | created_at | 1,415,662 |
| `encounters` | encounter_id | created_at | 921,472 |
| `vitals` | vital_id | recorded_at | 921,472 |
| `diagnoses` | diagnosis_id | created_at | 2,301,721 |
| `procedures` | procedure_id | created_at | 2,304,838 |
| `lab_results` | lab_id | result_date | 8,010,347 |
| `medications` | medication_id | prescribed_date | 1,612,807 |
| `prior_authorizations` | auth_id | request_date | 184,293 |

### Billing Tables

| Table | PK | Watermark | Notes |
|---|---|---|---|
| `claims` | claim_id | created_at | CDC mode — status changes over time |
| `claim_lines` | claim_line_id | service_date | One row per CPT code per claim |
| `remittances` | remittance_id | payment_date | ERA/835 payment batches |
| `payments` | payment_id | payment_date | Individual payment postings |
| `denials` | denial_id | denial_date | CDC — appeal status changes |
| `appeals` | appeal_id | submission_date | CDC — decision changes |
| `eligibility_checks` | eligibility_id | check_date | Real-time eligibility verifications |

**Total raw data: ~2.9 GB across 14 tables with source data**

---

## 6. Data Generation

**File:** `data_generator.py`

Generates realistic synthetic healthcare data using the `Faker` library plus domain-specific reference data. Key design decisions:

### Realistic Reference Data
- 20 medical specialties with proper taxonomy codes
- 16 real payers (Medicare, Medicaid, Aetna, BCBS, UHC, Cigna, etc.) with correct payer codes
- 44 ICD-10 codes covering the most common chronic and acute conditions
- 48 CPT codes covering office visits, procedures, labs, imaging, and surgery
- 22 drug entries with NDC codes, dosing, and days supply
- Lab panels: CBC, CMP, Lipid, and specialty tests with reference ranges

### Chunked Output
Each large table is written as multiple numbered chunk files (e.g., `appointments_part_000.csv`, `appointments_part_001.parquet`). This simulates real source systems that export data in batches. 60% of chunks are CSV, 40% are Parquet — forcing the pipeline to handle mixed formats.

### Age / Demographics Distribution
Patient ages follow a normal distribution centred at 58 (skewed toward Medicare age), reflecting a realistic hospital population. Gender, race, smoking status, and BMI category use weighted random selection matching US population statistics.

### Referential Integrity
All foreign keys are generated from actual parent IDs. Every encounter references a real patient, provider, and clinic. Every claim references a real encounter. This ensures the Gold joins produce meaningful results.

### Running the Generator

```powershell
# From project root
C:\pyspark_env\Scripts\python.exe data_generator.py
```

Output: `raw_data/` — approximately 2.19 GB across 15 table folders.

---

## 7. Data Quality Injection

**File:** `data_quality_injector.py`

Takes the clean `raw_data/` and produces `raw_data_with_issues/` with 22 types of realistic data quality problems injected at configurable rates.

### Issue Types Injected

| # | Issue Type | Example | Rate |
|---|---|---|---|
| 1 | `missing_value` | NULL in phone, email, NPI | 6% |
| 2 | `invalid_format` | `"not-a-date"` in date col, `"ABC"` in NPI | 4% |
| 3 | `out_of_range` | age=200, charge=-50000 | 4% |
| 4 | `future_date` | appointment_date = 2027-01-01 | 3% |
| 5 | `past_date` | dob = 1890-03-15 | 3% |
| 6 | `impossible_values` | systolic_bp=500, pain_scale=20 | 4% |
| 7 | `wrong_data_type` | `"TEXT"` in numeric column | 2.5% |
| 8 | `special_characters` | `"John!@#$%"` in name field | 3% |
| 9 | `duplicate_records` | Same row with `_DUP` suffix on ID | 4% |
| 10 | `inconsistent_data` | status=Active but termination_date in past | 4% |
| 11 | `logical_inconsistency` | discharge_date before admit_date | 4% |
| 12 | `negative_amounts` | paid_amount=-500 | 3% |
| 13 | `swapped_columns` | systolic/diastolic values exchanged | 2% |
| 14 | `truncated_text` | `"Hypertens"` instead of `"Hypertension"` | 3% |
| 15 | `case_inconsistency` | `"MALE"`, `"male"`, `"MaLe"` in same column | 4% |
| 16 | `whitespace_pollution` | `"  John  "`, `"New  York"` | 4% |
| 17 | `encoding_corruption` | Unicode lookalikes replacing ASCII chars | 2% |
| 18 | `orphan_records` | patient_id pointing to non-existent patient | 2.5% |
| 19 | `amount_mismatch` | deductible_met > deductible_individual | 3% |
| 20 | `date_sequence_error` | end_date < start_date | 2.5% |
| 21 | `referential_break` | FK replaced with `INVALID_<uuid>` | 2.5% |
| 22 | `stale_status` | status=Scheduled with date 5 years ago | 3% |

### Why Inject Issues?
The pipeline's Silver layer is designed to detect and flag these issues, not silently drop data. Every row that passes a DQ check gets `_dq_flag=False`. Every row that fails one or more checks gets `_dq_flag=True` and `_dq_issues="invalid_npi, negative_total_charge"`. This allows downstream analysts to query flagged rows separately and make informed decisions about whether to include or exclude them.

### Running the Injector

```powershell
C:\pyspark_env\Scripts\python.exe data_quality_injector.py
```

Output: `raw_data_with_issues/` — approximately same size as `raw_data/` but with injected problems.

---

## 8. Medallion Architecture — Bronze / Silver / Gold

The Medallion Architecture is a data lakehouse design pattern that organises data into three progressive quality layers. Each layer has a specific contract about what it contains and what guarantees it provides.

### Why Medallion?

**Alternative considered: single-layer ETL**
A traditional ETL would read raw data, clean it, and write directly to a final table. The problem: if the cleaning logic has a bug, you have to re-read the raw source (which may no longer be available) and re-run everything. There is no intermediate checkpoint.

**Why Medallion wins:**
- Bronze preserves the raw data exactly as received — it is the system of record
- Silver can be rebuilt from Bronze at any time by re-running the cleaning logic
- Gold can be rebuilt from Silver at any time by re-running the aggregations
- Each layer can be independently versioned, monitored, and audited
- DQ issues are flagged in Silver but not dropped — analysts can query them

### Bronze Layer Contract
- **What it is:** Raw data, exactly as received, with all columns as `StringType`
- **What it guarantees:** Every source row is preserved. No data is dropped. Schema drift is tolerated.
- **Metadata added:** `_ingested_at`, `_ingestion_date`, `_row_hash`, `_source` (raw_clean vs raw_issues), `_bronze_table`
- **Why all strings?** Source data has type errors injected (numbers in text columns, text in numeric columns). Casting everything to String at ingestion prevents Spark from rejecting rows with type mismatches. The Silver layer handles proper casting.

### Silver Layer Contract
- **What it is:** Cleaned, typed, deduplicated, DQ-flagged data
- **What it guarantees:** All columns have correct types. Strings are trimmed. Duplicates by PK are removed. Every row has a DQ flag.
- **Cleaning applied to every table:**
  - Trim whitespace from all string columns
  - Standardise case (proper-case names, UPPER codes/IDs, lower emails)
  - Cast to proper types (DateType, DoubleType, IntegerType, BooleanType)
  - Drop rows where PK is null
  - Remove exact duplicates by primary key
  - Validate formats: phone, email, NPI (10 digits), zip (5 digits), ICD-10 pattern
  - Flag impossible values: negative ages, future birthdates, BP=500
  - Fix date sequences: swap end < start pairs (and flag them)
  - Null out negative financial amounts (and flag them)
- **Metadata added:** `_silver_table`, `_cleaned_at`, `_silver_date`, `_dq_flag`, `_dq_issues`, `_row_hash`

### Gold Layer Contract
- **What it is:** Analytics-ready aggregations, dimensions, and facts
- **What it guarantees:** Business-meaningful columns only. No internal metadata columns. Joins are pre-computed. KPIs are pre-aggregated.
- **Tables produced:** 4 dimensions + 1 fact + 10 KPI tables (see Section 11)

---

## 9. Load Modes

Every table in every layer supports six load modes. The mode is chosen based on the nature of the data and the operational requirement.

### FULL
Truncate the target table and reload everything from source. Used for:
- Reference tables (clinics, providers, payers) — small enough that full reload is fast
- Weekly scheduled runs to catch any rows missed by incremental loads
- After a bug fix in cleaning logic that requires reprocessing all historical data

```powershell
python pipelines/run_pipeline.py --mode FULL
```

### INCREMENTAL
Read only rows where the watermark column (e.g., `created_at`) is newer than the last recorded watermark. Append to the target table. Used for:
- Nightly loads of transactional tables (appointments, encounters, lab_results)
- Any table where rows are only ever added, never updated

```powershell
python pipelines/run_pipeline.py --mode INCREMENTAL
```

### CDC (Change Data Capture)
MERGE source rows into the target using the primary key. Matched rows are updated; new rows are inserted. Used for:
- `claims` — claim status changes from Submitted → Adjudicated → Paid/Denied
- `patients` — address, phone, email can change
- `denials` — appeal status changes over time

```powershell
python pipelines/run_pipeline.py --mode CDC
```

### SCD2 (Slowly Changing Dimension Type 2)
When tracked columns change, expire the old row (set `_effective_to` = today, `_is_current` = false) and insert a new row. Preserves full history of changes. Used for:
- `providers` — specialty or accepting_new_patients status changes
- `clinics` — network_status changes (in-network → out-of-network)

```powershell
python pipelines/run_pipeline.py --mode SCD2
```

### WINDOW
Re-process a rolling N-day window. Idempotent — running it twice produces the same result. Used for:
- Backfill after a source system correction
- Re-cleaning a date range after a bug fix in Silver logic

```powershell
python pipelines/run_pipeline.py --mode WINDOW --window-days 30
```

### FDC (Full Diff Compare)
Hash every source row. Compare hashes with target. Write only rows whose hash changed (new or updated). Soft-delete rows that disappeared from source. Used for:
- Reference tables where the source system does not provide change timestamps
- Detecting silent updates (row changed but no updated_at column)

```powershell
python pipelines/run_pipeline.py --mode FDC
```

---

## 10. Pipeline Files — Detailed Walkthrough

### `pipeline_config.py`
Central configuration. Contains:
- **Path constants:** `BASE_DIR`, `RAW_CLEAN_DIR`, `BRONZE_DIR`, `SILVER_DIR`, `GOLD_DIR`, `CHECKPOINT_DIR`, `AUDIT_DIR`
- **Platform-aware Spark environment:** Detects Windows vs Linux (Docker) and sets `JAVA_HOME`, `PYSPARK_PYTHON`, `HADOOP_HOME` accordingly
- **Platform-aware memory:** Windows gets 6 GB driver/executor; Docker gets 4 GB to leave headroom for Airflow and the OS
- **`SPARK_CONF` dict:** All Spark configuration including Delta extensions, shuffle partitions, memory fractions, disk spill settings
- **`TableConfig` dataclass:** Per-table metadata — primary key, watermark column, default load mode, partition columns, SCD2 tracked columns
- **`TABLE_CONFIGS` dict:** One `TableConfig` entry for all 20 tables
- **`ALL_TABLES` list:** Ordered list of all table names

### `pipeline_utils.py`
Shared utilities used by all three layers:

**`get_spark(app_name)`**
- Stops any existing active session before creating a new one (critical for per-table session restart)
- Applies all `SPARK_CONF` settings
- Sets the checkpoint directory to `checkpoints/spark_ckpt/` so `.checkpoint()` spills to disk

**Write strategies:**
- `write_full(df, path, partition_cols)` — overwrites entire table; reads count from Delta log after write (not from the DataFrame, to avoid re-scanning)
- `write_append(df, path, partition_cols)` — appends rows; same count strategy
- `write_cdc_merge(spark, source_df, target_path, pk)` — Delta MERGE upsert
- `write_scd2(spark, source_df, target_path, pk, track_cols)` — expire old rows, insert new; uses `.checkpoint()` before counting to avoid double computation
- `write_fdc(spark, source_df, target_path, pk)` — hash comparison, write only deltas; uses `.checkpoint()` on intermediate DataFrames

**`dispatch_write()`** — routes to the correct write strategy based on mode

**Audit helpers:**
- `read_watermark(layer, table)` — reads last high-watermark from JSON
- `write_watermark(layer, table, watermark)` — persists watermark after successful run
- `write_audit_log(layer, table, stats)` — appends run record to JSON history (keeps last 50 runs)

### `01_bronze_ingestion.py`
Per-table ingestion flow:
1. Read all CSV + Parquet files from `raw_data/<table>/` and `raw_data_with_issues/<table>/`
2. Cast all columns to `StringType` (schema-agnostic)
3. Union clean and dirty sources
4. Detect broken Delta tables (empty `_delta_log`) and force FULL reload
5. Apply watermark filter for INCREMENTAL/WINDOW modes
6. Add bronze metadata columns
7. Write via `dispatch_write`
8. Persist watermark and audit log

**Per-table session restart:** Each table gets its own `get_spark()` call and `spark.stop()` in a `try/finally` block. This prevents JVM heap accumulation from killing subsequent tables.

### `02_silver_transform.py`
Per-table cleaning flow:
1. Check Delta log for valid commits before attempting to read Bronze (avoids errors on broken tables)
2. Read Bronze Delta table
3. Apply watermark filter for INCREMENTAL/WINDOW
4. Run table-specific cleaner function (one per table, ~20 lines each)
5. Add silver metadata and row hash
6. Write via `dispatch_write`
7. Read DQ counts from the written Delta table (not from the source DataFrame)
8. Persist watermark and audit log

**Table-specific cleaners** handle:
- `clean_patients`: age range, future DOB, email format, zip format, gender/smoking inconsistency
- `clean_vitals`: all vital sign ranges, systolic/diastolic swap detection
- `clean_claims`: NPI format, negative charges, date sequence
- `clean_lab_results`: order/result date sequence, result value range
- ... (one cleaner per table, all in `CLEANERS` dict)

### `03_gold_aggregations.py`
Per-table Gold build flow:
1. Read required Silver tables via `_silver(spark, table)` helper
2. Build the Gold DataFrame (joins, aggregations, computed columns)
3. Add Gold metadata (`_gold_table`, `_computed_at`, `_gold_date`)
4. Write via `dispatch_write` — no `rdd.isEmpty()` or `df.count()` before writing
5. Read row count from written Delta table
6. Write audit log

**Critical ordering:** `fact_claims` is built before KPIs that read it. Since each table gets its own Spark session, `fact_claims` is written to disk first, then KPI builders read it via `_gold(spark, "fact_claims")`.

### `run_pipeline.py`
Master orchestrator:
- Parses CLI arguments: `--layers`, `--mode`, `--bronze-mode`, `--silver-mode`, `--gold-mode`, `--tables`, `--window-days`, `--dry-run`
- Resolves per-layer modes (layer-specific overrides the global mode)
- Runs layers in sequence: bronze → silver → gold
- Counts OK vs SKIP vs ERROR per layer (SKIP = no source data, not a failure)
- Exits with code 0 if no ERRORs (PARTIAL/SKIP is acceptable)

---

## 11. Gold Layer — Dimensions, Facts, KPIs

### Why Dimensions, Facts, and KPIs?

The Gold layer follows a **star schema** design. Dimensions are the "who/what/where" lookup tables. The fact table is the central transactional table (claims). KPIs are pre-aggregated summaries that answer specific business questions without requiring analysts to write complex joins.

**Why pre-aggregate in Gold instead of querying Silver directly?**
- Silver has 8M lab_results rows. A KPI query on Silver runs a full scan every time. Gold pre-computes the answer once and stores it as ~2,000 rows.
- Gold removes all internal metadata columns (`_dq_flag`, `_ingested_at`, etc.) — analysts see only business columns.
- Gold joins are pre-computed — `fact_claims` already has encounter_date, place_of_service, and payment totals joined in.

### Dimension Tables

**`dim_patients`**
Joins `silver.patients` with `silver.patient_insurance` to add `primary_payer_id` and `primary_plan_type`. Deduplicates by `patient_id`. Keeps demographic columns: dob, age, gender, race, zip_code, smoking_status, bmi_category.

**`dim_providers`**
Joins `silver.providers` with `silver.clinics` to add `clinic_name` and `clinic_type`. Keeps: npi, specialty, credential, board_certified, accepting_new_patients, years_experience.

**`dim_clinics`**
Direct pass-through from `silver.clinics` with metadata columns dropped.

**`dim_payers`**
Direct pass-through from `silver.payers` with metadata columns dropped.

### Fact Table

**`fact_claims`**
The central fact table. Built by joining:
- `silver.claims` (base)
- `silver.claim_lines` aggregated by claim_id → total_billed, total_allowed, total_paid, total_patient_resp, line_count
- `silver.encounters` → encounter_date, place_of_service_code, encounter_type
- `silver.payments` aggregated by claim_id → payment_received, last_payment_date

Computed columns added:
- `days_to_adjudication` = adjudication_date - submission_date
- `is_paid` = claim_status == 'PAID'
- `is_denied` = claim_status == 'DENIED'
- `collection_rate` = payment_received / total_charge * 100

### KPI Tables

| KPI Table | Grain | Key Metrics |
|---|---|---|
| `kpi_revenue_summary` | Year + Month | total_claims, total_charge, total_paid, collection_rate_pct, denial_rate_pct, avg_days_to_adjudication |
| `kpi_denial_analysis` | denial_code + category | denial_count, total_denied_amount, appeal_count, overturn_rate_pct |
| `kpi_ar_aging` | aging_bucket + payer | claim_count, total_outstanding, avg_days_outstanding (buckets: 0-30, 31-60, 61-90, 91-120, 120+) |
| `kpi_provider_scorecard` | provider | total_claims, total_revenue, avg_charge, denial_rate_pct, paid_rate_pct, avg_days_to_adjudication |
| `kpi_payer_mix` | payer | claim_count, total_billed, total_paid, avg_allowed_pct, denial_rate_pct, pct_of_total_revenue |
| `kpi_procedure_revenue` | CPT code | utilization_count, total_billed, total_paid, avg_reimbursement_rate_pct |
| `kpi_diagnosis_burden` | ICD-10 code | patient_count, encounter_count, primary_dx_count, avg_encounters_per_patient |
| `kpi_appointment_ops` | Year + Month + type | total_scheduled, completed, cancelled, no_show, completion_rate_pct, no_show_rate_pct |
| `kpi_patient_risk` | patient | chronic_condition_count, total_diagnoses, risk_score (0-100), risk_tier (Low/Medium/High) |
| `kpi_lab_abnormals` | test_name + abnormal_flag | abnormal_count, total_tests, abnormal_rate_pct, avg/min/max result_value |

---

## 12. Audit Logs & Watermarks

**Location:** `audit_logs/bronze/<table>.json`, `audit_logs/silver/<table>.json`

Every successful table run writes a JSON record containing:

```json
{
  "table": "claims",
  "layer": "silver",
  "last_watermark": "2025-12-31 23:59:59",
  "updated_at": "2026-05-31T08:00:00+00:00",
  "history": [
    {
      "table": "claims",
      "mode": "FULL",
      "input_rows": 510000,
      "output_rows": 498234,
      "dropped_rows": 11766,
      "dq_flagged": 23410,
      "elapsed_s": 34.2,
      "status": "OK",
      "logged_at": "2026-05-31T08:00:00+00:00"
    }
  ]
}
```

**Why JSON instead of a database table?**
- No dependency on a running database
- Human-readable — you can inspect any table's history with a text editor
- The DQ check DAG reads these files directly to compute alert thresholds
- Keeps last 50 runs per table — sufficient for operational monitoring

**Watermark mechanism:**
- INCREMENTAL mode reads `last_watermark` and filters source data to rows where `watermark_col > last_watermark`
- After a successful write, the new watermark (max value of `watermark_col` in the batch) is saved
- If no prior watermark exists (first run), all rows are loaded (`cutoff = "1900-01-01"`)
- WINDOW mode ignores the stored watermark and uses `NOW() - window_days` as the cutoff

---

## 13. Apache Airflow — DAGs & Orchestration

### DAG: `med_billing_weekly_full_reload`
**Schedule:** Every Sunday at 01:00  
**Strategy:** Bronze FULL → Silver FULL → Gold FULL  
**Task graph:**
```
pipeline_start
      │
  bronze_full
      │
  silver_full
      │
  ┌───┴───┐
gold_dims  gold_fact_claims
  └───┬───┘
  gold_kpis
      │
pipeline_end
```
Gold dimensions and fact_claims run in parallel (both depend only on silver_full). Gold KPIs run after both (they read fact_claims from disk).

### DAG: `med_billing_daily_incremental`
**Schedule:** Every night at 02:00  
**Strategy:** Bronze INCREMENTAL → Silver CDC → Gold INCREMENTAL  
**Task graph:** Linear — bronze → silver → gold

### DAG: `med_billing_backfill`
**Schedule:** Manual trigger only  
**Strategy:** Configurable via DAG run config JSON  
**Parameters:**
```json
{
  "window_days": 30,
  "layers": "bronze,silver,gold",
  "mode": "WINDOW",
  "tables": "claims,claim_lines"
}
```

### DAG: `med_billing_data_quality_check`
**Schedule:** Every night at 06:00 (after the 02:00 incremental)  
**What it checks:**
- DQ flag rate > 10% per table → alert
- Row drop rate > 10% per table → alert
- Watermark not advanced in 26 hours → alert
- Any silver table with 0 output rows → alert

All checks run in parallel, then a final `compile_and_report` task aggregates all alerts. Alerts are logged — the pipeline does not fail on DQ alerts (informational only).

### Custom Operator: `MedBillingPipelineOperator`
**File:** `airflow/plugins/med_billing_operator.py`

Wraps `run_pipeline.py` as a subprocess. Streams stdout/stderr line-by-line to the Airflow task log so you can watch Spark progress in real time. Raises `AirflowException` only on non-zero exit code with actual ERRORs (SKIP statuses from missing source tables are not treated as failures).

```python
MedBillingPipelineOperator(
    task_id="bronze_full",
    layers=["bronze"],
    mode="FULL",
)
```

---

## 14. Running Airflow on Windows — Issues & Solutions

### The Core Problem
Apache Airflow officially supports only POSIX-compliant operating systems (Linux, macOS). On Windows it raises a `RuntimeWarning` and several features break:
- Symlink creation for log directories fails (`OSError: symlink`)
- The scheduler uses `fork()` which is not available on Windows
- File path separators cause issues in DAG file discovery

### Approach 1: Native Windows (Attempted, Abandoned)
Running Airflow directly in `C:\pyspark_env` using `python -m airflow` produces the POSIX warning and the symlink error. The scheduler starts but log rotation and some scheduler internals fail silently.

**Verdict:** Not reliable for production use. Abandoned.

### Approach 2: WSL2 (Considered)
Running Airflow inside WSL2 (Ubuntu) would give a proper Linux environment. However, accessing Windows filesystem paths from WSL2 adds complexity (`/mnt/c/...` paths), and PySpark inside WSL2 needs its own Java installation separate from the Windows one.

**Verdict:** Viable but complex. Not adopted for this project.

### Approach 3: Docker (Adopted)
Running Airflow in a Docker container (via Docker Desktop + WSL2 backend) gives:
- A proper Linux environment for Airflow
- The project directory mounted at `/opt/project` — same Delta files accessible from both Windows and the container
- PySpark + Delta Lake installed in the container image
- Java 17 installed in the container

**The docker-compose.yml mounts:**
```yaml
volumes:
  - ./dags:/opt/airflow/dags:ro
  - ./plugins:/opt/airflow/plugins:ro
  - airflow-data:/opt/airflow
  - ..:/opt/project          # ← entire project mounted here
```

**Platform detection in `pipeline_config.py`:**
```python
import platform as _platform

if _platform.system() == "Windows":
    SPARK_ENV = {
        "HADOOP_HOME": r"C:\hadoop",
        "JAVA_HOME": r"C:\Program Files\Eclipse Adoptium\jdk-17.0.19.10-hotspot",
        ...
    }
else:
    SPARK_ENV = {
        "JAVA_HOME": "/usr/lib/jvm/java-17-openjdk-amd64",
        "PYSPARK_PYTHON": "/home/airflow/.local/bin/python",
        ...
    }
```

This single config file works correctly on both Windows (direct pipeline runs) and Linux (Airflow container runs).

### Memory Allocation Difference
- **Windows direct run:** 6 GB driver + 6 GB executor (full machine available)
- **Docker container:** 4 GB driver + 4 GB executor (leaves headroom for Airflow scheduler, webserver, and OS)

---

## 15. Docker Setup

### Container Configuration

**Image:** `apache/airflow:2.9.3-python3.11` (base image already has PySpark 3.5.1 and delta-spark 3.2.0 installed)

**Executor:** `SequentialExecutor` — runs one task at a time. Appropriate for a laptop where Spark already uses most available memory. For production, switch to `LocalExecutor` (requires PostgreSQL instead of SQLite).

**Ports:** `8080` → Airflow web UI

### Starting the Container

```powershell
# From the airflow/ directory
cd airflow
docker-compose up -d

# Check it's healthy
docker ps --filter "name=medbilling"

# View logs
docker logs medbilling_airflow --tail 50 -f
```

### Stopping the Container

```powershell
docker-compose down
# To also remove the volume (wipes Airflow DB):
docker-compose down -v
```

### Accessing the Web UI

```
http://localhost:8080
Username: admin
Password: Admin123
```

### Running Commands Inside the Container

```bash
# List DAGs
docker exec medbilling_airflow airflow dags list

# Trigger a DAG manually
docker exec medbilling_airflow airflow dags trigger med_billing_weekly_full_reload

# Check task states
docker exec medbilling_airflow airflow tasks states-for-dag-run \
  med_billing_weekly_full_reload \
  manual__2026-05-31T06:57:32+00:00

# View task log
docker exec medbilling_airflow bash -c \
  "tail -50 /opt/airflow/logs/dag_id=med_billing_weekly_full_reload/run_id=.../task_id=bronze_full/attempt=1.log"

# Test PySpark inside container
docker exec medbilling_airflow python -c "
import sys
sys.path.insert(0, '/opt/project')
sys.path.insert(0, '/opt/project/pipelines')
from pipeline_utils import get_spark
spark = get_spark('test')
df = spark.read.format('delta').load('/opt/project/delta/silver/clinics')
print('rows:', df.count())
spark.stop()
"
```

### Rebuilding the Image (if Dockerfile changes)

```powershell
cd airflow
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

---

## 16. Technical Issues Diagnosed & Resolved

### Issue 1: Gold Layer Completely Empty

**Symptom:** After running the full pipeline, `delta/gold/` was empty. Bronze had 2.91 GB, Silver had 1.31 GB, Gold had 0 bytes.

**Root cause:** In `process_gold_table()`, the code called `df.rdd.isEmpty()` before writing. This triggers a full Spark action that materialises the entire DataFrame in memory. For large DataFrames (fact_claims joins claims + lines + encounters + payments), this caused an OOM error. The exception was caught, the function returned `"OK (empty)"`, and nothing was written.

**Diagnosis:** Traced through the code and found `df.rdd.isEmpty()` followed by `df.count()` — two full scans before a single byte was written.

**Fix:**
```python
# BEFORE (wrong)
if df is None or df.rdd.isEmpty():   # ← full scan, OOM
    stats["status"] = "OK (empty)"
    return stats
df = _meta(df, table_name)
row_count = df.count()               # ← second full scan
write_stats = dispatch_write(...)

# AFTER (correct)
if df is None:
    stats["status"] = "OK (empty)"
    return stats
df = _meta(df, table_name)
write_stats = dispatch_write(...)    # ← write first
row_count = spark.read.format("delta").load(target_path).count()  # ← count from disk
```

---

### Issue 2: Silver Layer Stopped After 9 Tables

**Symptom:** Silver processed only 9 of 20 tables. `diagnoses`, `medications`, `providers`, `payers`, `claims`, and others had no silver audit logs and no silver Delta tables.

**Root cause:** The silver `run()` function created one SparkSession and processed all tables sequentially. After `lab_results` (8M rows, 734 MB bronze), the JVM heap was exhausted. The SparkSession died. Python caught the exception at the table level, but the session was dead — all subsequent tables failed silently with no audit log written.

**Diagnosis:** Checked audit logs — only 9 tables had entries. Checked bronze table sizes — `lab_results` at 734 MB was the last table before the gap. Confirmed by checking container memory: 4.47 GB used out of 7.7 GB available.

**Fix:** Per-table SparkSession restart in all three layers:
```python
for table in tables:
    spark = None
    try:
        spark = get_spark("MedBilling_Silver")   # fresh JVM per table
        s = process_table(spark, table, mode=mode)
    except Exception as exc:
        s = {"table": table, "status": f"ERROR: {exc}"}
    finally:
        if spark is not None:
            try:
                spark.stop()                     # release JVM heap
            except Exception:
                pass
    all_stats.append(s)
```

Also updated `get_spark()` to stop any existing active session before creating a new one, since `getOrCreate()` would otherwise return the dead session.

---

### Issue 3: Claims Bronze Never Ingested

**Symptom:** `delta/bronze/claims/` existed but contained only an empty `_delta_log` directory with no commit files and no Parquet files. No bronze audit log for claims.

**Root cause:** Claims bronze ingestion failed in a previous run (likely OOM from the same JVM heap exhaustion issue). The write started, created the `_delta_log` directory, then crashed before writing any commit. This left a "broken" Delta table — the directory exists but has no valid state.

**Diagnosis:** `Get-ChildItem delta/bronze/claims/_delta_log` returned nothing. `Get-ChildItem delta/bronze/claims` showed only the empty `_delta_log` directory.

**Fix:** Added broken-Delta detection in `ingest_table()`:
```python
target_dir = BRONZE_DIR / table_name
if target_dir.exists():
    delta_log = target_dir / "_delta_log"
    if delta_log.exists():
        commit_files = list(delta_log.glob("*.json"))
        if len(commit_files) == 0:
            log.warning("[%s] broken delta log – forcing FULL reload", table_name)
            effective_mode = FULL
            shutil.rmtree(str(target_dir))   # remove broken directory
            target_dir.mkdir(parents=True, exist_ok=True)
```

---

### Issue 4: Appointments Bronze Took 922 Seconds

**Symptom:** The first successful bronze run took 40+ minutes. The `appointments` table alone took 922 seconds (15 minutes).

**Root cause:** `appointments` was configured with `partition_cols=["appointment_date"]`. The dataset spans 70 years of appointment history (1955–2026), creating **16,940 unique date partitions**. Writing 1.4M rows into 16,940 separate directories caused massive filesystem overhead — Spark had to create a directory, write a file, and close it 16,940 times.

**Diagnosis:** Checked `delta/bronze/appointments` — found 16,940 subdirectories named `appointment_date=YYYY-MM-DD`. Confirmed by timing: appointments took 922s while encounters (similar row count, no partitioning) took 652s.

**Fix:** Removed date-based partition columns from `appointments`, `encounters`, and `lab_results`:
```python
# BEFORE
TableConfig("appointments", "appointment_id",
            default_mode=INCREMENTAL, watermark_col="created_at",
            partition_cols=["appointment_date"])   # ← 16,940 partitions

# AFTER
TableConfig("appointments", "appointment_id",
            default_mode=INCREMENTAL, watermark_col="created_at")
```

**Result:** Appointments bronze went from 922 seconds to **54 seconds** — 17x faster.

**Tradeoff:** Without date partitioning, queries that filter by `appointment_date` will do a full table scan instead of partition pruning. For a 1.4M row table on a local setup, this is acceptable. In production with billions of rows, partition by year-month (`appointment_year_month`) instead of full date to limit partition count to ~840 instead of 16,940.

---

### Issue 5: Claims Bronze Failed with DeltaFileFormatWriter Error

**Symptom:** Claims bronze failed with `DeltaFileFormatWriter: Aborting job` during the retry run.

**Root cause:** `claims` was configured with `partition_cols=["submission_date"]`. Claims span 3+ years of submission dates — hundreds of unique dates, each creating a separate partition directory. The write failed mid-way through creating partitions.

**Fix:** Removed `submission_date` from claims partition columns (same fix as Issue 4).

---

### Issue 6: `write_full` and `write_append` Called `df.count()` After Writing

**Symptom:** Every table took longer than expected. Memory pressure was higher than necessary.

**Root cause:** Both write functions called `df.count()` after writing to get the row count for logging. This re-scanned the entire source DataFrame — the same data that was just written.

**Fix:** Read the count from the written Delta table instead:
```python
def write_full(df, path, partition_cols=None):
    w = df.write.format("delta").mode("overwrite")...
    w.save(path)
    # Count from Delta, not from source DF
    count = DeltaTable.forPath(spark, path).toDF().count()
    return count
```

---

### Issue 7: Airflow Previous Runs All Failed (Windows Path Issue)

**Symptom:** All previous Airflow DAG runs showed `failed` status. The container had been running for 15 hours but every run failed.

**Root cause:** `pipeline_config.py` had hardcoded Windows paths in `SPARK_ENV`:
```python
SPARK_ENV = {
    "HADOOP_HOME": r"C:\hadoop",           # ← Windows path, invalid in Linux
    "JAVA_HOME": r"C:\Program Files\...",  # ← Windows path, invalid in Linux
}
```
Inside the Docker container (Linux), these paths don't exist. PySpark failed to start.

**Diagnosis:** Checked previous run logs — all failed at the `bronze_full` task with Spark startup errors.

**Fix:** Platform detection:
```python
import platform as _platform
if _platform.system() == "Windows":
    SPARK_ENV = {"HADOOP_HOME": r"C:\hadoop", "JAVA_HOME": r"C:\Program Files\..."}
else:
    SPARK_ENV = {"JAVA_HOME": "/usr/lib/jvm/java-17-openjdk-amd64", ...}
```

---

### Issue 8: `write_scd2` Called `to_insert.count()` Twice

**Symptom:** SCD2 writes were slow and caused memory pressure.

**Root cause:**
```python
if to_insert.count() > 0:        # ← first full computation
    write_append(to_insert, ...)
return {"inserted": to_insert.count(), ...}  # ← second full computation
```

**Fix:** Checkpoint to disk, count once:
```python
to_insert = to_insert.checkpoint()   # spill to disk
insert_count = to_insert.count()     # count once
if insert_count > 0:
    write_append(to_insert, ...)
return {"inserted": insert_count, ...}
```

---

### Issue 9: Silver `process_table` Crashed on Broken Bronze Tables

**Symptom:** Silver threw exceptions for tables whose bronze Delta log was empty, causing the entire silver run to log errors for those tables.

**Fix:** Check Delta log before attempting to read:
```python
bronze_delta_log = Path(bronze_path) / "_delta_log"
if not bronze_delta_log.exists() or not list(bronze_delta_log.glob("*.json")):
    log.warning("[%s] bronze delta log missing – skipping silver", table_name)
    stats["status"] = "SKIP (bronze not ready)"
    return stats
```

---

### Issue 10: Airflow Task Failed on PARTIAL Status (SKIP Tables)

**Symptom:** The `bronze_full` Airflow task failed even when 14/20 tables succeeded. The 6 "errors" were actually SKIP statuses for tables with no source data (`eligibility_checks`, `remittances`, `payments`, `denials`, `appeals`).

**Root cause:** `run_pipeline.py` exited with code 1 if any layer had a non-OK status. SKIP was counted as an error.

**Fix:** Distinguish ERROR from SKIP in the exit code logic:
```python
# Count separately
ok   = sum(1 for s in stats if s.get("status","").startswith("OK"))
skip = sum(1 for s in stats if s.get("status","").startswith("SKIP"))
err  = sum(1 for s in stats if s.get("status","").startswith("ERROR"))

# Only fail on actual errors
failed = [l for l, r in results.items()
          if not r["status"].startswith("OK") and not r["status"].startswith("PARTIAL")]
sys.exit(1 if failed else 0)
```

---

## 17. Pipeline Optimizations

### 1. Per-Table SparkSession Restart
**Problem:** One session for all tables accumulates JVM heap garbage across tables. After processing large tables, the heap is exhausted and subsequent tables fail.  
**Solution:** Each table gets its own `get_spark()` + `spark.stop()`. The JVM is fully recycled between tables.  
**Cost:** ~5-10 seconds per table for JVM startup. For 14 tables, that's ~2 minutes overhead.  
**Benefit:** Eliminates OOM failures on large tables. Every table gets a clean 4-6 GB heap.

### 2. Write First, Count from Delta
**Problem:** Calling `df.count()` before or after writing re-scans the entire DataFrame.  
**Solution:** Write first, then read the count from the Delta table (which reads from the Delta log metadata, not a full scan).  
**Benefit:** Eliminates one full DataFrame scan per table per layer = ~30% reduction in total compute time.

### 3. Remove High-Cardinality Date Partitions
**Problem:** Partitioning `appointments` by `appointment_date` created 16,940 directories.  
**Solution:** Remove date partitioning from tables with high-cardinality date columns.  
**Benefit:** 17x speedup for appointments (922s → 54s).  
**Rule of thumb:** Only partition by a column if it has fewer than ~500 distinct values in the dataset. For time-series data, partition by year-month, not full date.

### 4. Disk Spill Configuration
```python
"spark.shuffle.spill": "true",
"spark.shuffle.spill.compress": "true",
"spark.memory.fraction": "0.8",
"spark.memory.storageFraction": "0.3",
```
When Spark runs out of heap for shuffle data, it spills to disk instead of failing. This is essential for large joins (fact_claims joins 4 tables).

### 5. Checkpoint for Multi-Step DataFrames
For SCD2 and FDC operations that compute a DataFrame, then use it multiple times (count + write), `.checkpoint()` materialises it to disk once. Subsequent uses read from the checkpoint file instead of recomputing the entire lineage.

### 6. Delta Optimise Write + Auto Compact
```python
"spark.databricks.delta.optimizeWrite.enabled": "true",
"spark.databricks.delta.autoCompact.enabled": "true",
```
`optimizeWrite` coalesces small output files into larger ones during the write. `autoCompact` merges small files after the write. Both prevent the "small files problem" where thousands of tiny Parquet files slow down subsequent reads.

### 7. Partition Discovery Parallelism
```python
"spark.sql.sources.parallelPartitionDiscovery.threshold": "32",
"spark.sql.sources.parallelPartitionDiscovery.parallelism": "8",
```
For tables with many source files (lab_results has 161 CSV/Parquet files), parallel partition discovery speeds up the initial file listing.

### 8. Reduced Shuffle Partitions
```python
"spark.sql.shuffle.partitions": "8",
"spark.default.parallelism": "8",
```
Default is 200 shuffle partitions. On a local machine with 8 cores, 200 partitions creates 200 tiny tasks that spend more time on task scheduling overhead than actual work. Setting to 8 matches the available parallelism.

---

## 18. Running Everything — All Commands

### Prerequisites

```powershell
# Install Python environment (one time)
python -m venv C:\pyspark_env
C:\pyspark_env\Scripts\activate
pip install pyspark==3.5.1 delta-spark==3.2.0 pandas faker numpy pyarrow

# Install Java 17 (required by PySpark on Windows)
# Download from: https://adoptium.net/
# Install to: C:\Program Files\Eclipse Adoptium\jdk-17.0.19.10-hotspot

# Install Hadoop winutils (required by PySpark on Windows)
# Download winutils.exe and hadoop.dll to C:\hadoop\bin\
```

### Generate Data

```powershell
# From project root
C:\pyspark_env\Scripts\python.exe data_generator.py
# Output: raw_data/ (~2.19 GB)

C:\pyspark_env\Scripts\python.exe data_quality_injector.py
# Output: raw_data_with_issues/ (~same size with DQ issues)
```

### Run Pipeline Directly (Windows, no Airflow)

```powershell
# Full pipeline, all layers, FULL mode
python pipelines/run_pipeline.py

# Incremental load
python pipelines/run_pipeline.py --mode INCREMENTAL

# Only bronze layer
python pipelines/run_pipeline.py --layers bronze

# Only silver and gold
python pipelines/run_pipeline.py --layers silver,gold

# Specific tables only
python pipelines/run_pipeline.py --tables claims,claim_lines,patients

# Different mode per layer
python pipelines/run_pipeline.py --bronze-mode FULL --silver-mode CDC --gold-mode INCREMENTAL

# Rolling 30-day window backfill
python pipelines/run_pipeline.py --mode WINDOW --window-days 30

# Dry run (print plan without executing)
python pipelines/run_pipeline.py --dry-run

# Run individual layers
python pipelines/01_bronze_ingestion.py --mode FULL
python pipelines/02_silver_transform.py --mode FULL
python pipelines/03_gold_aggregations.py --mode FULL

# Run individual layers for specific tables
python pipelines/01_bronze_ingestion.py --mode INCREMENTAL --tables claims,claim_lines
python pipelines/02_silver_transform.py --mode CDC --tables claims
```

### Start Airflow (Docker)

```powershell
# Start container
cd airflow
docker-compose up -d

# Check health
docker ps --filter "name=medbilling"
# Wait for: (healthy) status

# Open web UI
start http://localhost:8080
# Login: admin / Admin123
```

### Trigger DAGs via CLI

```powershell
# Weekly full reload (all layers, FULL mode)
docker exec medbilling_airflow airflow dags trigger med_billing_weekly_full_reload

# Daily incremental
docker exec medbilling_airflow airflow dags trigger med_billing_daily_incremental

# Backfill with config
docker exec medbilling_airflow airflow dags trigger med_billing_backfill `
  --conf '{"window_days": 30, "layers": "bronze,silver,gold", "mode": "WINDOW"}'

# Data quality check
docker exec medbilling_airflow airflow dags trigger med_billing_data_quality_check

# List all DAGs
docker exec medbilling_airflow airflow dags list

# List recent runs
docker exec medbilling_airflow airflow dags list-runs -d med_billing_weekly_full_reload

# Check task states for a specific run
docker exec medbilling_airflow airflow tasks states-for-dag-run \
  med_billing_weekly_full_reload \
  manual__2026-05-31T06:57:32+00:00

# Clear a failed task to retry immediately
docker exec medbilling_airflow airflow tasks clear \
  med_billing_weekly_full_reload -s 2026-05-31 -e 2026-06-01 --yes
```

### Monitor Pipeline Progress

```powershell
# Watch bronze task log live
docker exec medbilling_airflow bash -c \
  "tail -f /opt/airflow/logs/dag_id=med_billing_weekly_full_reload/run_id=.../task_id=bronze_full/attempt=1.log"

# Check container resource usage
docker stats medbilling_airflow --no-stream

# Check audit logs
Get-Content audit_logs/bronze/claims.json | ConvertFrom-Json | Select-Object -ExpandProperty history | Select-Object -Last 3

# Check which silver tables completed
Get-ChildItem audit_logs/silver/*.json | ForEach-Object {
    $j = Get-Content $_ | ConvertFrom-Json
    "$($_.BaseName): $($j.history[-1].status) out=$($j.history[-1].output_rows)"
}

# Check Delta table sizes
Get-ChildItem delta/silver -Directory | ForEach-Object {
    $size = (Get-ChildItem $_.FullName -Recurse -Filter "*.parquet" | Measure-Object -Property Length -Sum).Sum
    "$($_.Name): $([math]::Round($size/1MB,1)) MB"
}
```

### Stop Everything

```powershell
# Stop Airflow container
cd airflow
docker-compose down

# Stop all Docker containers
docker stop $(docker ps -q)
```

---

## 19. Best Practices & Tradeoffs

### Best Practices Applied

**1. Schema-agnostic Bronze ingestion**  
All columns ingested as `StringType`. This means Bronze never rejects a row due to a type mismatch. The Silver layer handles casting. This is the correct pattern for a data lake — preserve everything, clean later.

**2. Idempotent writes**  
FULL mode uses `mode("overwrite")` with `overwriteSchema=true`. Running it twice produces the same result. WINDOW mode re-processes the same date range idempotently. This makes reruns safe.

**3. Watermark-based incremental loads**  
Watermarks are stored in JSON files, not in the data itself. This means the watermark survives even if the target table is dropped and rebuilt. The watermark is only updated after a successful write — a failed write does not advance the watermark.

**4. DQ flagging, not dropping**  
Rows with DQ issues are flagged (`_dq_flag=True`) but not dropped. This preserves data for investigation. Analysts can query `WHERE _dq_flag = FALSE` for clean data or `WHERE _dq_flag = TRUE` to investigate issues.

**5. Audit trail**  
Every table run writes a JSON audit record with input rows, output rows, dropped rows, DQ flagged count, elapsed time, and status. This provides full operational visibility without a separate monitoring database.

**6. Per-table session isolation**  
Each table gets its own SparkSession. A failure in one table does not affect others. The JVM heap is fully recycled between tables.

**7. Disk-spill-safe configuration**  
`spark.shuffle.spill=true` ensures Spark spills to disk rather than OOMing when processing large joins. The checkpoint directory is set explicitly so `.checkpoint()` writes to a known location.

### Tradeoffs

**Per-table session restart vs. single session**  
- Single session: faster (no JVM startup overhead), but one OOM kills all subsequent tables
- Per-table restart: ~5-10s overhead per table, but each table is isolated
- **Decision:** Per-table restart. On a laptop with 8-16 GB RAM processing 2.9 GB of data, isolation is more important than the startup overhead.

**No date partitioning on large tables**  
- With partitioning: faster queries that filter by date (partition pruning)
- Without partitioning: faster writes (no directory explosion), simpler maintenance
- **Decision:** No date partitioning for tables with high-cardinality dates. Use Delta's built-in file statistics for query optimisation instead.

**SQLite vs PostgreSQL for Airflow**  
- SQLite: zero setup, single file, sufficient for SequentialExecutor
- PostgreSQL: required for LocalExecutor (parallel tasks), more robust
- **Decision:** SQLite for local development. Switch to PostgreSQL for production.

**SequentialExecutor vs LocalExecutor**  
- SequentialExecutor: one task at a time, no parallelism
- LocalExecutor: multiple tasks in parallel (requires PostgreSQL)
- **Decision:** SequentialExecutor. With Spark already using 4-6 GB, running two Spark jobs in parallel would OOM the machine.

**JSON audit logs vs Delta table**  
- JSON files: no Spark dependency, human-readable, instant writes
- Delta table: queryable with SQL, supports time travel
- **Decision:** JSON files. The DQ check DAG reads them with plain Python — no Spark session needed for monitoring.

**All-string Bronze vs typed Bronze**  
- All-string: tolerates any source data, never rejects rows
- Typed: faster downstream processing, catches type errors early
- **Decision:** All-string. The whole point of Bronze is to be a faithful copy of the source. Type enforcement belongs in Silver.

---

## 20. Glossary

| Term | Definition |
|---|---|
| **Medallion Architecture** | A data lakehouse design pattern with Bronze (raw), Silver (clean), and Gold (aggregated) layers |
| **Delta Lake** | An open-source storage layer that adds ACID transactions, schema enforcement, and time travel to Parquet files |
| **ACID** | Atomicity, Consistency, Isolation, Durability — guarantees that database transactions are processed reliably |
| **CDC** | Change Data Capture — detecting and capturing changes (inserts, updates, deletes) in source data |
| **SCD2** | Slowly Changing Dimension Type 2 — a technique for tracking historical changes by expiring old rows and inserting new ones |
| **Watermark** | A high-water mark timestamp used to track the last successfully processed record in incremental loads |
| **DQ Flag** | Data Quality Flag — a boolean column (`_dq_flag`) indicating whether a row has any data quality issues |
| **Partition** | A physical subdivision of a Delta/Parquet table by column value (e.g., one directory per date) |
| **Shuffle** | A Spark operation that redistributes data across partitions, typically triggered by joins and aggregations |
| **Checkpoint** | Writing a DataFrame to disk to break the lineage chain and prevent recomputation |
| **JVM Heap** | The Java Virtual Machine memory used by Spark. When exhausted, Spark throws OutOfMemoryError |
| **Spill** | When Spark runs out of heap memory, it writes intermediate data to disk (spill) instead of failing |
| **DAG** | Directed Acyclic Graph — in Airflow, a collection of tasks with defined dependencies |
| **SequentialExecutor** | An Airflow executor that runs one task at a time (suitable for local development) |
| **Star Schema** | A data warehouse design with a central fact table surrounded by dimension tables |
| **Fact Table** | A table containing measurable, quantitative data about business events (e.g., claims) |
| **Dimension Table** | A table containing descriptive attributes about business entities (e.g., patients, providers) |
| **KPI** | Key Performance Indicator — a pre-aggregated metric that answers a specific business question |
| **ICD-10** | International Classification of Diseases, 10th revision — standardised diagnosis codes |
| **CPT** | Current Procedural Terminology — standardised procedure codes used for billing |
| **NPI** | National Provider Identifier — a 10-digit unique identifier for healthcare providers |
| **ERA/835** | Electronic Remittance Advice — the electronic version of an Explanation of Benefits from a payer |
| **AR Aging** | Accounts Receivable Aging — a report showing how long outstanding claims have been unpaid |
| **Adjudication** | The process by which a payer evaluates and processes a claim for payment |
| **Denial** | When a payer refuses to pay a claim, typically with a reason code |
| **Prior Authorization** | Approval required from a payer before certain services can be rendered |
| **HCC** | Hierarchical Condition Category — a risk adjustment model used by Medicare |
| **DRG** | Diagnosis Related Group — a classification system for inpatient hospital stays used for payment |
| **winutils** | A Windows utility required by Hadoop (and therefore PySpark) to handle file system operations on Windows |

---

*Built with PySpark 3.5.1 · Delta Lake 3.2.0 · Apache Airflow 2.9.3 · Docker · Python 3.11*
