"""
pipeline_config.py
==================
Central configuration for the Medical Billing Medallion Pipeline.

Load Modes
----------
FULL        – Truncate-and-reload the entire target table.
INCREMENTAL – Append only rows newer than the last high-watermark.
CDC         – Change-Data-Capture: MERGE (upsert + soft-delete) into target.
SCD2        – Slowly-Changing Dimension Type-2: expire old rows, insert new.
WINDOW      – Reload a rolling N-day window (idempotent re-processing).
FDC         – Full-Diff-Compare: compare source vs target, apply only deltas.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# ── Base paths ────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent
RAW_CLEAN_DIR   = BASE_DIR / "raw_data"
RAW_ISSUES_DIR  = BASE_DIR / "raw_data_with_issues"
BRONZE_DIR      = BASE_DIR / "delta" / "bronze"
SILVER_DIR      = BASE_DIR / "delta" / "silver"
GOLD_DIR        = BASE_DIR / "delta" / "gold"
CHECKPOINT_DIR  = BASE_DIR / "checkpoints"
AUDIT_DIR       = BASE_DIR / "audit_logs"

# ── Spark environment ─────────────────────────────────────────────────────────
import platform as _platform

if _platform.system() == "Windows":
    SPARK_ENV = {
        "HADOOP_HOME":           r"C:\hadoop",
        "JAVA_HOME":             r"C:\Program Files\Eclipse Adoptium\jdk-17.0.19.10-hotspot",
        "PYSPARK_PYTHON":        r"C:\pyspark_env\Scripts\python.exe",
        "PYSPARK_DRIVER_PYTHON": r"C:\pyspark_env\Scripts\python.exe",
        "SPARK_LOCAL_IP":        "127.0.0.1",
    }
else:
    # Linux / Docker container
    SPARK_ENV = {
        "JAVA_HOME":             "/usr/lib/jvm/java-17-openjdk-amd64",
        "PYSPARK_PYTHON":        "/home/airflow/.local/bin/python",
        "PYSPARK_DRIVER_PYTHON": "/home/airflow/.local/bin/python",
        "SPARK_LOCAL_IP":        "127.0.0.1",
    }

if _platform.system() == "Windows":
    _DRIVER_MEM  = "6g"
    _EXEC_MEM    = "6g"
    _RESULT_SIZE = "2g"
else:
    # Docker container — leave headroom for Airflow + OS
    _DRIVER_MEM  = "4g"
    _EXEC_MEM    = "4g"
    _RESULT_SIZE = "1g"

SPARK_CONF = {
    "spark.sql.extensions":
        "io.delta.sql.DeltaSparkSessionExtension",
    "spark.sql.catalog.spark_catalog":
        "org.apache.spark.sql.delta.catalog.DeltaCatalog",
    "spark.databricks.delta.schema.autoMerge.enabled": "true",
    "spark.databricks.delta.retentionDurationCheck.enabled": "false",
    "spark.sql.shuffle.partitions":  "8",
    "spark.default.parallelism":     "8",
    "spark.driver.memory":           _DRIVER_MEM,
    "spark.executor.memory":         _EXEC_MEM,
    "spark.driver.maxResultSize":    _RESULT_SIZE,
    "spark.memory.fraction":         "0.8",
    "spark.memory.storageFraction":  "0.3",
    # Spill to disk when memory is exhausted
    "spark.shuffle.spill":           "true",
    "spark.shuffle.spill.compress":  "true",
    "spark.sql.inMemoryColumnarStorage.compressed": "true",
    # Limit partition listing overhead for heavily partitioned tables
    "spark.sql.sources.parallelPartitionDiscovery.threshold": "32",
    "spark.sql.sources.parallelPartitionDiscovery.parallelism": "8",
    "spark.ui.showConsoleProgress":  "false",
    "spark.ui.enabled":              "false",
    # Delta optimise writes
    "spark.databricks.delta.optimizeWrite.enabled":  "true",
    "spark.databricks.delta.autoCompact.enabled":    "true",
    **( {"spark.hadoop.hadoop.home.dir": r"C:\hadoop"} if _platform.system() == "Windows" else {} ),
}

# ── Load-mode constants ───────────────────────────────────────────────────────
FULL        = "FULL"
INCREMENTAL = "INCREMENTAL"
CDC         = "CDC"
SCD2        = "SCD2"
WINDOW      = "WINDOW"
FDC         = "FDC"

VALID_MODES = {FULL, INCREMENTAL, CDC, SCD2, WINDOW, FDC}

# ── Table metadata ────────────────────────────────────────────────────────────
@dataclass
class TableConfig:
    name:            str
    primary_key:     str                        # single PK column
    watermark_col:   Optional[str]  = None      # column used for INCREMENTAL / WINDOW
    default_mode:    str            = FULL      # default load mode
    window_days:     int            = 7         # days for WINDOW mode
    partition_cols:  list           = field(default_factory=list)
    scd2_track_cols: list           = field(default_factory=list)  # cols that trigger SCD2
    is_reference:    bool           = False     # small lookup table


# Reference tables – small, full-reload is fine
_REF = {"default_mode": FULL, "is_reference": True}

# Transactional tables – incremental by default
_TXN = {"default_mode": INCREMENTAL}

TABLE_CONFIGS: dict[str, TableConfig] = {
    t.name: t for t in [
        # ── Reference ──────────────────────────────────────────────────────
        TableConfig("clinics",   "clinic_id",   "created_at",  **_REF,
                    scd2_track_cols=["clinic_name","address","network_status"]),
        TableConfig("providers", "provider_id", "created_at",  **_REF,
                    scd2_track_cols=["specialty","status","accepting_new_patients"]),
        TableConfig("payers",    "payer_id",    None,          **_REF),

        # ── Patients / Insurance ───────────────────────────────────────────
        TableConfig("patients",          "patient_id",          "updated_at",
                    default_mode=CDC,
                    scd2_track_cols=["address","phone","email","employment_status"]),
        TableConfig("patient_insurance", "patient_insurance_id",
                    watermark_col="created_at", default_mode=INCREMENTAL),

        # ── Clinical ──────────────────────────────────────────────────────
        TableConfig("appointments",       "appointment_id",
                    default_mode=INCREMENTAL, watermark_col="created_at"),
        TableConfig("encounters",         "encounter_id",
                    default_mode=INCREMENTAL, watermark_col="created_at"),
        TableConfig("vitals",             "vital_id",
                    default_mode=INCREMENTAL, watermark_col="recorded_at"),
        TableConfig("diagnoses",          "diagnosis_id",
                    default_mode=INCREMENTAL, watermark_col="created_at"),
        TableConfig("procedures",         "procedure_id",
                    default_mode=INCREMENTAL, watermark_col="created_at"),
        TableConfig("lab_results",        "lab_id",
                    default_mode=INCREMENTAL, watermark_col="result_date"),
        TableConfig("medications",        "medication_id",
                    default_mode=INCREMENTAL, watermark_col="prescribed_date"),
        TableConfig("prior_authorizations","auth_id",
                    default_mode=INCREMENTAL, watermark_col="request_date"),
        TableConfig("eligibility_checks", "eligibility_id",
                    default_mode=INCREMENTAL, watermark_col="check_date"),

        # ── Billing ───────────────────────────────────────────────────────
        TableConfig("claims",      "claim_id",
                    default_mode=CDC, watermark_col="created_at",
                    scd2_track_cols=["claim_status","adjudication_date","total_charge"]),
        TableConfig("claim_lines", "claim_line_id",
                    default_mode=INCREMENTAL, watermark_col="service_date"),
        TableConfig("remittances", "remittance_id",
                    default_mode=INCREMENTAL, watermark_col="payment_date"),
        TableConfig("payments",    "payment_id",
                    default_mode=INCREMENTAL, watermark_col="payment_date"),
        TableConfig("denials",     "denial_id",
                    default_mode=CDC, watermark_col="denial_date",
                    scd2_track_cols=["appeal_status","corrected_claim_submitted"]),
        TableConfig("appeals",     "appeal_id",
                    default_mode=CDC, watermark_col="submission_date",
                    scd2_track_cols=["decision","outcome"]),
    ]
}

ALL_TABLES = list(TABLE_CONFIGS.keys())
