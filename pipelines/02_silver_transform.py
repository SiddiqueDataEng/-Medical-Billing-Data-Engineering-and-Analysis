"""
02_silver_transform.py
======================
Silver Layer — Cleaning, Validation & Standardisation (Production-Grade)

Supported load modes
--------------------
FULL        – Re-clean the entire bronze table and overwrite silver.
INCREMENTAL – Clean only rows ingested since the last watermark, append to silver.
WINDOW      – Re-clean a rolling N-day window (idempotent).
CDC         – Merge cleaned rows into silver using primary key (upsert).
SCD2        – Slowly-Changing Dimension Type-2 for reference/patient tables.
FDC         – Full-Diff-Compare: only write rows whose hash changed.

Cleaning applied to every table
--------------------------------
  * Trim whitespace from all string columns
  * Standardise case  (proper-case names, UPPER codes/IDs, lower emails)
  * Remove exact duplicate rows (business-key dedup)
  * Cast columns to proper types (DateType, DoubleType, IntegerType, BooleanType)
  * Null handling  — drop rows where PK is null
  * Format validation  — phone, email, NPI (10 digits), zip (5 digits), ICD-10
  * Impossible-value removal  (negative ages, future birthdates, etc.)
  * Date-sequence repair  (swap end < start pairs)
  * _dq_flag / _dq_issues  metadata columns on every output row

Usage (CLI)
-----------
    python pipelines/02_silver_transform.py [--mode FULL|INCREMENTAL|WINDOW|CDC|SCD2|FDC]
                                            [--tables clinics,patients,...]
                                            [--window-days 7]
"""

import argparse
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "pipelines"))

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DateType, DoubleType, IntegerType, BooleanType, StringType,
)

from pipeline_config import (
    BRONZE_DIR, SILVER_DIR, ALL_TABLES, TABLE_CONFIGS,
    FULL, INCREMENTAL, WINDOW, CDC, SCD2, FDC,
)
from pipeline_utils import (
    get_spark, read_watermark, write_watermark, write_audit_log,
    dispatch_write, print_run_summary, log,
)

LAYER = "silver"


# =============================================================================
#  Generic DQ helpers
# =============================================================================

def _col_exists(df: DataFrame, col: str) -> bool:
    return col in df.columns


def _trim_strings(df: DataFrame) -> DataFrame:
    for field in df.schema.fields:
        if isinstance(field.dataType, StringType):
            df = df.withColumn(field.name, F.trim(F.col(field.name)))
    return df


def _cast_date(df, c):
    return df.withColumn(c, F.col(c).cast(DateType())) if _col_exists(df, c) else df

def _cast_double(df, c):
    return df.withColumn(c, F.col(c).cast(DoubleType())) if _col_exists(df, c) else df

def _cast_int(df, c):
    return df.withColumn(c, F.col(c).cast(IntegerType())) if _col_exists(df, c) else df

def _cast_bool(df, c):
    return df.withColumn(c, F.col(c).cast(BooleanType())) if _col_exists(df, c) else df


def _init_dq(df: DataFrame) -> DataFrame:
    return df.withColumn("_dq_flag", F.lit(False)).withColumn("_dq_issues", F.lit(""))


def _flag(df: DataFrame, condition, issue_text: str) -> DataFrame:
    df = df.withColumn("_dq_flag",
        F.when(condition, F.lit(True)).otherwise(F.col("_dq_flag")))
    df = df.withColumn("_dq_issues",
        F.when(condition,
            F.when(F.col("_dq_issues") == "", F.lit(issue_text))
             .otherwise(F.concat(F.col("_dq_issues"), F.lit(", "), F.lit(issue_text)))
        ).otherwise(F.col("_dq_issues")))
    return df


def _fix_date_seq(df, start_col, end_col, label):
    if not (_col_exists(df, start_col) and _col_exists(df, end_col)):
        return df
    swapped = (F.col(end_col) < F.col(start_col)) & \
              F.col(start_col).isNotNull() & F.col(end_col).isNotNull()
    df = _flag(df, swapped, label)
    tmp = f"_tmp_{start_col}"
    df = df.withColumn(tmp, F.col(start_col))
    df = df.withColumn(start_col, F.when(swapped, F.col(end_col)).otherwise(F.col(start_col)))
    df = df.withColumn(end_col,   F.when(swapped, F.col(tmp)).otherwise(F.col(end_col)))
    return df.drop(tmp)


def _null_neg(df, col, label):
    if not _col_exists(df, col):
        return df
    neg = F.col(col) < 0
    df = _flag(df, neg, label)
    return df.withColumn(col, F.when(neg, F.lit(None)).otherwise(F.col(col)))


def _range_flag(df, col, lo, hi, label):
    if not _col_exists(df, col):
        return df
    out = (F.col(col) < lo) | (F.col(col) > hi)
    return _flag(df, out & F.col(col).isNotNull(), label)


def _add_silver_meta(df, table_name):
    return (df
        .withColumn("_silver_table", F.lit(table_name))
        .withColumn("_cleaned_at",   F.current_timestamp())
        .withColumn("_silver_date",  F.current_date())
        .withColumn("_dq_issues",
            F.when(F.col("_dq_issues") == "", F.lit(None))
             .otherwise(F.col("_dq_issues"))))


def _add_row_hash(df, pk):
    business_cols = [c for c in df.columns if not c.startswith("_")]
    return df.withColumn("_row_hash",
        F.md5(F.concat_ws("|", *[F.coalesce(F.col(c), F.lit("")) for c in business_cols])))


# =============================================================================
#  Per-table cleaners
# =============================================================================

def clean_clinics(df):
    df = _trim_strings(df); df = _init_dq(df)
    for c in ["clinic_name","address","city","state"]:
        if _col_exists(df,c): df = df.withColumn(c, F.initcap(F.col(c)))
    for c in ["clinic_id","npi","tax_id"]:
        if _col_exists(df,c): df = df.withColumn(c, F.upper(F.col(c)))
    df = _cast_int(df,"bed_count")
    if _col_exists(df,"npi"):
        df = _flag(df, ~F.col("npi").rlike(r"^\d{10}$") & F.col("npi").isNotNull(), "invalid_npi")
    df = df.filter(F.col("clinic_id").isNotNull()).dropDuplicates(["clinic_id"])
    return df

def clean_providers(df):
    df = _trim_strings(df); df = _init_dq(df)
    for c in ["first_name","last_name","specialty","city","state"]:
        if _col_exists(df,c): df = df.withColumn(c, F.initcap(F.col(c)))
    for c in ["provider_id","npi"]:
        if _col_exists(df,c): df = df.withColumn(c, F.upper(F.col(c)))
    if _col_exists(df,"email"): df = df.withColumn("email", F.lower(F.col("email")))
    df = _cast_int(df,"years_experience"); df = _cast_bool(df,"board_certified")
    df = _cast_bool(df,"accepting_new_patients")
    if _col_exists(df,"npi"):
        df = _flag(df, ~F.col("npi").rlike(r"^\d{10}$") & F.col("npi").isNotNull(), "invalid_npi")
    if _col_exists(df,"years_experience"):
        df = _flag(df, F.col("years_experience") < 0, "negative_years_experience")
    df = df.filter(F.col("provider_id").isNotNull()).dropDuplicates(["provider_id"])
    return df

def clean_payers(df):
    df = _trim_strings(df); df = _init_dq(df)
    for c in ["payer_name","payer_type"]:
        if _col_exists(df,c): df = df.withColumn(c, F.initcap(F.col(c)))
    if _col_exists(df,"payer_id"): df = df.withColumn("payer_id", F.upper(F.col("payer_id")))
    df = _cast_int(df,"timely_filing_limit_days"); df = _cast_bool(df,"supports_electronic_claims")
    if _col_exists(df,"timely_filing_limit_days"):
        df = _flag(df, F.col("timely_filing_limit_days") < 0, "negative_timely_filing_limit_days")
    df = df.filter(F.col("payer_id").isNotNull()).dropDuplicates(["payer_id"])
    return df

def clean_patients(df):
    df = _trim_strings(df); df = _init_dq(df)
    for c in ["first_name","last_name","city","state","gender"]:
        if _col_exists(df,c): df = df.withColumn(c, F.initcap(F.col(c)))
    if _col_exists(df,"patient_id"): df = df.withColumn("patient_id", F.upper(F.col("patient_id")))
    if _col_exists(df,"email"): df = df.withColumn("email", F.lower(F.col("email")))
    df = _cast_date(df,"dob"); df = _cast_int(df,"age")
    if _col_exists(df,"age"):
        df = _flag(df, (F.col("age") < 0) | (F.col("age") > 120), "age_out_of_range")
    if _col_exists(df,"dob"):
        df = _flag(df, F.col("dob") > F.current_date(), "future_dob")
    if _col_exists(df,"email"):
        df = _flag(df, ~F.col("email").rlike(r"^[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}$") & F.col("email").isNotNull(), "invalid_email")
    if _col_exists(df,"zip_code"):
        df = _flag(df, ~F.col("zip_code").rlike(r"^\d{5}$") & F.col("zip_code").isNotNull(), "invalid_zip_code")
    df = df.filter(F.col("patient_id").isNotNull()).dropDuplicates(["patient_id"])
    return df

def clean_patient_insurance(df):
    df = _trim_strings(df); df = _init_dq(df)
    for c in ["patient_insurance_id","patient_id","payer_id","plan_name","plan_type","member_id"]:
        if _col_exists(df,c): df = df.withColumn(c, F.upper(F.col(c)))
    df = _cast_date(df,"effective_date"); df = _cast_date(df,"termination_date")
    for a in ["copay_office","copay_specialist","copay_er","deductible_individual",
              "deductible_family","deductible_met","out_of_pocket_max","out_of_pocket_met"]:
        df = _cast_double(df,a)
    df = _cast_bool(df,"requires_referral"); df = _cast_bool(df,"requires_prior_auth")
    df = _fix_date_seq(df,"effective_date","termination_date","swapped_effective_termination")
    for a in ["copay_office","copay_specialist","copay_er","deductible_individual","deductible_family"]:
        df = _null_neg(df,a,f"negative_{a}")
    df = df.filter(F.col("patient_insurance_id").isNotNull()).dropDuplicates(["patient_insurance_id"])
    return df

def clean_appointments(df):
    df = _trim_strings(df); df = _init_dq(df)
    for c in ["appointment_id","patient_id","provider_id","clinic_id","appointment_type","appointment_status"]:
        if _col_exists(df,c): df = df.withColumn(c, F.upper(F.col(c)))
    df = _cast_date(df,"appointment_date"); df = _cast_int(df,"duration_minutes")
    if _col_exists(df,"duration_minutes"):
        bad = (F.col("duration_minutes") < 5) | (F.col("duration_minutes") > 480)
        df = _flag(df, bad & F.col("duration_minutes").isNotNull(), "duration_out_of_range")
        df = df.withColumn("duration_minutes", F.when(bad, F.lit(None)).otherwise(F.col("duration_minutes")))
    df = df.filter(F.col("appointment_id").isNotNull()).dropDuplicates(["appointment_id"])
    return df

def clean_encounters(df):
    df = _trim_strings(df); df = _init_dq(df)
    for c in ["encounter_id","patient_id","provider_id","clinic_id","encounter_type","drg_code","place_of_service_code"]:
        if _col_exists(df,c): df = df.withColumn(c, F.upper(F.col(c)))
    df = _cast_date(df,"encounter_date"); df = _cast_date(df,"admit_date"); df = _cast_date(df,"discharge_date")
    df = _cast_int(df,"los_days"); df = _cast_double(df,"drg_base_rate")
    df = _fix_date_seq(df,"admit_date","discharge_date","swapped_admit_discharge")
    if _col_exists(df,"los_days"): df = _flag(df, F.col("los_days") < 0, "negative_los_days")
    df = df.filter(F.col("encounter_id").isNotNull()).dropDuplicates(["encounter_id"])
    return df

def clean_vitals(df):
    df = _trim_strings(df); df = _init_dq(df)
    for c in ["vital_id","patient_id","encounter_id"]:
        if _col_exists(df,c): df = df.withColumn(c, F.upper(F.col(c)))
    for c in ["height_inches","weight_lbs","bmi","systolic_bp","diastolic_bp",
              "heart_rate","respiratory_rate","temperature_f","oxygen_saturation","pain_scale"]:
        df = _cast_double(df,c)
    ranges = {
        "height_inches":(36,96),"weight_lbs":(50,700),"bmi":(10,80),
        "systolic_bp":(60,250),"diastolic_bp":(30,150),"heart_rate":(20,250),
        "temperature_f":(90,110),"oxygen_saturation":(50,100),"pain_scale":(0,10),
    }
    for col,(lo,hi) in ranges.items():
        df = _range_flag(df,col,lo,hi,f"{col}_out_of_range")
    df = df.filter(F.col("vital_id").isNotNull()).dropDuplicates(["vital_id"])
    return df

def clean_diagnoses(df):
    df = _trim_strings(df); df = _init_dq(df)
    for c in ["diagnosis_id","patient_id","encounter_id","icd10_code","diagnosis_type"]:
        if _col_exists(df,c): df = df.withColumn(c, F.upper(F.col(c)))
    if _col_exists(df,"diagnosis_description"):
        df = df.withColumn("diagnosis_description", F.initcap(F.col("diagnosis_description")))
    df = _cast_bool(df,"chronic_flag")
    if _col_exists(df,"icd10_code"):
        df = _flag(df, ~F.col("icd10_code").rlike(r"^[A-Z]\d{2}(\.\d+)?$") & F.col("icd10_code").isNotNull(), "invalid_icd10")
    df = df.filter(F.col("diagnosis_id").isNotNull()).dropDuplicates(["diagnosis_id"])
    return df

def clean_procedures(df):
    df = _trim_strings(df); df = _init_dq(df)
    for c in ["procedure_id","patient_id","encounter_id","provider_id","cpt_code","revenue_code"]:
        if _col_exists(df,c): df = df.withColumn(c, F.upper(F.col(c)))
    df = _cast_int(df,"units"); df = _cast_double(df,"procedure_charge")
    if _col_exists(df,"units"):
        df = _range_flag(df,"units",1,99,"units_out_of_range")
    df = _null_neg(df,"procedure_charge","negative_procedure_charge")
    df = df.filter(F.col("procedure_id").isNotNull()).dropDuplicates(["procedure_id"])
    return df

def clean_lab_results(df):
    df = _trim_strings(df); df = _init_dq(df)
    for c in ["lab_id","patient_id","encounter_id","loinc_code","abnormal_flag","status"]:
        if _col_exists(df,c): df = df.withColumn(c, F.upper(F.col(c)))
    df = _cast_date(df,"order_date"); df = _cast_date(df,"result_date")
    df = _cast_double(df,"result_value"); df = _cast_double(df,"reference_range_low")
    df = _cast_double(df,"reference_range_high")
    df = _fix_date_seq(df,"order_date","result_date","swapped_order_result_dates")
    df = df.filter(F.col("lab_id").isNotNull()).dropDuplicates(["lab_id"])
    return df

def clean_medications(df):
    df = _trim_strings(df); df = _init_dq(df)
    for c in ["medication_id","patient_id","encounter_id","prescriber_id","ndc_code","status"]:
        if _col_exists(df,c): df = df.withColumn(c, F.upper(F.col(c)))
    df = _cast_date(df,"prescribed_date"); df = _cast_date(df,"fill_date")
    df = _cast_int(df,"quantity"); df = _cast_int(df,"days_supply"); df = _cast_int(df,"refills")
    df = _cast_bool(df,"prior_auth_required")
    if _col_exists(df,"quantity"):
        df = _flag(df, (F.col("quantity") <= 0) & F.col("quantity").isNotNull(), "quantity_not_positive")
    df = _fix_date_seq(df,"prescribed_date","fill_date","swapped_prescribed_fill_dates")
    df = df.filter(F.col("medication_id").isNotNull()).dropDuplicates(["medication_id"])
    return df

def clean_prior_authorizations(df):
    df = _trim_strings(df); df = _init_dq(df)
    for c in ["auth_id","patient_id","provider_id","payer_id","service_type","status"]:
        if _col_exists(df,c): df = df.withColumn(c, F.upper(F.col(c)))
    df = _cast_date(df,"request_date"); df = _cast_date(df,"decision_date")
    df = _cast_date(df,"approved_from_date"); df = _cast_date(df,"approved_to_date")
    df = _cast_int(df,"approved_units")
    df = _fix_date_seq(df,"request_date","decision_date","swapped_request_decision")
    df = _fix_date_seq(df,"approved_from_date","approved_to_date","swapped_approved_dates")
    if _col_exists(df,"approved_units"):
        df = _flag(df, (F.col("approved_units") < 0) & F.col("approved_units").isNotNull(), "negative_approved_units")
    df = df.filter(F.col("auth_id").isNotNull()).dropDuplicates(["auth_id"])
    return df

def clean_claims(df):
    df = _trim_strings(df); df = _init_dq(df)
    for c in ["claim_id","patient_id","provider_id","payer_id","encounter_id",
              "claim_type","claim_status","rendering_provider_npi","billing_provider_npi"]:
        if _col_exists(df,c): df = df.withColumn(c, F.upper(F.col(c)))
    df = _cast_date(df,"submission_date"); df = _cast_date(df,"adjudication_date")
    df = _cast_double(df,"total_charge")
    df = _fix_date_seq(df,"submission_date","adjudication_date","swapped_submission_adjudication")
    df = _null_neg(df,"total_charge","negative_total_charge")
    for npi in ["rendering_provider_npi","billing_provider_npi"]:
        if _col_exists(df,npi):
            df = _flag(df, ~F.col(npi).rlike(r"^\d{10}$") & F.col(npi).isNotNull(), f"invalid_{npi}")
    df = df.filter(F.col("claim_id").isNotNull()).dropDuplicates(["claim_id"])
    return df

def clean_claim_lines(df):
    df = _trim_strings(df); df = _init_dq(df)
    for c in ["claim_line_id","claim_id","cpt_code","revenue_code","modifier"]:
        if _col_exists(df,c): df = df.withColumn(c, F.upper(F.col(c)))
    df = _cast_date(df,"service_date"); df = _cast_int(df,"units")
    for a in ["billed_amount","allowed_amount","paid_amount","patient_responsibility",
              "coinsurance_amount","copay_amount","deductible_amount","adjustment_amount"]:
        df = _cast_double(df,a)
    for a in ["billed_amount","allowed_amount","paid_amount"]:
        df = _null_neg(df,a,f"negative_{a}")
    if _col_exists(df,"units"):
        df = _flag(df, (F.col("units") < 1) & F.col("units").isNotNull(), "units_less_than_1")
    df = df.filter(F.col("claim_line_id").isNotNull()).dropDuplicates(["claim_line_id"])
    return df

def clean_remittances(df):
    df = _trim_strings(df); df = _init_dq(df)
    for c in ["remittance_id","payer_id","payment_method","eft_trace_number"]:
        if _col_exists(df,c): df = df.withColumn(c, F.upper(F.col(c)))
    df = _cast_date(df,"payment_date"); df = _cast_double(df,"total_payment_amount")
    df = _cast_int(df,"claim_count")
    df = _null_neg(df,"total_payment_amount","negative_total_payment_amount")
    df = df.filter(F.col("remittance_id").isNotNull()).dropDuplicates(["remittance_id"])
    return df

def clean_payments(df):
    df = _trim_strings(df); df = _init_dq(df)
    for c in ["payment_id","claim_id","remittance_id","payment_source"]:
        if _col_exists(df,c): df = df.withColumn(c, F.upper(F.col(c)))
    df = _cast_date(df,"payment_date"); df = _cast_date(df,"posted_date")
    df = _cast_double(df,"paid_amount")
    df = _null_neg(df,"paid_amount","negative_paid_amount")
    df = _fix_date_seq(df,"payment_date","posted_date","swapped_payment_posted")
    df = df.filter(F.col("payment_id").isNotNull()).dropDuplicates(["payment_id"])
    return df

def clean_denials(df):
    df = _trim_strings(df); df = _init_dq(df)
    for c in ["denial_id","claim_id","denial_code","denial_category","appeal_status"]:
        if _col_exists(df,c): df = df.withColumn(c, F.upper(F.col(c)))
    df = _cast_date(df,"denial_date"); df = _cast_date(df,"appeal_deadline")
    df = _cast_bool(df,"corrected_claim_submitted")
    df = _fix_date_seq(df,"denial_date","appeal_deadline","swapped_denial_appeal_deadline")
    df = df.filter(F.col("denial_id").isNotNull()).dropDuplicates(["denial_id"])
    return df

def clean_appeals(df):
    df = _trim_strings(df); df = _init_dq(df)
    for c in ["appeal_id","denial_id","claim_id","patient_id","appeal_level","decision"]:
        if _col_exists(df,c): df = df.withColumn(c, F.upper(F.col(c)))
    df = _cast_date(df,"submission_date"); df = _cast_date(df,"decision_date")
    df = _fix_date_seq(df,"submission_date","decision_date","swapped_submission_decision")
    df = df.filter(F.col("appeal_id").isNotNull()).dropDuplicates(["appeal_id"])
    return df

def clean_eligibility_checks(df):
    df = _trim_strings(df); df = _init_dq(df)
    for c in ["eligibility_id","patient_id","payer_id","appointment_id","eligibility_status","coverage_type"]:
        if _col_exists(df,c): df = df.withColumn(c, F.upper(F.col(c)))
    df = _cast_date(df,"check_date")
    for a in ["copay","deductible","out_of_pocket_max"]:
        df = _cast_double(df,a); df = _null_neg(df,a,f"negative_{a}")
    df = df.filter(F.col("eligibility_id").isNotNull()).dropDuplicates(["eligibility_id"])
    return df


CLEANERS = {
    "clinics":               clean_clinics,
    "providers":             clean_providers,
    "payers":                clean_payers,
    "patients":              clean_patients,
    "patient_insurance":     clean_patient_insurance,
    "appointments":          clean_appointments,
    "encounters":            clean_encounters,
    "vitals":                clean_vitals,
    "diagnoses":             clean_diagnoses,
    "procedures":            clean_procedures,
    "lab_results":           clean_lab_results,
    "medications":           clean_medications,
    "prior_authorizations":  clean_prior_authorizations,
    "claims":                clean_claims,
    "claim_lines":           clean_claim_lines,
    "remittances":           clean_remittances,
    "payments":              clean_payments,
    "denials":               clean_denials,
    "appeals":               clean_appeals,
    "eligibility_checks":    clean_eligibility_checks,
}


# =============================================================================
#  Watermark filter for INCREMENTAL / WINDOW
# =============================================================================

def _apply_watermark_filter(df, table_name, mode, window_days):
    cfg    = TABLE_CONFIGS.get(table_name)
    wm_col = cfg.watermark_col if cfg else None
    if not wm_col or not _col_exists(df, wm_col):
        return df, datetime.now(timezone.utc).isoformat()
    if mode == WINDOW:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat()
    else:
        cutoff = read_watermark(LAYER, table_name) or "1900-01-01"
    filtered = df.filter(F.col(wm_col).cast("timestamp") > F.lit(cutoff).cast("timestamp"))
    max_row  = filtered.agg(F.max(F.col(wm_col).cast("timestamp")).alias("mx")).collect()
    new_wm   = str(max_row[0]["mx"]) if max_row and max_row[0]["mx"] else cutoff
    return filtered, new_wm


# =============================================================================
#  Per-table pipeline
# =============================================================================

def process_table(spark: SparkSession, table_name: str,
                  mode: str, window_days: int = 7) -> dict:
    cfg         = TABLE_CONFIGS.get(table_name)
    pk          = cfg.primary_key     if cfg else table_name + "_id"
    part_cols   = cfg.partition_cols  if cfg else []
    track_cols  = cfg.scd2_track_cols if cfg else []
    bronze_path = str(BRONZE_DIR / table_name)
    silver_path = str(SILVER_DIR / table_name)

    t0 = time.time()
    stats = {"table": table_name, "mode": mode, "input_rows": 0,
             "output_rows": 0, "dropped_rows": 0, "dq_flagged": 0,
             "watermark": None, "status": "SKIP"}

    # 1. Read bronze — check delta log first to avoid reading broken/empty tables
    bronze_delta_log = Path(bronze_path) / "_delta_log"
    if not bronze_delta_log.exists() or not list(bronze_delta_log.glob("*.json")):
        log.warning("[%s] bronze delta log missing or empty – skipping silver", table_name)
        stats["status"] = "SKIP (bronze not ready)"
        return stats
    try:
        df_bronze = spark.read.format("delta").load(bronze_path)
    except Exception as exc:
        log.error("[%s] cannot read bronze: %s", table_name, exc)
        stats["status"] = f"ERROR: {exc}"
        return stats

    input_rows = df_bronze.count()
    stats["input_rows"] = input_rows

    # 2. Watermark filter
    new_wm = datetime.now(timezone.utc).isoformat()
    if mode in (INCREMENTAL, WINDOW):
        df_bronze, new_wm = _apply_watermark_filter(df_bronze, table_name, mode, window_days)
        if df_bronze.count() == 0:
            stats["status"] = "OK (no new rows)"
            return stats

    # 3. Clean
    cleaner = CLEANERS.get(table_name)
    if not cleaner:
        stats["status"] = "SKIP (no cleaner)"
        return stats
    df_clean = cleaner(df_bronze)

    # 4. Silver metadata + row hash
    df_clean = _add_silver_meta(df_clean, table_name)
    df_clean = _add_row_hash(df_clean, pk)

    # 5. Write — write first, then read counts from Delta to avoid double computation
    write_stats = dispatch_write(
        spark, df_clean, silver_path,
        mode=mode, pk=pk,
        partition_cols=part_cols,
        track_cols=track_cols,
        hash_col="_row_hash",
    )

    # Count DQ flags from the written Delta table to avoid re-scanning source DF
    try:
        written_df = spark.read.format("delta").load(silver_path)
        output_rows = written_df.count()
        dq_flagged  = written_df.filter(F.col("_dq_flag") == True).count()
    except Exception:
        output_rows = write_stats.get("rows_written", 0)
        dq_flagged  = 0

    elapsed = round(time.time() - t0, 2)
    stats.update({
        "output_rows":  output_rows,
        "dropped_rows": input_rows - output_rows,
        "dq_flagged":   dq_flagged,
        "watermark":    new_wm,
        "elapsed_s":    elapsed,
        "status":       "OK",
        **write_stats,
    })

    if mode in (INCREMENTAL, WINDOW):
        write_watermark(LAYER, table_name, new_wm)
    write_audit_log(LAYER, table_name, stats)

    log.info("[%s] ✓ %s  in=%d  out=%d  dq=%d  %.1fs",
             table_name, mode, input_rows, output_rows, dq_flagged, elapsed)
    return stats


# =============================================================================
#  Entry point
# =============================================================================

def run(mode: str = FULL, tables: list = None, window_days: int = 7) -> list:
    mode   = mode.upper()
    tables = tables or ALL_TABLES
    t_start = time.time()
    log.info("SILVER TRANSFORM  mode=%s  tables=%d", mode, len(tables))

    (SILVER_DIR).mkdir(parents=True, exist_ok=True)
    all_stats = []

    for table in tables:
        spark = None
        try:
            spark = get_spark("MedBilling_Silver")
            s = process_table(spark, table, mode=mode, window_days=window_days)
        except Exception as exc:
            log.error("[%s] FAILED: %s", table, exc, exc_info=True)
            s = {"table": table, "mode": mode, "input_rows": 0,
                 "output_rows": 0, "dropped_rows": 0, "dq_flagged": 0,
                 "status": f"ERROR: {exc}"}
        finally:
            if spark is not None:
                try:
                    spark.stop()
                except Exception:
                    pass
        all_stats.append(s)

    print_run_summary("silver", all_stats, time.time() - t_start)
    return all_stats


def _parse_args():
    p = argparse.ArgumentParser(description="Silver transform pipeline")
    p.add_argument("--mode", default=FULL,
                   choices=[FULL, INCREMENTAL, WINDOW, CDC, SCD2, FDC])
    p.add_argument("--tables", default=None)
    p.add_argument("--window-days", type=int, default=7)
    return p.parse_args()


if __name__ == "__main__":
    args   = _parse_args()
    tables = [t.strip() for t in args.tables.split(",")] if args.tables else None
    run(mode=args.mode, tables=tables, window_days=args.window_days)
