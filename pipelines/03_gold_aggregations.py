"""
03_gold_aggregations.py
=======================
Gold Layer — Dimensions, Facts & KPIs (Production-Grade)

Supported load modes
--------------------
FULL        – Recompute and overwrite every gold table.
INCREMENTAL – Recompute only the partitions touched since last watermark.
WINDOW      – Recompute a rolling N-day window.
CDC         – Merge updated rows into gold using primary key.
FDC         – Full-Diff-Compare: only write changed rows.

Gold tables produced
--------------------
Dimensions : dim_patients, dim_providers, dim_clinics, dim_payers
Fact       : fact_claims
KPIs       : kpi_revenue_summary, kpi_denial_analysis, kpi_ar_aging,
             kpi_provider_scorecard, kpi_payer_mix, kpi_procedure_revenue,
             kpi_diagnosis_burden, kpi_appointment_ops, kpi_patient_risk,
             kpi_lab_abnormals

Usage (CLI)
-----------
    python pipelines/03_gold_aggregations.py [--mode FULL|INCREMENTAL|WINDOW|CDC|FDC]
                                             [--tables dim_patients,fact_claims,...]
                                             [--window-days 7]
"""

import argparse
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "pipelines"))

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from pipeline_config import (
    SILVER_DIR, GOLD_DIR, ALL_TABLES,
    FULL, INCREMENTAL, WINDOW, CDC, FDC,
)
from pipeline_utils import (
    get_spark, read_watermark, write_watermark, write_audit_log,
    dispatch_write, print_run_summary, log,
)

LAYER = "gold"


# =============================================================================
#  Helpers
# =============================================================================

def _silver(spark: SparkSession, table: str) -> DataFrame | None:
    try:
        return spark.read.format("delta").load(f"{SILVER_DIR}/{table}")
    except Exception as exc:
        log.warning("Cannot read silver.%s: %s", table, exc)
        return None


def _gold(spark: SparkSession, table: str) -> DataFrame | None:
    try:
        return spark.read.format("delta").load(f"{GOLD_DIR}/{table}")
    except Exception as exc:
        log.warning("Cannot read gold.%s: %s", table, exc)
        return None


def _meta(df: DataFrame, table_name: str) -> DataFrame:
    return (df
        .withColumn("_gold_table",  F.lit(table_name))
        .withColumn("_computed_at", F.current_timestamp())
        .withColumn("_gold_date",   F.current_date()))


def _safe_pct(num, den):
    return F.round(
        F.when(F.coalesce(den, F.lit(0)) != 0,
               F.coalesce(num, F.lit(0)) / den * 100)
         .otherwise(F.lit(None).cast("double")), 2)


def _drop_meta(df: DataFrame) -> DataFrame:
    drop = {"_bronze_table","_ingested_at","_ingestion_date","_source",
            "_silver_table","_cleaned_at","_silver_date","_dq_flag","_dq_issues",
            "_row_hash","_gold_table","_computed_at","_gold_date",
            "_effective_from","_effective_to","_is_current","_is_deleted"}
    return df.select([c for c in df.columns if c not in drop])


# =============================================================================
#  DIMENSION BUILDERS
# =============================================================================

def build_dim_patients(spark):
    patients = _silver(spark, "patients")
    if patients is None:
        return spark.createDataFrame([], "patient_id string")
    ins = _silver(spark, "patient_insurance")
    if ins is not None:
        primary = (ins.filter(F.upper(F.col("insurance_priority")) == "PRIMARY")
                      .select(F.col("patient_id").alias("_pi"),
                              F.col("payer_id").alias("primary_payer_id"),
                              F.col("plan_type").alias("primary_plan_type")))
        df = patients.join(primary, patients["patient_id"] == primary["_pi"], "left").drop("_pi")
    else:
        df = patients.withColumn("primary_payer_id", F.lit(None).cast("string")) \
                     .withColumn("primary_plan_type", F.lit(None).cast("string"))
    keep = ["patient_id","mrn","first_name","last_name","dob","age","gender",
            "race","ethnicity","marital_status","preferred_language","employment_status",
            "income_bracket","smoking_status","bmi_category","city","state","zip_code",
            "primary_payer_id","primary_plan_type","created_at"]
    df = df.select([c for c in keep if c in df.columns])
    return _drop_meta(df).dropDuplicates(["patient_id"])


def build_dim_providers(spark):
    providers = _silver(spark, "providers")
    if providers is None:
        return spark.createDataFrame([], "provider_id string")
    clinics = _silver(spark, "clinics")
    if clinics is not None:
        cl = clinics.select(F.col("clinic_id").alias("_cl"),
                            F.col("clinic_name"), F.col("clinic_type"))
        df = providers.join(cl, providers["clinic_id"] == cl["_cl"], "left").drop("_cl")
    else:
        df = providers.withColumn("clinic_name", F.lit(None).cast("string")) \
                      .withColumn("clinic_type", F.lit(None).cast("string"))
    keep = ["provider_id","npi","first_name","last_name","credential","specialty",
            "sub_specialty","taxonomy_code","clinic_id","clinic_name","clinic_type",
            "years_experience","board_certified","accepting_new_patients","status","created_at"]
    df = df.select([c for c in keep if c in df.columns])
    return _drop_meta(df).dropDuplicates(["provider_id"])


def build_dim_clinics(spark):
    df = _silver(spark, "clinics")
    if df is None:
        return spark.createDataFrame([], "clinic_id string")
    return _drop_meta(df).dropDuplicates(["clinic_id"])


def build_dim_payers(spark):
    df = _silver(spark, "payers")
    if df is None:
        return spark.createDataFrame([], "payer_id string")
    return _drop_meta(df).dropDuplicates(["payer_id"])


# =============================================================================
#  FACT BUILDER
# =============================================================================

def build_fact_claims(spark):
    claims = _silver(spark, "claims")
    if claims is None:
        return spark.createDataFrame([], "claim_id string")

    # Claim lines aggregation
    cl = _silver(spark, "claim_lines")
    if cl is not None:
        lines_agg = cl.groupBy("claim_id").agg(
            F.sum("billed_amount").alias("total_billed"),
            F.sum("allowed_amount").alias("total_allowed"),
            F.sum("paid_amount").alias("total_paid"),
            F.sum("patient_responsibility").alias("total_patient_resp"),
            F.count("*").alias("line_count"),
        )
        claims = claims.join(lines_agg, on="claim_id", how="left")
    else:
        for c,t in [("total_billed","double"),("total_allowed","double"),
                    ("total_paid","double"),("total_patient_resp","double"),("line_count","long")]:
            claims = claims.withColumn(c, F.lit(None).cast(t))

    # Encounters join
    enc = _silver(spark, "encounters")
    if enc is not None:
        e = enc.select(F.col("encounter_id").alias("_eid"),
                       "encounter_date","place_of_service_code","encounter_type")
        claims = claims.join(e, claims["encounter_id"] == e["_eid"], "left").drop("_eid")
    else:
        for c,t in [("encounter_date","date"),("place_of_service_code","string"),("encounter_type","string")]:
            claims = claims.withColumn(c, F.lit(None).cast(t))

    # Payments aggregation
    pay = _silver(spark, "payments")
    if pay is not None:
        pay_agg = pay.groupBy("claim_id").agg(
            F.sum("paid_amount").alias("payment_received"),
            F.max("payment_date").alias("last_payment_date"),
        )
        claims = claims.join(pay_agg, on="claim_id", how="left")
    else:
        claims = claims.withColumn("payment_received", F.lit(None).cast("double")) \
                       .withColumn("last_payment_date", F.lit(None).cast("date"))

    # Computed columns
    claims = (claims
        .withColumn("days_to_adjudication",
                    F.datediff(F.col("adjudication_date"), F.col("submission_date")))
        .withColumn("is_paid",   F.upper(F.col("claim_status")) == "PAID")
        .withColumn("is_denied", F.upper(F.col("claim_status")) == "DENIED")
        .withColumn("collection_rate",
                    F.when(F.col("total_charge") > 0,
                           F.round(F.col("payment_received") / F.col("total_charge") * 100, 2))
                     .otherwise(F.lit(None).cast("double"))))

    keep = ["claim_id","encounter_id","patient_id","provider_id","clinic_id",
            "payer_id","secondary_payer_id","claim_number","claim_type","claim_status",
            "submission_date","adjudication_date","total_charge","total_billed",
            "total_allowed","total_paid","total_patient_resp","payment_received",
            "last_payment_date","line_count","encounter_date","place_of_service_code",
            "encounter_type","filing_indicator","days_to_adjudication",
            "is_paid","is_denied","collection_rate"]
    return claims.select([c for c in keep if c in claims.columns])


# =============================================================================
#  KPI BUILDERS
# =============================================================================

def build_kpi_revenue_summary(spark):
    fact = _gold(spark, "fact_claims")
    if fact is None:
        return spark.createDataFrame([], "year int, month int")
    return (fact
        .groupBy(F.year("submission_date").alias("year"),
                 F.month("submission_date").alias("month"))
        .agg(
            F.count("*").alias("total_claims"),
            F.round(F.sum("total_charge"),2).alias("total_charge"),
            F.round(F.sum("total_allowed"),2).alias("total_allowed"),
            F.round(F.sum("total_paid"),2).alias("total_paid"),
            F.round(F.sum("total_patient_resp"),2).alias("total_patient_resp"),
            F.round(F.avg("days_to_adjudication"),1).alias("avg_days_to_adjudication"),
            _safe_pct(F.sum("total_paid"), F.sum("total_charge")).alias("collection_rate_pct"),
            F.count(F.when(F.col("is_denied")==True,1)).alias("denied_claims"),
            _safe_pct(F.count(F.when(F.col("is_denied")==True,1)),
                      F.count("*")).alias("denial_rate_pct"),
        )
        .orderBy("year","month"))


def build_kpi_denial_analysis(spark):
    denials = _silver(spark, "denials")
    if denials is None:
        return spark.createDataFrame([], "denial_code string")
    claims = _silver(spark, "claims")
    if claims is not None:
        c = claims.select(F.col("claim_id").alias("_cid"),
                          F.col("total_charge").alias("claim_charge"))
        df = denials.join(c, denials["claim_id"] == c["_cid"], "left").drop("_cid")
    else:
        df = denials.withColumn("claim_charge", F.lit(None).cast("double"))
    return (df
        .groupBy(F.upper(F.col("denial_code")).alias("denial_code"),
                 F.col("denial_category"), F.col("denial_reason"))
        .agg(
            F.count("*").alias("denial_count"),
            F.round(F.sum("claim_charge"),2).alias("total_denied_amount"),
            F.count(F.when(F.col("appeal_status").isNotNull(),1)).alias("appeal_count"),
            _safe_pct(F.count(F.when(F.upper(F.col("appeal_status"))=="APPROVED",1)),
                      F.count(F.when(F.col("appeal_status").isNotNull(),1))).alias("overturn_rate_pct"),
        )
        .orderBy(F.col("denial_count").desc()))


def build_kpi_ar_aging(spark):
    claims = _silver(spark, "claims")
    if claims is None:
        return spark.createDataFrame([], "aging_bucket string")
    df = claims.filter(~F.upper(F.col("claim_status")).isin("PAID","VOID","ADJUSTED"))
    df = df.withColumn("days_outstanding",
                       F.datediff(F.current_date(), F.col("submission_date")))
    df = df.withColumn("aging_bucket",
        F.when(F.col("days_outstanding") <= 30,  "01_0-30 days")
         .when(F.col("days_outstanding") <= 60,  "02_31-60 days")
         .when(F.col("days_outstanding") <= 90,  "03_61-90 days")
         .when(F.col("days_outstanding") <= 120, "04_91-120 days")
         .otherwise("05_120+ days"))
    return (df
        .groupBy("aging_bucket","payer_id")
        .agg(
            F.count("*").alias("claim_count"),
            F.round(F.sum("total_charge"),2).alias("total_outstanding"),
            F.round(F.avg("days_outstanding"),1).alias("avg_days_outstanding"),
            F.round(F.max("days_outstanding"),0).alias("max_days_outstanding"),
        )
        .orderBy("aging_bucket"))


def build_kpi_provider_scorecard(spark):
    fact = _gold(spark, "fact_claims")
    if fact is None:
        return spark.createDataFrame([], "provider_id string")
    providers = _silver(spark, "providers")
    if providers is not None:
        p = providers.select(F.col("provider_id").alias("_pid"),
                             "first_name","last_name","specialty","credential")
        df = fact.join(p, fact["provider_id"] == p["_pid"], "left").drop("_pid")
    else:
        df = fact
    return (df
        .groupBy("provider_id","first_name","last_name","specialty","credential")
        .agg(
            F.count("*").alias("total_claims"),
            F.round(F.sum("total_paid"),2).alias("total_revenue"),
            F.round(F.avg("total_charge"),2).alias("avg_charge_per_claim"),
            F.round(F.avg("total_paid"),2).alias("avg_paid_per_claim"),
            _safe_pct(F.count(F.when(F.col("is_denied")==True,1)),
                      F.count("*")).alias("denial_rate_pct"),
            F.round(F.avg("days_to_adjudication"),1).alias("avg_days_to_adjudication"),
            F.count(F.when(F.col("is_paid")==True,1)).alias("paid_claims"),
            _safe_pct(F.count(F.when(F.col("is_paid")==True,1)),
                      F.count("*")).alias("paid_rate_pct"),
        )
        .orderBy(F.col("total_revenue").desc()))


def build_kpi_payer_mix(spark):
    fact = _gold(spark, "fact_claims")
    if fact is None:
        return spark.createDataFrame([], "payer_id string")
    payers = _silver(spark, "payers")
    if payers is not None:
        p = payers.select(F.col("payer_id").alias("_pid"),
                          "payer_name","payer_type")
        df = fact.join(p, fact["payer_id"] == p["_pid"], "left").drop("_pid")
    else:
        df = fact
    agg = (df
        .groupBy("payer_id","payer_name","payer_type")
        .agg(
            F.count("*").alias("claim_count"),
            F.round(F.sum("total_charge"),2).alias("total_billed"),
            F.round(F.sum("total_paid"),2).alias("total_paid"),
            _safe_pct(F.avg("total_allowed"), F.avg("total_charge")).alias("avg_allowed_pct"),
            F.round(F.avg("days_to_adjudication"),1).alias("avg_days_to_adjudication"),
            F.count(F.when(F.col("is_denied")==True,1)).alias("denial_count"),
            _safe_pct(F.count(F.when(F.col("is_denied")==True,1)),
                      F.count("*")).alias("denial_rate_pct"),
        ))
    win = Window.orderBy(F.lit(1))
    agg = agg.withColumn("pct_of_total_revenue",
        F.round(F.col("total_paid") / F.sum("total_paid").over(win) * 100, 2))
    return agg.orderBy(F.col("total_paid").desc())


def build_kpi_procedure_revenue(spark):
    cl = _silver(spark, "claim_lines")
    if cl is None:
        return spark.createDataFrame([], "cpt_code string")
    procs = _silver(spark, "procedures")
    if procs is not None:
        p = procs.select(F.col("cpt_code").alias("_cpt"),
                         "procedure_description").dropDuplicates(["_cpt"])
        df = cl.join(p, cl["cpt_code"] == p["_cpt"], "left").drop("_cpt")
    else:
        df = cl.withColumn("procedure_description", F.lit(None).cast("string"))
    return (df
        .groupBy("cpt_code","procedure_description")
        .agg(
            F.count("*").alias("utilization_count"),
            F.round(F.sum("billed_amount"),2).alias("total_billed"),
            F.round(F.sum("paid_amount"),2).alias("total_paid"),
            F.round(F.avg("billed_amount"),2).alias("avg_billed"),
            F.round(F.avg("paid_amount"),2).alias("avg_paid"),
            _safe_pct(F.avg("paid_amount"), F.avg("billed_amount")).alias("avg_reimbursement_rate_pct"),
        )
        .orderBy(F.col("total_paid").desc()))


def build_kpi_diagnosis_burden(spark):
    diag = _silver(spark, "diagnoses")
    if diag is None:
        return spark.createDataFrame([], "icd10_code string")
    pid_col = "patient_id" if "patient_id" in diag.columns else None
    if pid_col is None:
        enc = _silver(spark, "encounters")
        if enc is not None:
            e = enc.select(F.col("encounter_id").alias("_eid"),
                           F.col("patient_id").alias("enc_pid"))
            diag = diag.join(e, diag["encounter_id"] == e["_eid"], "left").drop("_eid")
            diag = diag.withColumn("patient_id", F.col("enc_pid")).drop("enc_pid")
    return (diag
        .groupBy("icd10_code","diagnosis_description","chronic_flag")
        .agg(
            F.countDistinct("patient_id").alias("patient_count"),
            F.count("*").alias("encounter_count"),
            F.count(F.when(F.upper(F.col("diagnosis_type"))=="PRIMARY",1)).alias("primary_dx_count"),
            F.round(F.count("*") / F.countDistinct("patient_id"), 2).alias("avg_encounters_per_patient"),
        )
        .orderBy(F.col("patient_count").desc()))


def build_kpi_appointment_ops(spark):
    df = _silver(spark, "appointments")
    if df is None:
        return spark.createDataFrame([], "year int, month int")
    return (df
        .groupBy(F.year("appointment_date").alias("year"),
                 F.month("appointment_date").alias("month"),
                 "appointment_type")
        .agg(
            F.count("*").alias("total_scheduled"),
            F.count(F.when(F.upper(F.col("appointment_status"))=="COMPLETED",1)).alias("completed"),
            F.count(F.when(F.upper(F.col("appointment_status"))=="CANCELLED",1)).alias("cancelled"),
            F.count(F.when(F.upper(F.col("appointment_status"))=="NO SHOW",1)).alias("no_show"),
            _safe_pct(F.count(F.when(F.upper(F.col("appointment_status"))=="COMPLETED",1)),
                      F.count("*")).alias("completion_rate_pct"),
            _safe_pct(F.count(F.when(F.upper(F.col("appointment_status"))=="NO SHOW",1)),
                      F.count("*")).alias("no_show_rate_pct"),
            F.round(F.avg("duration_minutes"),1).alias("avg_duration_minutes"),
        )
        .orderBy("year","month"))


def build_kpi_patient_risk(spark):
    patients = _silver(spark, "patients")
    if patients is None:
        return spark.createDataFrame([], "patient_id string")
    diag = _silver(spark, "diagnoses")
    if diag is not None and "patient_id" in diag.columns:
        d = diag.select("patient_id","icd10_code","chronic_flag")
        df = patients.join(d, on="patient_id", how="left")
    else:
        df = patients.withColumn("icd10_code", F.lit(None).cast("string")) \
                     .withColumn("chronic_flag", F.lit(None).cast("boolean"))
    grp = [c for c in ["patient_id","first_name","last_name","age","gender",
                        "bmi_category","smoking_status"] if c in df.columns]
    agg = (df.groupBy(*grp).agg(
        F.count(F.when(F.col("chronic_flag")==True,1)).alias("chronic_condition_count"),
        F.count("*").alias("total_diagnoses"),
        F.countDistinct("icd10_code").alias("distinct_icd10_codes"),
    ))
    agg = agg.withColumn("risk_score",
        F.least(F.col("chronic_condition_count")*2 + F.col("total_diagnoses"), F.lit(100)))
    agg = agg.withColumn("risk_tier",
        F.when(F.col("risk_score")>=20,"High")
         .when(F.col("risk_score")>=10,"Medium")
         .otherwise("Low"))
    return agg.orderBy(F.col("risk_score").desc())


def build_kpi_lab_abnormals(spark):
    df = _silver(spark, "lab_results")
    if df is None:
        return spark.createDataFrame([], "test_name string")
    df = df.filter(F.col("abnormal_flag").isNotNull() &
                   (F.upper(F.col("abnormal_flag")) != "NORMAL"))
    win = Window.partitionBy("test_name")
    df = df.withColumn("total_tests_for_name", F.count("*").over(win))
    return (df
        .groupBy("test_name","loinc_code","abnormal_flag")
        .agg(
            F.count("*").alias("abnormal_count"),
            F.first("total_tests_for_name").alias("total_tests"),
            _safe_pct(F.count("*"), F.first("total_tests_for_name")).alias("abnormal_rate_pct"),
            F.round(F.avg("result_value"),3).alias("avg_result_value"),
            F.round(F.min("result_value"),3).alias("min_result_value"),
            F.round(F.max("result_value"),3).alias("max_result_value"),
        )
        .orderBy(F.col("abnormal_count").desc()))


# =============================================================================
#  Dispatch table  (order matters: fact_claims before KPIs that read it)
# =============================================================================

GOLD_TABLES = OrderedDict([
    ("dim_patients",           build_dim_patients),
    ("dim_providers",          build_dim_providers),
    ("dim_clinics",            build_dim_clinics),
    ("dim_payers",             build_dim_payers),
    ("fact_claims",            build_fact_claims),
    ("kpi_revenue_summary",    build_kpi_revenue_summary),
    ("kpi_denial_analysis",    build_kpi_denial_analysis),
    ("kpi_ar_aging",           build_kpi_ar_aging),
    ("kpi_provider_scorecard", build_kpi_provider_scorecard),
    ("kpi_payer_mix",          build_kpi_payer_mix),
    ("kpi_procedure_revenue",  build_kpi_procedure_revenue),
    ("kpi_diagnosis_burden",   build_kpi_diagnosis_burden),
    ("kpi_appointment_ops",    build_kpi_appointment_ops),
    ("kpi_patient_risk",       build_kpi_patient_risk),
    ("kpi_lab_abnormals",      build_kpi_lab_abnormals),
])

ALL_GOLD_TABLES = list(GOLD_TABLES.keys())


# =============================================================================
#  Per-table pipeline
# =============================================================================

def process_gold_table(spark: SparkSession, table_name: str,
                       builder_fn, mode: str, window_days: int = 7) -> dict:
    target_path = f"{GOLD_DIR}/{table_name}"
    t0 = time.time()
    stats = {"table": table_name, "mode": mode, "input_rows": 0,
             "output_rows": 0, "dropped_rows": 0, "dq_flagged": 0,
             "watermark": None, "status": "SKIP"}
    try:
        df = builder_fn(spark)
        if df is None:
            stats["status"] = "OK (empty)"
            return stats

        df = _meta(df, table_name)

        # Write directly to disk — do NOT call rdd.isEmpty() or df.count() before writing
        # as that materialises the entire DataFrame in memory
        write_stats = dispatch_write(
            spark, df, target_path,
            mode=mode,
            pk="gold_row_id",   # gold tables use FULL/INCREMENTAL only; pk unused for those modes
            partition_cols=[],
        )

        # Read row count from the written Delta table
        try:
            row_count = spark.read.format("delta").load(target_path).count()
        except Exception:
            row_count = write_stats.get("rows_written", 0)

        elapsed = round(time.time() - t0, 2)
        stats.update({"output_rows": row_count, "elapsed_s": elapsed,
                      "status": "OK", **write_stats})
        write_audit_log(LAYER, table_name, stats)
        log.info("[%s] ✓ %s  rows=%d  %.1fs", table_name, mode, row_count, elapsed)
    except Exception as exc:
        log.error("[%s] FAILED: %s", table_name, exc, exc_info=True)
        stats["status"] = f"ERROR: {exc}"
    return stats


# =============================================================================
#  Entry point
# =============================================================================

def run(mode: str = FULL, tables: list = None, window_days: int = 7) -> list:
    mode   = mode.upper()
    tables = tables or ALL_GOLD_TABLES
    t_start = time.time()
    log.info("GOLD AGGREGATIONS  mode=%s  tables=%d", mode, len(tables))

    Path(GOLD_DIR).mkdir(parents=True, exist_ok=True)
    all_stats = []

    for table in tables:
        builder = GOLD_TABLES.get(table)
        if builder is None:
            log.warning("Unknown gold table: %s", table)
            continue
        spark = None
        try:
            spark = get_spark("MedBilling_Gold")
            s = process_gold_table(spark, table, builder, mode=mode, window_days=window_days)
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

    print_run_summary("gold", all_stats, time.time() - t_start)
    return all_stats


def _parse_args():
    p = argparse.ArgumentParser(description="Gold aggregations pipeline")
    p.add_argument("--mode", default=FULL,
                   choices=[FULL, INCREMENTAL, WINDOW, CDC, FDC])
    p.add_argument("--tables", default=None)
    p.add_argument("--window-days", type=int, default=7)
    return p.parse_args()


if __name__ == "__main__":
    args   = _parse_args()
    tables = [t.strip() for t in args.tables.split(",")] if args.tables else None
    run(mode=args.mode, tables=tables, window_days=args.window_days)
