"""
dag_daily_incremental.py
========================
DAG: med_billing_daily_incremental

Schedule : Every night at 02:00 (local time)
Strategy :
  bronze  → INCREMENTAL  (append new rows since last watermark)
  silver  → CDC          (merge cleaned rows, upsert by PK)
  gold    → INCREMENTAL  (recompute touched partitions)

Dependency chain:
  bronze_incremental
       ↓
  silver_cdc
       ↓
  gold_incremental

On failure: retry once after 5 minutes, then alert via email callback.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator

import sys
from pathlib import Path

# Make the plugins folder importable inside DAG files
_DAG_DIR    = Path(__file__).resolve().parent
_AIRFLOW_DIR = _DAG_DIR.parent
sys.path.insert(0, str(_AIRFLOW_DIR / "plugins"))

from med_billing_operator import MedBillingPipelineOperator

# ── Default args ──────────────────────────────────────────────────────────────
default_args = {
    "owner":            "data_engineering",
    "depends_on_past":  False,
    "email":            ["alerts@medbilling.local"],
    "email_on_failure": False,   # set True when SMTP is configured
    "email_on_retry":   False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=5),
    "execution_timeout": timedelta(hours=3),
}

# ── DAG ───────────────────────────────────────────────────────────────────────
with DAG(
    dag_id="med_billing_daily_incremental",
    description="Daily incremental load: Bronze INCREMENTAL → Silver CDC → Gold INCREMENTAL",
    default_args=default_args,
    schedule_interval="0 2 * * *",      # 02:00 every night
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["medical_billing", "incremental", "daily"],
    doc_md="""
## Daily Incremental Pipeline

Runs every night at **02:00**.

| Layer  | Mode        | What happens                                      |
|--------|-------------|---------------------------------------------------|
| Bronze | INCREMENTAL | Appends rows newer than last watermark             |
| Silver | CDC         | Merges cleaned rows (upsert by primary key)        |
| Gold   | INCREMENTAL | Recomputes KPIs for partitions touched today       |

**Retry**: 1 retry after 5 minutes on any task failure.
**SLA**: Pipeline must complete within 3 hours.
    """,
) as dag:

    start = EmptyOperator(task_id="pipeline_start")
    end   = EmptyOperator(task_id="pipeline_end")

    # ── Bronze ────────────────────────────────────────────────────────────────
    bronze = MedBillingPipelineOperator(
        task_id="bronze_incremental",
        layers=["bronze"],
        mode="INCREMENTAL",
        doc_md="Append new raw rows (clean + dirty sources) since last watermark.",
    )

    # ── Silver ────────────────────────────────────────────────────────────────
    silver = MedBillingPipelineOperator(
        task_id="silver_cdc",
        layers=["silver"],
        mode="CDC",
        doc_md="Merge cleaned/validated rows into silver Delta tables (upsert by PK).",
    )

    # ── Gold ──────────────────────────────────────────────────────────────────
    gold = MedBillingPipelineOperator(
        task_id="gold_incremental",
        layers=["gold"],
        mode="INCREMENTAL",
        doc_md="Recompute dimensions, fact_claims, and all KPI tables.",
    )

    # ── Dependencies ──────────────────────────────────────────────────────────
    start >> bronze >> silver >> gold >> end
