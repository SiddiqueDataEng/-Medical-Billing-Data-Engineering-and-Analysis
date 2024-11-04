"""
dag_weekly_full_reload.py
=========================
DAG: med_billing_weekly_full_reload

Schedule : Every Sunday at 01:00
Strategy :
  bronze  → FULL   (truncate + reload all 20 tables from both raw sources)
  silver  → FULL   (re-clean everything, overwrite silver)
  gold    → FULL   (recompute all dims, fact, and KPIs from scratch)

Use case:
  - Catch any rows missed by incremental loads
  - Apply schema changes or new cleaning rules retroactively
  - Rebuild gold KPIs with corrected historical data

Dependency chain:
  bronze_full
       ↓
  silver_full
       ↓
  gold_dims          gold_fact
       ↓                  ↓
       └──── gold_kpis ───┘
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.empty import EmptyOperator

import sys
from pathlib import Path

_DAG_DIR     = Path(__file__).resolve().parent
_AIRFLOW_DIR = _DAG_DIR.parent
sys.path.insert(0, str(_AIRFLOW_DIR / "plugins"))

from med_billing_operator import MedBillingPipelineOperator

default_args = {
    "owner":             "data_engineering",
    "depends_on_past":   False,
    "email":             ["alerts@medbilling.local"],
    "email_on_failure":  False,
    "email_on_retry":    False,
    "retries":           1,
    "retry_delay":       timedelta(minutes=10),
    "execution_timeout": timedelta(hours=6),
}

with DAG(
    dag_id="med_billing_weekly_full_reload",
    description="Weekly full reload: Bronze FULL → Silver FULL → Gold FULL",
    default_args=default_args,
    schedule_interval="0 1 * * 0",     # 01:00 every Sunday
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["medical_billing", "full_reload", "weekly"],
    doc_md="""
## Weekly Full Reload Pipeline

Runs every **Sunday at 01:00**.

| Layer  | Mode | What happens                                          |
|--------|------|-------------------------------------------------------|
| Bronze | FULL | Truncate + reload all 20 tables from both raw sources |
| Silver | FULL | Re-clean everything, overwrite silver Delta tables    |
| Gold   | FULL | Recompute all dims, fact_claims, and 10 KPI tables    |

**Retry**: 1 retry after 10 minutes.
**SLA**: Must complete within 6 hours.
    """,
) as dag:

    start = EmptyOperator(task_id="pipeline_start")
    end   = EmptyOperator(task_id="pipeline_end")

    # ── Bronze: full reload of all 20 tables ──────────────────────────────────
    bronze_full = MedBillingPipelineOperator(
        task_id="bronze_full",
        layers=["bronze"],
        mode="FULL",
        doc_md="Full truncate-and-reload of all 20 bronze tables.",
    )

    # ── Silver: full re-clean ─────────────────────────────────────────────────
    silver_full = MedBillingPipelineOperator(
        task_id="silver_full",
        layers=["silver"],
        mode="FULL",
        doc_md="Full re-clean of all 20 silver tables with DQ flagging.",
    )

    # ── Gold: split into reference tables, fact, then KPIs ───────────────────
    gold_dims = MedBillingPipelineOperator(
        task_id="gold_dimensions",
        layers=["gold"],
        mode="FULL",
        tables=["dim_patients", "dim_providers", "dim_clinics", "dim_payers"],
        doc_md="Rebuild all four dimension tables.",
    )

    gold_fact = MedBillingPipelineOperator(
        task_id="gold_fact_claims",
        layers=["gold"],
        mode="FULL",
        tables=["fact_claims"],
        doc_md="Rebuild fact_claims (joins claims + lines + encounters + payments).",
    )

    gold_kpis = MedBillingPipelineOperator(
        task_id="gold_kpis",
        layers=["gold"],
        mode="FULL",
        tables=[
            "kpi_revenue_summary",
            "kpi_denial_analysis",
            "kpi_ar_aging",
            "kpi_provider_scorecard",
            "kpi_payer_mix",
            "kpi_procedure_revenue",
            "kpi_diagnosis_burden",
            "kpi_appointment_ops",
            "kpi_patient_risk",
            "kpi_lab_abnormals",
        ],
        doc_md="Recompute all 10 KPI aggregation tables.",
    )

    # ── Dependencies ──────────────────────────────────────────────────────────
    start >> bronze_full >> silver_full >> [gold_dims, gold_fact] >> gold_kpis >> end
