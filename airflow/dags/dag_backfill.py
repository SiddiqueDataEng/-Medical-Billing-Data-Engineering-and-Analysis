"""
dag_backfill.py
===============
DAG: med_billing_backfill

Schedule : Manual trigger only (no automatic schedule)
Strategy : WINDOW mode — re-process a configurable rolling window

Use cases:
  - Re-process data after a bug fix in cleaning logic
  - Reload a specific date range after source data correction
  - Recover from a failed incremental run

Trigger with Airflow UI "Trigger DAG w/ config":
  {
    "window_days": 30,
    "layers": "bronze,silver,gold",
    "mode": "WINDOW",
    "tables": ""
  }

Or via CLI:
  airflow dags trigger med_billing_backfill \
    --conf '{"window_days": 30, "layers": "bronze,silver,gold"}'
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator

import sys
from pathlib import Path

_DAG_DIR     = Path(__file__).resolve().parent
_AIRFLOW_DIR = _DAG_DIR.parent
sys.path.insert(0, str(_AIRFLOW_DIR / "plugins"))

from med_billing_operator import MedBillingPipelineOperator

default_args = {
    "owner":             "data_engineering",
    "depends_on_past":   False,
    "email_on_failure":  False,
    "retries":           0,
    "execution_timeout": timedelta(hours=8),
}


def _parse_conf(**context):
    """Extract and validate DAG run config, push to XCom."""
    conf = context["dag_run"].conf or {}
    window_days = int(conf.get("window_days", 7))
    layers      = conf.get("layers", "bronze,silver,gold").split(",")
    mode        = conf.get("mode", "WINDOW").upper()
    tables_raw  = conf.get("tables", "")
    tables      = [t.strip() for t in tables_raw.split(",") if t.strip()] or None

    context["ti"].xcom_push(key="window_days", value=window_days)
    context["ti"].xcom_push(key="layers",      value=layers)
    context["ti"].xcom_push(key="mode",        value=mode)
    context["ti"].xcom_push(key="tables",      value=tables)

    print(f"Backfill config:")
    print(f"  window_days : {window_days}")
    print(f"  layers      : {layers}")
    print(f"  mode        : {mode}")
    print(f"  tables      : {tables or 'ALL'}")


with DAG(
    dag_id="med_billing_backfill",
    description="Manual backfill: re-process a rolling window (WINDOW mode)",
    default_args=default_args,
    schedule_interval=None,             # manual trigger only
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["medical_billing", "backfill", "manual"],
    params={
        "window_days": 7,
        "layers":      "bronze,silver,gold",
        "mode":        "WINDOW",
        "tables":      "",
    },
    doc_md="""
## Backfill Pipeline

**Manual trigger only** — no automatic schedule.

Trigger via UI with optional config:
```json
{
  "window_days": 30,
  "layers": "bronze,silver,gold",
  "mode": "WINDOW",
  "tables": "claims,claim_lines"
}
```

| Parameter    | Default              | Description                          |
|--------------|----------------------|--------------------------------------|
| window_days  | 7                    | How many days back to re-process     |
| layers       | bronze,silver,gold   | Which layers to run                  |
| mode         | WINDOW               | Load mode (WINDOW, FULL, FDC, etc.)  |
| tables       | (empty = all)        | Comma-separated table names          |
    """,
) as dag:

    start = EmptyOperator(task_id="backfill_start")
    end   = EmptyOperator(task_id="backfill_end")

    parse_config = PythonOperator(
        task_id="parse_config",
        python_callable=_parse_conf,
    )

    bronze_window = MedBillingPipelineOperator(
        task_id="bronze_window",
        layers=["bronze"],
        mode="WINDOW",
        window_days="{{ ti.xcom_pull(task_ids='parse_config', key='window_days') | int }}",
        doc_md="Re-ingest bronze for the configured window.",
    )

    silver_window = MedBillingPipelineOperator(
        task_id="silver_window",
        layers=["silver"],
        mode="WINDOW",
        window_days="{{ ti.xcom_pull(task_ids='parse_config', key='window_days') | int }}",
        doc_md="Re-clean silver for the configured window.",
    )

    gold_window = MedBillingPipelineOperator(
        task_id="gold_window",
        layers=["gold"],
        mode="FULL",
        doc_md="Recompute gold (always FULL to ensure consistency after backfill).",
    )

    start >> parse_config >> bronze_window >> silver_window >> gold_window >> end
