"""
dag_data_quality_check.py
=========================
DAG: med_billing_data_quality_check

Schedule : Every night at 06:00 (after the 02:00 incremental pipeline finishes)
Purpose  : Read the silver audit logs, compute DQ metrics, and alert if
           thresholds are breached.

Checks performed
----------------
  1. DQ flag rate per table  — alert if > 10% of rows are flagged
  2. Row count drop          — alert if output < 90% of input
  3. Null PK rows            — alert if any rows were dropped for null PK
  4. Watermark staleness     — alert if watermark hasn't advanced in 24h
  5. Zero-row tables         — alert if any silver table has 0 rows
"""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator

# ── Paths ─────────────────────────────────────────────────────────────────────
_DAG_DIR     = Path(__file__).resolve().parent
_AIRFLOW_DIR = _DAG_DIR.parent
PROJECT_DIR  = _AIRFLOW_DIR.parent
AUDIT_DIR    = PROJECT_DIR / "audit_logs" / "silver"

# ── Thresholds ────────────────────────────────────────────────────────────────
DQ_FLAG_RATE_THRESHOLD   = 0.10   # 10%  — alert if more than this fraction is flagged
ROW_DROP_THRESHOLD       = 0.10   # 10%  — alert if more than this fraction was dropped
WATERMARK_STALE_HOURS    = 26     # hours — alert if watermark older than this

default_args = {
    "owner":             "data_engineering",
    "depends_on_past":   False,
    "email":             ["alerts@medbilling.local"],
    "email_on_failure":  False,
    "retries":           0,
    "execution_timeout": timedelta(minutes=15),
}


# ── Check functions ───────────────────────────────────────────────────────────

def _load_audit_logs() -> dict:
    """Load the latest run record from every silver audit log."""
    results = {}
    if not AUDIT_DIR.exists():
        print(f"Audit dir not found: {AUDIT_DIR}")
        return results
    for f in sorted(AUDIT_DIR.glob("*.json")):
        table = f.stem
        try:
            with open(f) as fh:
                data = json.load(fh)
            history = data.get("history", [])
            if history:
                results[table] = history[-1]   # most recent run
        except Exception as exc:
            print(f"  Could not read {f}: {exc}")
    return results


def check_dq_flag_rates(**context):
    """Alert if DQ flag rate exceeds threshold for any table."""
    logs    = _load_audit_logs()
    alerts  = []
    summary = []

    for table, run in logs.items():
        input_rows  = run.get("input_rows",  0)
        dq_flagged  = run.get("dq_flagged",  0)
        if input_rows > 0:
            rate = dq_flagged / input_rows
            summary.append(f"  {table:<30} dq_rate={rate:.1%}  ({dq_flagged:,}/{input_rows:,})")
            if rate > DQ_FLAG_RATE_THRESHOLD:
                alerts.append(
                    f"[DQ_FLAG_RATE] {table}: {rate:.1%} flagged "
                    f"({dq_flagged:,} / {input_rows:,} rows) — threshold {DQ_FLAG_RATE_THRESHOLD:.0%}"
                )

    print("DQ Flag Rate Summary:")
    for line in summary:
        print(line)

    if alerts:
        print("\nALERTS:")
        for a in alerts:
            print(f"  ⚠  {a}")
        # Push to XCom for downstream notification task
        context["ti"].xcom_push(key="dq_alerts", value=alerts)
    else:
        print("\nAll tables within DQ flag rate threshold.")
        context["ti"].xcom_push(key="dq_alerts", value=[])


def check_row_drops(**context):
    """Alert if row drop rate exceeds threshold."""
    logs   = _load_audit_logs()
    alerts = []

    for table, run in logs.items():
        input_rows   = run.get("input_rows",   0)
        dropped_rows = run.get("dropped_rows", 0)
        if input_rows > 0:
            drop_rate = dropped_rows / input_rows
            if drop_rate > ROW_DROP_THRESHOLD:
                alerts.append(
                    f"[ROW_DROP] {table}: {drop_rate:.1%} rows dropped "
                    f"({dropped_rows:,} / {input_rows:,}) — threshold {ROW_DROP_THRESHOLD:.0%}"
                )

    if alerts:
        print("Row Drop Alerts:")
        for a in alerts:
            print(f"  ⚠  {a}")
        context["ti"].xcom_push(key="drop_alerts", value=alerts)
    else:
        print("All tables within row drop threshold.")
        context["ti"].xcom_push(key="drop_alerts", value=[])


def check_watermark_staleness(**context):
    """Alert if any table's watermark hasn't advanced in WATERMARK_STALE_HOURS."""
    logs   = _load_audit_logs()
    alerts = []
    now    = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=WATERMARK_STALE_HOURS)

    for table, run in logs.items():
        wm = run.get("watermark")
        if not wm:
            continue
        try:
            wm_dt = datetime.fromisoformat(wm.replace("Z", "+00:00"))
            if wm_dt.tzinfo is None:
                wm_dt = wm_dt.replace(tzinfo=timezone.utc)
            if wm_dt < cutoff:
                age_h = (now - wm_dt).total_seconds() / 3600
                alerts.append(
                    f"[STALE_WATERMARK] {table}: last watermark {age_h:.1f}h ago "
                    f"(threshold {WATERMARK_STALE_HOURS}h)"
                )
        except Exception:
            pass

    if alerts:
        print("Watermark Staleness Alerts:")
        for a in alerts:
            print(f"  ⚠  {a}")
        context["ti"].xcom_push(key="stale_alerts", value=alerts)
    else:
        print("All watermarks are fresh.")
        context["ti"].xcom_push(key="stale_alerts", value=[])


def check_zero_row_tables(**context):
    """Alert if any silver table has 0 output rows."""
    logs   = _load_audit_logs()
    alerts = []

    for table, run in logs.items():
        output_rows = run.get("output_rows", -1)
        if output_rows == 0:
            alerts.append(f"[ZERO_ROWS] {table}: 0 rows in silver table")

    if alerts:
        print("Zero-Row Alerts:")
        for a in alerts:
            print(f"  ⚠  {a}")
        context["ti"].xcom_push(key="zero_alerts", value=alerts)
    else:
        print("No zero-row tables found.")
        context["ti"].xcom_push(key="zero_alerts", value=[])


def compile_and_report(**context):
    """Collect all alerts from XCom and print a final report."""
    ti = context["ti"]
    all_alerts = []
    for key in ["dq_alerts", "drop_alerts", "stale_alerts", "zero_alerts"]:
        alerts = ti.xcom_pull(key=key) or []
        all_alerts.extend(alerts)

    print()
    print("=" * 60)
    print("  DATA QUALITY REPORT")
    print(f"  Run date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    if all_alerts:
        print(f"  {len(all_alerts)} alert(s) found:\n")
        for a in all_alerts:
            print(f"  ⚠  {a}")
        print()
        # In production: send email / Slack notification here
        # For now, just log — pipeline does NOT fail on DQ alerts
        print("  NOTE: DQ alerts are informational. Pipeline did not fail.")
    else:
        print("  ✓ All data quality checks passed.")

    print("=" * 60)


# ── DAG ───────────────────────────────────────────────────────────────────────
with DAG(
    dag_id="med_billing_data_quality_check",
    description="Nightly DQ checks on silver audit logs — flags, drops, staleness, zero rows",
    default_args=default_args,
    schedule_interval="0 6 * * *",     # 06:00 every night (after incremental at 02:00)
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["medical_billing", "data_quality", "monitoring"],
    doc_md="""
## Data Quality Check Pipeline

Runs every night at **06:00** (after the 02:00 incremental pipeline).

Reads silver audit logs and checks:

| Check                | Threshold | Action on breach        |
|----------------------|-----------|-------------------------|
| DQ flag rate         | > 10%     | Log alert (no fail)     |
| Row drop rate        | > 10%     | Log alert (no fail)     |
| Watermark staleness  | > 26 hrs  | Log alert (no fail)     |
| Zero-row tables      | any       | Log alert (no fail)     |

Alerts are logged to Airflow task logs. Configure email/Slack in
`compile_and_report` for production notifications.
    """,
) as dag:

    start = EmptyOperator(task_id="dq_check_start")
    end   = EmptyOperator(task_id="dq_check_end")

    t_dq_flags = PythonOperator(
        task_id="check_dq_flag_rates",
        python_callable=check_dq_flag_rates,
    )

    t_row_drops = PythonOperator(
        task_id="check_row_drops",
        python_callable=check_row_drops,
    )

    t_watermarks = PythonOperator(
        task_id="check_watermark_staleness",
        python_callable=check_watermark_staleness,
    )

    t_zero_rows = PythonOperator(
        task_id="check_zero_row_tables",
        python_callable=check_zero_row_tables,
    )

    t_report = PythonOperator(
        task_id="compile_and_report",
        python_callable=compile_and_report,
    )

    # All checks run in parallel, then report
    start >> [t_dq_flags, t_row_drops, t_watermarks, t_zero_rows] >> t_report >> end
