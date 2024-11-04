# Airflow — Medical Billing Pipeline

**SQLite + LocalExecutor + Docker** — single container, no Postgres, no Redis.

> Airflow has Linux-only dependencies (`fcntl`, `grp`, `pwd`) that prevent it
> from running natively on Windows. Docker is the minimal working solution —
> you already have it installed.

---

## Start (3 commands, run once then just the last two)

```powershell
# 1. Go to the airflow folder
cd medical_billing_project_v2\airflow

# 2. Build image + start  (first time — takes ~5 min to download/build)
docker compose up --build -d

# 3. Open the UI
start http://localhost:8080
#    Username: admin
#    Password: admin
```

After the first build, just use:
```powershell
docker compose up -d      # start
docker compose down       # stop
```

---

## What's running inside the container

```
Single Docker container
  ├── Airflow Scheduler   (background process)
  ├── Airflow Webserver   → http://localhost:8080
  └── SQLite DB           → persisted in Docker named volume
```

Your project files are **mounted read-write** at `/opt/project` inside the
container — Delta tables written by the pipeline land back on your Windows
filesystem at `medical_billing_project_v2/delta/`.

---

## DAGs

| DAG | Schedule | What it does |
|-----|----------|-------------|
| `med_billing_daily_incremental` | 02:00 nightly | Bronze INCREMENTAL → Silver CDC → Gold INCREMENTAL |
| `med_billing_weekly_full_reload` | 01:00 Sunday | Bronze FULL → Silver FULL → Gold FULL |
| `med_billing_backfill` | Manual only | Re-process any window via UI config |
| `med_billing_data_quality_check` | 06:00 nightly | DQ monitoring on silver audit logs |

Toggle each DAG **ON** in the UI after first login.

---

## Useful commands

```powershell
# View live logs
docker compose logs -f airflow

# Run a specific task manually (for testing)
docker exec medbilling_airflow airflow tasks test med_billing_daily_incremental bronze_incremental 2024-01-01

# Trigger backfill with config
docker exec medbilling_airflow airflow dags trigger med_billing_backfill --conf "{\"window_days\": 30}"

# Open a shell inside the container
docker exec -it medbilling_airflow bash

# Stop and remove everything (keeps your project data)
docker compose down

# Stop and also delete the Airflow DB (full reset)
docker compose down -v
```

---

## Architecture

```
Windows 11
│
├── Docker Desktop (already installed)
│     └── Container: medbilling_airflow
│           ├── Airflow 2.9.3
│           ├── Python 3.11
│           ├── PySpark 3.5.1
│           ├── Delta Spark 3.2.0
│           └── Java 17
│
├── Volumes (persisted)
│     ├── airflow-db    → SQLite database
│     └── airflow-logs  → task logs
│
└── medical_billing_project_v2/   ← mounted at /opt/project
      ├── pipelines/run_pipeline.py
      ├── raw_data/
      ├── raw_data_with_issues/
      └── delta/                  ← Delta Lake output
```

---

## File structure

```
airflow/
├── Dockerfile                    ← Airflow + Java + PySpark image
├── docker-compose.yml            ← single-service setup
├── dags/
│   ├── dag_daily_incremental.py
│   ├── dag_weekly_full_reload.py
│   ├── dag_backfill.py
│   └── dag_data_quality_check.py
└── plugins/
    └── med_billing_operator.py   ← custom operator
```
