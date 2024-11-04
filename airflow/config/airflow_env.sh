#!/bin/bash
# airflow_env.sh
# ==============
# Source this inside WSL before running any airflow CLI commands.
#
# Usage (inside WSL terminal):
#   source /mnt/.../airflow/config/airflow_env.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WSL_AIRFLOW="$(dirname "$SCRIPT_DIR")"
WSL_PROJECT="$(dirname "$WSL_AIRFLOW")"

export AIRFLOW_HOME="$WSL_AIRFLOW"
export AIRFLOW__CORE__DAGS_FOLDER="$WSL_AIRFLOW/dags"
export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN="sqlite:///$WSL_AIRFLOW/airflow.db"
export AIRFLOW__CORE__LOAD_EXAMPLES="False"
export AIRFLOW__CORE__EXECUTOR="LocalExecutor"
export AIRFLOW__SCHEDULER__CATCHUP_BY_DEFAULT="False"
export AIRFLOW__WEBSERVER__WEB_SERVER_PORT="8080"
export AIRFLOW__WEBSERVER__SECRET_KEY="med_billing_local_dev_secret_change_in_prod"
export AIRFLOW__LOGGING__LOGGING_LEVEL="INFO"

# PySpark / Java
export JAVA_HOME="/usr/lib/jvm/java-17-openjdk-amd64"
export SPARK_LOCAL_IP="127.0.0.1"
export PYSPARK_PYTHON="$HOME/airflow_venv/bin/python3"
export PYSPARK_DRIVER_PYTHON="$HOME/airflow_venv/bin/python3"

# Make pipeline modules importable
export PYTHONPATH="$WSL_PROJECT/pipelines:$WSL_PROJECT:$PYTHONPATH"

echo "Airflow env set. AIRFLOW_HOME=$AIRFLOW_HOME"
