# airflow_env.ps1
# ================
# Source this file in any terminal before running airflow CLI commands.
#
# Usage:
#   . .\airflow\config\airflow_env.ps1

$PROJECT_DIR  = (Get-Item $PSScriptRoot).Parent.Parent.FullName
$AIRFLOW_HOME = Join-Path $PROJECT_DIR "airflow"

$env:AIRFLOW_HOME                        = $AIRFLOW_HOME
$env:AIRFLOW__CORE__DAGS_FOLDER          = "$AIRFLOW_HOME\dags"
$env:AIRFLOW__DATABASE__SQL_ALCHEMY_CONN = "sqlite:///$AIRFLOW_HOME/airflow.db"
$env:AIRFLOW__CORE__LOAD_EXAMPLES        = "False"
$env:AIRFLOW__CORE__EXECUTOR             = "LocalExecutor"
$env:AIRFLOW__SCHEDULER__CATCHUP_BY_DEFAULT = "False"

# Spark / Java / Hadoop
$env:HADOOP_HOME           = "C:\hadoop"
$env:JAVA_HOME             = "C:\Program Files\Eclipse Adoptium\jdk-17.0.19.10-hotspot"
$env:PYSPARK_PYTHON        = "C:\pyspark_env\Scripts\python.exe"
$env:PYSPARK_DRIVER_PYTHON = "C:\pyspark_env\Scripts\python.exe"
$env:SPARK_LOCAL_IP        = "127.0.0.1"

Write-Host "Airflow environment set. AIRFLOW_HOME = $AIRFLOW_HOME"
