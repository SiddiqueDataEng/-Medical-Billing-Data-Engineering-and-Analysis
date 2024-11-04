# setup_airflow.ps1
# =================
# One-time setup: initialise Airflow DB and create admin user.
# Airflow is already installed in C:\pyspark_env.
#
# Run once from the medical_billing_project_v2 directory:
#   .\airflow\setup_airflow.ps1

$PYTHON       = "C:\pyspark_env\Scripts\python.exe"
$AIRFLOW_HOME = "$PSScriptRoot"   # the airflow\ folder next to this script

$env:AIRFLOW_HOME = $AIRFLOW_HOME

Write-Host ""
Write-Host "========================================================"
Write-Host "  Medical Billing — Airflow Setup (SQLite / LocalExecutor)"
Write-Host "========================================================"
Write-Host "  AIRFLOW_HOME : $AIRFLOW_HOME"
Write-Host "  Python       : $PYTHON"
Write-Host ""

# Create logs folder (Airflow needs it to exist)
New-Item -ItemType Directory -Force -Path "$AIRFLOW_HOME\logs" | Out-Null

# Initialise the SQLite database
Write-Host "[1/2] Initialising Airflow database ..."
& $PYTHON -m airflow db init
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: db init failed. Check output above." -ForegroundColor Red
    exit 1
}

# Create admin user (ignore error if already exists)
Write-Host ""
Write-Host "[2/2] Creating admin user (admin / admin) ..."
& $PYTHON -m airflow users create `
    --username admin `
    --password admin `
    --firstname Admin `
    --lastname User `
    --role Admin `
    --email admin@medbilling.local
# Don't exit on error — user may already exist

Write-Host ""
Write-Host "========================================================"
Write-Host "  Setup complete!"
Write-Host ""
Write-Host "  Start Airflow:  .\airflow\start_airflow.ps1"
Write-Host "  Web UI        : http://localhost:8080  (admin / admin)"
Write-Host "========================================================"
Write-Host ""
