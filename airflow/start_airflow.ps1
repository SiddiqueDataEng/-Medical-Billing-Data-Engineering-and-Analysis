# start_airflow.ps1
# =================
# Start Airflow scheduler + webserver (two background jobs).
#
# Run from the medical_billing_project_v2 directory:
#   .\airflow\start_airflow.ps1

$PYTHON       = "C:\pyspark_env\Scripts\python.exe"
$AIRFLOW_HOME = "$PSScriptRoot"

$env:AIRFLOW_HOME = $AIRFLOW_HOME

Write-Host ""
Write-Host "========================================================"
Write-Host "  Starting Airflow  (SQLite / LocalExecutor)"
Write-Host "  AIRFLOW_HOME : $AIRFLOW_HOME"
Write-Host "========================================================"
Write-Host ""

# Start scheduler in a new window
Write-Host "Starting scheduler ..."
$schedArgs = "-NoExit -Command `"`$env:AIRFLOW_HOME='$AIRFLOW_HOME'; & '$PYTHON' -m airflow scheduler`""
Start-Process powershell -ArgumentList $schedArgs -WindowStyle Normal

Start-Sleep -Seconds 3

# Start webserver in a new window
Write-Host "Starting webserver on http://localhost:8080 ..."
$webArgs = "-NoExit -Command `"`$env:AIRFLOW_HOME='$AIRFLOW_HOME'; & '$PYTHON' -m airflow webserver --port 8080`""
Start-Process powershell -ArgumentList $webArgs -WindowStyle Normal

Write-Host ""
Write-Host "  Two terminal windows opened (scheduler + webserver)."
Write-Host "  Web UI : http://localhost:8080"
Write-Host "  Login  : admin / admin"
Write-Host ""
Write-Host "  To stop: close both terminal windows."
Write-Host ""
