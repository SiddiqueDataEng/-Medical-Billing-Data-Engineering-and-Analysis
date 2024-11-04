# stop_airflow.ps1
# ================
# Stop Airflow running inside WSL2.
#
# Usage:
#   .\airflow\stop_airflow.ps1

$WIN_PROJECT = (Get-Item $PSScriptRoot).Parent.FullName
$WIN_AIRFLOW = Join-Path $WIN_PROJECT "airflow"
$WSL_AIRFLOW = wsl wslpath -u $WIN_AIRFLOW.Replace('\','/')

Write-Host "Stopping Airflow ..."
wsl bash "$WSL_AIRFLOW/stop_airflow.sh"
