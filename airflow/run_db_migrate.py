"""Run airflow db migrate programmatically — avoids CLI subprocess issues on Windows."""
import warnings
warnings.filterwarnings("ignore")

import os
os.environ.setdefault("AIRFLOW_HOME", os.path.join(os.path.expanduser("~"), "airflow"))

print(f"AIRFLOW_HOME: {os.environ['AIRFLOW_HOME']}")

from airflow.utils.db import initdb
from airflow import settings

print("Initialising Airflow database...")
initdb()
print("Database initialised OK.")
