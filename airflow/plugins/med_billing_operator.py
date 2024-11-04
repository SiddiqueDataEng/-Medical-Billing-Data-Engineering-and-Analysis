"""
Custom operator — calls run_pipeline.py via the mounted project volume.
PySpark runs inside the container using the pre-installed airflow Python.
The project files are mounted at /opt/project.
"""
import os
import subprocess
from pathlib import Path
from airflow.models import BaseOperator
from airflow.exceptions import AirflowException
from airflow.utils.decorators import apply_defaults

PROJECT_DIR  = Path("/opt/project")
RUN_SCRIPT   = PROJECT_DIR / "pipelines" / "run_pipeline.py"
PYTHON       = "python"


class MedBillingPipelineOperator(BaseOperator):
    template_fields = ("mode",)
    ui_color = "#d4eaf7"

    @apply_defaults
    def __init__(self, layers=None, mode="FULL", bronze_mode=None,
                 silver_mode=None, gold_mode=None, tables=None,
                 window_days=7, **kwargs):
        super().__init__(**kwargs)
        self.layers      = layers or ["bronze", "silver", "gold"]
        self.mode        = mode.upper()
        self.bronze_mode = bronze_mode
        self.silver_mode = silver_mode
        self.gold_mode   = gold_mode
        self.tables      = tables
        self.window_days = window_days

    def execute(self, context):
        cmd = [PYTHON, str(RUN_SCRIPT),
               "--layers", ",".join(self.layers),
               "--mode", self.mode,
               "--window-days", str(self.window_days)]
        if self.bronze_mode: cmd += ["--bronze-mode", self.bronze_mode]
        if self.silver_mode: cmd += ["--silver-mode", self.silver_mode]
        if self.gold_mode:   cmd += ["--gold-mode",   self.gold_mode]
        if self.tables:      cmd += ["--tables", ",".join(self.tables)]

        self.log.info("Running: %s", " ".join(cmd))

        env = os.environ.copy()
        env["PYTHONPATH"] = f"{PROJECT_DIR}/pipelines:{PROJECT_DIR}"

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                env=env, cwd=str(PROJECT_DIR),
                                text=True, bufsize=1)
        lines = []
        for line in proc.stdout:
            self.log.info(line.rstrip())
            lines.append(line)
        proc.wait()

        if proc.returncode != 0:
            raise AirflowException(
                f"Pipeline failed (exit {proc.returncode})\n" +
                "".join(lines[-30:]))
        return {"exit_code": 0}
