# Dockerfile — Minimal Airflow + PySpark + Delta Lake
# =====================================================
# Built on the official apache/airflow image.
# Adds Java 17 (required by PySpark) and installs pyspark + delta-spark.

FROM apache/airflow:2.9.3-python3.11

USER root

# Java 17 for PySpark
RUN apt-get update -qq \
 && apt-get install -y -qq --no-install-recommends openjdk-17-jdk-headless \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

USER airflow

# PySpark + Delta Lake — same versions as the pipeline
RUN pip install --no-cache-dir \
    "pyspark==3.5.1" \
    "delta-spark==3.2.0"
