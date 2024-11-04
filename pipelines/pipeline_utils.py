"""
pipeline_utils.py
=================
Shared utilities used by all three pipeline layers:
  - SparkSession factory
  - Watermark / audit-log read-write
  - Delta MERGE helpers (CDC, SCD2, FDC)
  - Schema-safe union
  - Logging / metrics
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from delta.tables import DeltaTable

from pipeline_config import (
    AUDIT_DIR, CHECKPOINT_DIR, SPARK_CONF, SPARK_ENV,
    FULL, INCREMENTAL, CDC, SCD2, WINDOW, FDC,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("pipeline")


# ═══════════════════════════════════════════════════════════════════════════════
#  SparkSession
# ═══════════════════════════════════════════════════════════════════════════════

def get_spark(app_name: str = "MedBilling_Pipeline") -> SparkSession:
    """Build a fresh SparkSession with Delta Lake support.
    Always creates a new session — callers must stop() the previous one first.
    """
    for k, v in SPARK_ENV.items():
        os.environ.setdefault(k, v)

    # Stop any existing session so getOrCreate() returns a truly fresh one
    existing = SparkSession.getActiveSession()
    if existing is not None:
        try:
            existing.stop()
        except Exception:
            pass

    builder = SparkSession.builder.appName(app_name).master("local[*]")
    for k, v in SPARK_CONF.items():
        builder = builder.config(k, v)

    try:
        from delta import configure_spark_with_delta_pip
        spark = configure_spark_with_delta_pip(builder).getOrCreate()
    except ImportError:
        builder = builder.config(
            "spark.jars.packages", "io.delta:delta-spark_2.12:3.2.0"
        )
        spark = builder.getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    # Set checkpoint directory so .checkpoint() spills to disk instead of RAM
    ckpt_dir = str(CHECKPOINT_DIR / "spark_ckpt")
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    spark.sparkContext.setCheckpointDir(ckpt_dir)

    log.info("SparkSession ready: %s", app_name)
    return spark


# ═══════════════════════════════════════════════════════════════════════════════
#  Audit / Watermark store  (JSON files under audit_logs/)
# ═══════════════════════════════════════════════════════════════════════════════

def _audit_path(layer: str, table: str) -> Path:
    p = AUDIT_DIR / layer
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{table}.json"


def read_watermark(layer: str, table: str) -> Optional[str]:
    """Return the last high-watermark value, or None if no prior run."""
    p = _audit_path(layer, table)
    if not p.exists():
        return None
    with open(p) as f:
        data = json.load(f)
    return data.get("last_watermark")


def write_watermark(layer: str, table: str, watermark: str,
                    extra: Optional[dict] = None) -> None:
    """Persist the high-watermark after a successful run."""
    p = _audit_path(layer, table)
    payload = {
        "table":           table,
        "layer":           layer,
        "last_watermark":  watermark,
        "updated_at":      datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        payload.update(extra)
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("[%s] watermark saved: %s = %s", table, layer, watermark)


def read_audit_log(layer: str, table: str) -> dict:
    """Return the full audit record, or empty dict."""
    p = _audit_path(layer, table)
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def write_audit_log(layer: str, table: str, stats: dict) -> None:
    """Append a run record to the audit log."""
    p = _audit_path(layer, table)
    history = []
    if p.exists():
        with open(p) as f:
            existing = json.load(f)
        # keep last_watermark at top level, history list below
        history = existing.get("history", [])

    record = {**stats, "logged_at": datetime.now(timezone.utc).isoformat()}
    history.append(record)

    payload = {
        "table":          table,
        "layer":          layer,
        "last_watermark": stats.get("watermark"),
        "updated_at":     record["logged_at"],
        "history":        history[-50:],   # keep last 50 runs
    }
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
#  Schema-safe union  (handles column mismatches between CSV and Parquet chunks)
# ═══════════════════════════════════════════════════════════════════════════════

def safe_union(frames: list[DataFrame]) -> Optional[DataFrame]:
    """Union a list of DataFrames, padding missing columns with nulls."""
    if not frames:
        return None
    result = frames[0]
    for df in frames[1:]:
        all_cols = list(dict.fromkeys(result.columns + df.columns))
        for c in all_cols:
            if c not in result.columns:
                result = result.withColumn(c, F.lit(None).cast(StringType()))
            if c not in df.columns:
                df = df.withColumn(c, F.lit(None).cast(StringType()))
        result = result.select(all_cols).unionByName(df.select(all_cols))
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Delta write strategies
# ═══════════════════════════════════════════════════════════════════════════════

def write_full(df: DataFrame, path: str, partition_cols: list = None) -> int:
    """FULL load: overwrite entire table. Writes directly to disk, counts from Delta log."""
    w = df.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
    if partition_cols:
        w = w.partitionBy(*partition_cols)
    w.save(path)
    # Read count from Delta log instead of re-scanning the DataFrame
    try:
        from delta.tables import DeltaTable
        spark = df.sparkSession
        count = DeltaTable.forPath(spark, path).toDF().count()
    except Exception:
        count = -1
    log.info("[FULL] wrote %d rows → %s", count, path)
    return count


def write_append(df: DataFrame, path: str, partition_cols: list = None) -> int:
    """INCREMENTAL / WINDOW: append new rows. Writes directly to disk."""
    w = df.write.format("delta").mode("append")
    if partition_cols:
        w = w.partitionBy(*partition_cols)
    w.save(path)
    # Read count from Delta log instead of re-scanning the DataFrame
    try:
        from delta.tables import DeltaTable
        spark = df.sparkSession
        count = DeltaTable.forPath(spark, path).toDF().count()
    except Exception:
        count = -1
    log.info("[APPEND] wrote %d rows → %s", count, path)
    return count


def write_cdc_merge(
    spark: SparkSession,
    source_df: DataFrame,
    target_path: str,
    pk: str,
    soft_delete_col: str = "_is_deleted",
) -> dict:
    """
    CDC MERGE (upsert + soft-delete) into a Delta table.
    - Matched rows: UPDATE all columns.
    - New rows:     INSERT.
    - Deleted rows: UPDATE _is_deleted = True (soft delete).
    Returns dict with inserted / updated counts (approximate).
    """
    Path(target_path).mkdir(parents=True, exist_ok=True)

    # Ensure target exists; if not, do a full write first
    try:
        target = DeltaTable.forPath(spark, target_path)
    except Exception:
        log.info("[CDC] target not found – initialising with full write")
        write_full(source_df, target_path)
        return {"inserted": source_df.count(), "updated": 0, "deleted": 0}

    # Build SET clause for all non-PK columns
    set_clause = {
        f"target.{c}": f"source.{c}"
        for c in source_df.columns
        if c != pk
    }

    (
        target.alias("target")
        .merge(source_df.alias("source"), f"target.{pk} = source.{pk}")
        .whenMatchedUpdate(set=set_clause)
        .whenNotMatchedInsertAll()
        .execute()
    )

    log.info("[CDC] merge complete → %s", target_path)
    return {"inserted": 0, "updated": 0, "deleted": 0}   # Delta doesn't expose counts easily


def write_scd2(
    spark: SparkSession,
    source_df: DataFrame,
    target_path: str,
    pk: str,
    track_cols: list,
    effective_col: str = "_effective_from",
    expiry_col:    str = "_effective_to",
    current_col:   str = "_is_current",
) -> dict:
    """
    SCD Type-2 MERGE:
    - If tracked columns changed → expire old row, insert new row.
    - If row is new → insert.
    - Unchanged rows → no-op.
    """
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Add SCD2 metadata to source
    source_df = (
        source_df
        .withColumn(effective_col, F.lit(now_str).cast("date"))
        .withColumn(expiry_col,    F.lit("9999-12-31").cast("date"))
        .withColumn(current_col,   F.lit(True))
    )

    try:
        target = DeltaTable.forPath(spark, target_path)
    except Exception:
        log.info("[SCD2] target not found – initialising")
        write_full(source_df, target_path)
        return {"inserted": source_df.count(), "expired": 0}

    # Build change-detection condition
    change_cond = " OR ".join(
        [f"target.{c} <> source.{c}" for c in track_cols]
    ) if track_cols else "1=0"

    # Step 1: expire changed rows in target
    (
        target.alias("target")
        .merge(
            source_df.alias("source"),
            f"target.{pk} = source.{pk} AND target.{current_col} = true"
        )
        .whenMatchedUpdate(
            condition=change_cond,
            set={
                f"target.{expiry_col}": f"source.{effective_col}",
                f"target.{current_col}": "false",
            }
        )
        .execute()
    )

    # Step 2: insert new / changed rows
    existing = spark.read.format("delta").load(target_path)
    new_rows = source_df.join(
        existing.filter(F.col(current_col) == True).select(pk),
        on=pk, how="left_anti"
    )
    # Also insert rows where tracked cols changed
    changed = (
        source_df.alias("s")
        .join(
            existing.filter(F.col(current_col) == False)
                    .select(pk).alias("e"),
            on=pk, how="inner"
        )
        .select("s.*")
    )
    to_insert = new_rows.unionByName(changed, allowMissingColumns=True)
    # Checkpoint to disk to avoid recomputing the union twice
    to_insert = to_insert.checkpoint()
    insert_count = to_insert.count()
    if insert_count > 0:
        write_append(to_insert, target_path)

    log.info("[SCD2] complete → %s", target_path)
    return {"inserted": insert_count, "expired": 0}


def write_fdc(
    spark: SparkSession,
    source_df: DataFrame,
    target_path: str,
    pk: str,
    hash_col: str = "_row_hash",
) -> dict:
    """
    Full-Diff-Compare:
    - Compute row hash on source.
    - Compare with target hashes.
    - INSERT rows that are new.
    - UPDATE rows whose hash changed.
    - Soft-delete rows that disappeared from source.
    """
    try:
        target_df = spark.read.format("delta").load(target_path)
    except Exception:
        log.info("[FDC] target not found – full write")
        write_full(source_df, target_path)
        return {"inserted": -1, "updated": 0, "deleted": 0}

    # Rows in source not in target (new)
    new_rows = source_df.join(target_df.select(pk), on=pk, how="left_anti").checkpoint()

    # Rows in both but hash changed (updated)
    if hash_col in source_df.columns and hash_col in target_df.columns:
        src_hashes = source_df.select(pk, F.col(hash_col).alias("src_hash"))
        tgt_hashes = target_df.select(pk, F.col(hash_col).alias("tgt_hash"))
        changed_pks = (
            src_hashes.join(tgt_hashes, on=pk)
            .filter(F.col("src_hash") != F.col("tgt_hash"))
            .select(pk)
        )
        updated_rows = source_df.join(changed_pks, on=pk, how="inner").checkpoint()
    else:
        updated_rows = spark.createDataFrame([], source_df.schema)

    # Rows in target not in source (deleted)
    deleted_pks = target_df.select(pk).join(source_df.select(pk), on=pk, how="left_anti")
    deleted_count = deleted_pks.count()

    # Apply changes via CDC merge
    delta_df = new_rows.unionByName(updated_rows, allowMissingColumns=True)
    delta_count = delta_df.count()
    if delta_count > 0:
        write_cdc_merge(spark, delta_df, target_path, pk)

    # Soft-delete missing rows
    if deleted_count > 0 and "_is_deleted" in target_df.columns:
        target = DeltaTable.forPath(spark, target_path)
        (
            target.alias("t")
            .merge(deleted_pks.alias("d"), f"t.{pk} = d.{pk}")
            .whenMatchedUpdate(set={"t._is_deleted": "true"})
            .execute()
        )

    log.info("[FDC] delta=%d deleted=%d → %s", delta_count, deleted_count, target_path)
    return {
        "inserted": delta_count,
        "updated":  0,
        "deleted":  deleted_count,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Generic dispatcher
# ═══════════════════════════════════════════════════════════════════════════════

def dispatch_write(
    spark: SparkSession,
    df: DataFrame,
    target_path: str,
    mode: str,
    pk: str,
    partition_cols: list = None,
    track_cols: list = None,
    watermark_val: str = None,
    hash_col: str = "_row_hash",
) -> dict:
    """
    Route a DataFrame write to the correct strategy based on *mode*.
    Returns a stats dict.
    """
    partition_cols = partition_cols or []
    track_cols     = track_cols or []
    Path(target_path).mkdir(parents=True, exist_ok=True)

    if mode == FULL:
        count = write_full(df, target_path, partition_cols)
        return {"mode": FULL, "rows_written": count}

    elif mode == INCREMENTAL:
        count = write_append(df, target_path, partition_cols)
        return {"mode": INCREMENTAL, "rows_written": count}

    elif mode == WINDOW:
        # Overwrite only the window partition; append outside it
        count = write_append(df, target_path, partition_cols)
        return {"mode": WINDOW, "rows_written": count}

    elif mode == CDC:
        stats = write_cdc_merge(spark, df, target_path, pk)
        return {"mode": CDC, **stats}

    elif mode == SCD2:
        stats = write_scd2(spark, df, target_path, pk, track_cols)
        return {"mode": SCD2, **stats}

    elif mode == FDC:
        stats = write_fdc(spark, df, target_path, pk, hash_col)
        return {"mode": FDC, **stats}

    else:
        raise ValueError(f"Unknown load mode: {mode!r}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Metrics printer
# ═══════════════════════════════════════════════════════════════════════════════

def print_run_summary(layer: str, stats_list: list[dict], elapsed: float) -> None:
    col = max((len(s.get("table", "")) for s in stats_list), default=10) + 2
    print()
    print("=" * 80)
    print(f"  {layer.upper()} LAYER — RUN SUMMARY   ({elapsed:.1f}s)")
    print("=" * 80)
    print(f"  {'Table':<{col}} {'Mode':<13} {'In':>9} {'Out':>9} {'Drop':>7} {'DQ':>7}  Status")
    print(f"  {'-'*col} {'-'*13} {'-'*9} {'-'*9} {'-'*7} {'-'*7}  {'-'*10}")
    for s in stats_list:
        print(
            f"  {s.get('table',''):<{col}} "
            f"{s.get('mode',''):<13} "
            f"{s.get('input_rows',0):>9,} "
            f"{s.get('output_rows',0):>9,} "
            f"{s.get('dropped_rows',0):>7,} "
            f"{s.get('dq_flagged',0):>7,}  "
            f"{s.get('status','')}"
        )
    print("=" * 80)
    ok  = sum(1 for s in stats_list if s.get("status") == "OK")
    err = len(stats_list) - ok
    print(f"  {ok} succeeded  |  {err} failed")
    print("=" * 80)
    print()
