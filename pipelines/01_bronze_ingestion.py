"""
01_bronze_ingestion.py
======================
Bronze Layer — Raw Ingestion (Production-Grade)

Supported load modes
--------------------
FULL        – Drop and reload the entire bronze table from both raw sources.
INCREMENTAL – Read only files modified after the last watermark timestamp,
              append to the existing bronze table.
WINDOW      – Re-ingest a rolling N-day window (idempotent; overwrites that
              window's data in the target).
CDC         – Merge source rows into bronze using the primary key (upsert).
FDC         – Full-Diff-Compare: hash every source row, write only changed rows.

Usage (CLI)
-----------
    python pipelines/01_bronze_ingestion.py [--mode FULL|INCREMENTAL|WINDOW|CDC|FDC]
                                            [--tables clinics,patients,...]
                                            [--window-days 7]

Usage (import)
--------------
    from pipelines import bronze_ingestion
    bronze_ingestion.run(mode="INCREMENTAL", tables=["claims","claim_lines"])
"""

import argparse
import shutil
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ── project root on path ──────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "pipelines"))

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

from pipeline_config import (
    RAW_CLEAN_DIR, RAW_ISSUES_DIR, BRONZE_DIR, CHECKPOINT_DIR,
    ALL_TABLES, TABLE_CONFIGS,
    FULL, INCREMENTAL, WINDOW, CDC, FDC,
)
from pipeline_utils import (
    get_spark, safe_union,
    read_watermark, write_watermark, write_audit_log,
    dispatch_write, print_run_summary, log,
)

LAYER = "bronze"


# ═══════════════════════════════════════════════════════════════════════════════
#  Source readers
# ═══════════════════════════════════════════════════════════════════════════════

def _read_folder_all_string(spark: SparkSession, folder: Path,
                             source_tag: str) -> DataFrame | None:
    """
    Read every CSV + Parquet file in *folder*.
    All columns are cast to StringType so clean and dirty data can always
    be unioned regardless of schema drift or injected type errors.
    """
    csv_files = sorted(folder.glob("*.csv"))
    pq_files  = sorted(folder.glob("*.parquet"))

    frames = []

    if csv_files:
        df = (
            spark.read
            .option("header",      "true")
            .option("inferSchema", "false")
            .option("multiLine",   "true")
            .option("escape",      '"')
            .option("encoding",    "UTF-8")
            .csv([str(f) for f in csv_files])
        )
        frames.append(df)

    if pq_files:
        df = spark.read.parquet(*[str(f) for f in pq_files])
        df = df.select([F.col(c).cast(StringType()).alias(c) for c in df.columns])
        frames.append(df)

    if not frames:
        return None

    result = safe_union(frames)
    return result.withColumn("_source", F.lit(source_tag)) if result else None


def _add_bronze_metadata(df: DataFrame, table_name: str) -> DataFrame:
    """Stamp standard bronze metadata columns onto every row."""
    business_cols = [c for c in df.columns if not c.startswith("_")]
    return (
        df
        .withColumn("_bronze_table",   F.lit(table_name))
        .withColumn("_ingested_at",    F.current_timestamp())
        .withColumn("_ingestion_date", F.current_date())
        .withColumn("_row_hash",
                    F.md5(F.concat_ws("|", *[
                        F.coalesce(F.col(c), F.lit("")) for c in business_cols
                    ])))
    )


def _read_source(spark: SparkSession, table_name: str) -> DataFrame | None:
    """
    Union clean + dirty source folders for *table_name*.
    Returns a single all-string DataFrame or None if no data found.
    """
    clean_dir  = RAW_CLEAN_DIR  / table_name
    issues_dir = RAW_ISSUES_DIR / table_name

    frames = []
    for folder, tag in [(clean_dir, "raw_clean"), (issues_dir, "raw_issues")]:
        if folder.exists():
            df = _read_folder_all_string(spark, folder, tag)
            if df is not None:
                frames.append(df)
                log.info("[%s] read %s: %d files", table_name, tag,
                         len(list(folder.glob("*.csv"))) + len(list(folder.glob("*.parquet"))))
        else:
            log.warning("[%s] source folder not found: %s", table_name, folder)

    return safe_union(frames)


# ═══════════════════════════════════════════════════════════════════════════════
#  Watermark filtering  (INCREMENTAL / WINDOW)
# ═══════════════════════════════════════════════════════════════════════════════

def _apply_watermark_filter(df: DataFrame, table_name: str,
                             mode: str, window_days: int) -> tuple[DataFrame, str]:
    """
    Filter *df* to only rows newer than the last watermark.
    Returns (filtered_df, new_watermark_value).
    """
    cfg = TABLE_CONFIGS.get(table_name)
    wm_col = cfg.watermark_col if cfg else None

    if not wm_col or wm_col not in df.columns:
        log.warning("[%s] no watermark column – returning full dataset", table_name)
        return df, datetime.now(timezone.utc).isoformat()

    if mode == WINDOW:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat()
        log.info("[%s] WINDOW filter: %s >= %s", table_name, wm_col, cutoff)
    else:  # INCREMENTAL
        cutoff = read_watermark(LAYER, table_name)
        if cutoff is None:
            log.info("[%s] no prior watermark – loading all rows", table_name)
            cutoff = "1900-01-01"
        else:
            log.info("[%s] INCREMENTAL filter: %s > %s", table_name, wm_col, cutoff)

    filtered = df.filter(F.col(wm_col).cast("timestamp") > F.lit(cutoff).cast("timestamp"))

    # New watermark = max value in this batch
    max_row = filtered.agg(F.max(F.col(wm_col).cast("timestamp")).alias("mx")).collect()
    new_wm  = str(max_row[0]["mx"]) if max_row and max_row[0]["mx"] else cutoff

    return filtered, new_wm


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-table ingestion
# ═══════════════════════════════════════════════════════════════════════════════

def ingest_table(spark: SparkSession, table_name: str,
                 mode: str, window_days: int = 7) -> dict:
    """
    Ingest one table end-to-end.
    Returns a stats dict for the run summary.
    """
    cfg         = TABLE_CONFIGS.get(table_name)
    pk          = cfg.primary_key    if cfg else table_name + "_id"
    part_cols   = cfg.partition_cols if cfg else []
    track_cols  = cfg.scd2_track_cols if cfg else []
    target_path = str(BRONZE_DIR / table_name)

    t0 = time.time()
    stats = {
        "table":        table_name,
        "mode":         mode,
        "input_rows":   0,
        "output_rows":  0,
        "dropped_rows": 0,
        "dq_flagged":   0,
        "watermark":    None,
        "status":       "SKIP",
    }

    # ── 1. Read source ────────────────────────────────────────────────────────
    df = _read_source(spark, table_name)
    if df is None:
        log.warning("[%s] no source data found – skipping", table_name)
        return stats

    input_rows = df.count()
    stats["input_rows"] = input_rows
    log.info("[%s] source rows: %d", table_name, input_rows)

    # ── 1b. If target exists but is empty/broken, force FULL mode ─────────────
    effective_mode = mode
    target_dir = BRONZE_DIR / table_name
    if target_dir.exists():
        delta_log = target_dir / "_delta_log"
        if delta_log.exists():
            commit_files = list(delta_log.glob("*.json"))
            if len(commit_files) == 0:
                log.warning("[%s] broken delta log (no commits) – forcing FULL reload", table_name)
                effective_mode = FULL
                # Remove the broken delta directory so write_full can start clean
                shutil.rmtree(str(target_dir))
                target_dir.mkdir(parents=True, exist_ok=True)

    # ── 2. Watermark filter (INCREMENTAL / WINDOW) ────────────────────────────
    new_wm = datetime.now(timezone.utc).isoformat()
    if effective_mode in (INCREMENTAL, WINDOW):
        df, new_wm = _apply_watermark_filter(df, table_name, effective_mode, window_days)
        filtered_rows = df.count()
        log.info("[%s] after watermark filter: %d rows", table_name, filtered_rows)
        if filtered_rows == 0:
            log.info("[%s] no new rows – nothing to write", table_name)
            stats["status"] = "OK (no new rows)"
            return stats

    # ── 3. Add bronze metadata ────────────────────────────────────────────────
    df = _add_bronze_metadata(df, table_name)

    # ── 4. Write via strategy ─────────────────────────────────────────────────
    write_stats = dispatch_write(
        spark, df, target_path,
        mode=effective_mode, pk=pk,
        partition_cols=part_cols,
        track_cols=track_cols,
        hash_col="_row_hash",
    )

    output_rows = write_stats.get("rows_written", 0)
    elapsed     = round(time.time() - t0, 2)

    stats.update({
        "output_rows":  output_rows,
        "dropped_rows": input_rows - output_rows if output_rows >= 0 else 0,
        "watermark":    new_wm,
        "elapsed_s":    elapsed,
        "status":       "OK",
        **write_stats,
    })

    # ── 5. Persist watermark + audit log ─────────────────────────────────────
    if effective_mode in (INCREMENTAL, WINDOW):
        write_watermark(LAYER, table_name, new_wm)
    write_audit_log(LAYER, table_name, stats)

    log.info("[%s] ✓ %s  in=%d  out=%d  %.1fs",
             table_name, mode, input_rows, output_rows, elapsed)
    return stats


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def run(mode: str = FULL,
        tables: list[str] = None,
        window_days: int = 7) -> list[dict]:
    """
    Run the bronze ingestion layer.
    Each table gets its own SparkSession to prevent JVM heap accumulation
    from killing subsequent tables on a single-node local setup.
    """
    mode   = mode.upper()
    tables = tables or ALL_TABLES

    if mode not in {FULL, INCREMENTAL, WINDOW, CDC, FDC}:
        raise ValueError(f"Invalid mode for bronze: {mode!r}")

    t_start = time.time()
    log.info("=" * 70)
    log.info("BRONZE INGESTION  mode=%s  tables=%d  started=%s",
             mode, len(tables), datetime.now(timezone.utc).isoformat())
    log.info("=" * 70)

    BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    (CHECKPOINT_DIR / "bronze").mkdir(parents=True, exist_ok=True)

    all_stats = []

    for table in tables:
        log.info("\n── %s ──", table)
        spark = None
        try:
            spark = get_spark("MedBilling_Bronze")
            s = ingest_table(spark, table, mode=mode, window_days=window_days)
        except Exception as exc:
            log.error("[%s] FAILED: %s", table, exc, exc_info=True)
            s = {"table": table, "mode": mode, "input_rows": 0,
                 "output_rows": 0, "dropped_rows": 0, "dq_flagged": 0,
                 "status": f"ERROR: {exc}"}
        finally:
            if spark is not None:
                try:
                    spark.stop()
                except Exception:
                    pass
        all_stats.append(s)

    elapsed = time.time() - t_start
    print_run_summary(LAYER, all_stats, elapsed)
    return all_stats


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(description="Bronze ingestion pipeline")
    p.add_argument("--mode",        default=FULL,
                   choices=[FULL, INCREMENTAL, WINDOW, CDC, FDC],
                   help="Load mode (default: FULL)")
    p.add_argument("--tables",      default=None,
                   help="Comma-separated table names (default: all)")
    p.add_argument("--window-days", type=int, default=7,
                   help="Days for WINDOW mode (default: 7)")
    return p.parse_args()


if __name__ == "__main__":
    args   = _parse_args()
    tables = [t.strip() for t in args.tables.split(",")] if args.tables else None
    run(mode=args.mode, tables=tables, window_days=args.window_days)
