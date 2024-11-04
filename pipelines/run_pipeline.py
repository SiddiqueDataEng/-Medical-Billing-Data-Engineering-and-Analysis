"""
run_pipeline.py
===============
Master Orchestrator — Medical Billing Medallion Pipeline

Runs Bronze → Silver → Gold in sequence (or any subset), with full
support for all load modes at each layer.

Load Modes
----------
FULL        – Truncate-and-reload the entire target table.
INCREMENTAL – Append only rows newer than the last high-watermark.
CDC         – Change-Data-Capture MERGE (upsert + soft-delete).
SCD2        – Slowly-Changing Dimension Type-2 (expire + insert).
WINDOW      – Re-process a rolling N-day window (idempotent).
FDC         – Full-Diff-Compare: write only rows whose hash changed.

Usage
-----
# Full pipeline, all tables, FULL load
python pipelines/run_pipeline.py

# Incremental load for all layers
python pipelines/run_pipeline.py --mode INCREMENTAL

# Only bronze + silver, specific tables
python pipelines/run_pipeline.py --layers bronze,silver --tables claims,claim_lines

# Different mode per layer
python pipelines/run_pipeline.py --bronze-mode FULL --silver-mode CDC --gold-mode INCREMENTAL

# Rolling 14-day window
python pipelines/run_pipeline.py --mode WINDOW --window-days 14

# Dry-run: print plan without executing
python pipelines/run_pipeline.py --dry-run
"""

import argparse
import importlib.util
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "pipelines"))

from pipeline_config import FULL, INCREMENTAL, CDC, SCD2, WINDOW, FDC, VALID_MODES
from pipeline_utils import log

# ── Load pipeline modules (filenames start with digits → use importlib) ───────

def _load_module(filename: str):
    path = Path(__file__).parent / filename
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_bronze = _load_module("01_bronze_ingestion.py")
_silver = _load_module("02_silver_transform.py")
_gold   = _load_module("03_gold_aggregations.py")

LAYER_MODULES = {
    "bronze": _bronze,
    "silver": _silver,
    "gold":   _gold,
}
ALL_LAYERS = ["bronze", "silver", "gold"]


# ═══════════════════════════════════════════════════════════════════════════════
#  Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

def run(
    layers:      list[str] = None,
    mode:        str       = FULL,
    bronze_mode: str       = None,
    silver_mode: str       = None,
    gold_mode:   str       = None,
    tables:      list[str] = None,
    window_days: int       = 7,
    dry_run:     bool      = False,
) -> dict:
    """
    Orchestrate the full medallion pipeline.

    Parameters
    ----------
    layers      : layers to run, e.g. ["bronze","silver"] (default: all three)
    mode        : default load mode for all layers
    bronze_mode : override mode for bronze layer
    silver_mode : override mode for silver layer
    gold_mode   : override mode for gold layer
    tables      : restrict to specific source tables (bronze + silver only)
    window_days : days for WINDOW mode
    dry_run     : print plan without executing
    """
    layers = [l.lower() for l in (layers or ALL_LAYERS)]
    unknown = [l for l in layers if l not in LAYER_MODULES]
    if unknown:
        raise ValueError(f"Unknown layer(s): {unknown}")

    # Resolve per-layer modes
    layer_modes = {
        "bronze": (bronze_mode or mode).upper(),
        "silver": (silver_mode or mode).upper(),
        "gold":   (gold_mode   or mode).upper(),
    }
    for lyr, m in layer_modes.items():
        if m not in VALID_MODES:
            raise ValueError(f"Invalid mode for {lyr}: {m!r}. Valid: {VALID_MODES}")

    t_pipeline_start = time.time()

    print()
    print("╔" + "═" * 68 + "╗")
    print("║  MEDICAL BILLING PIPELINE  —  Medallion Architecture" + " " * 15 + "║")
    print("╠" + "═" * 68 + "╣")
    print(f"║  Started   : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC" + " " * 33 + "║")
    print(f"║  Layers    : {', '.join(layers):<55}║")
    for lyr in layers:
        print(f"║  {lyr.capitalize():<10}: mode={layer_modes[lyr]:<10} window_days={window_days:<5}" + " " * 22 + "║")
    if tables:
        print(f"║  Tables    : {', '.join(tables):<55}║")
    if dry_run:
        print("║  *** DRY RUN — no data will be written ***" + " " * 25 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    if dry_run:
        print("Dry-run plan:")
        for lyr in layers:
            print(f"  [{lyr.upper()}]  mode={layer_modes[lyr]}  tables={tables or 'ALL'}")
        return {}

    results = {}

    for lyr in layers:
        m   = layer_modes[lyr]
        mod = LAYER_MODULES[lyr]

        print(f"\n{'─' * 70}")
        print(f"  LAYER: {lyr.upper()}  │  MODE: {m}")
        print(f"{'─' * 70}")

        t0 = time.time()
        try:
            if lyr == "gold":
                # Gold tables are named differently from source tables
                gold_tables = None
                if tables:
                    # Map source table names to gold table names where possible
                    gold_tables = [t for t in _gold.ALL_GOLD_TABLES
                                   if any(src in t for src in tables)] or None
                stats = mod.run(mode=m, tables=gold_tables, window_days=window_days)
            else:
                stats = mod.run(mode=m, tables=tables, window_days=window_days)

            elapsed = time.time() - t0
            ok   = sum(1 for s in stats if s.get("status","").startswith("OK"))
            skip = sum(1 for s in stats if s.get("status","").startswith("SKIP"))
            err  = sum(1 for s in stats if s.get("status","").startswith("ERROR"))
            results[lyr] = {
                "status":  "OK" if err == 0 else f"PARTIAL ({err} errors)",
                "elapsed": elapsed,
                "tables":  len(stats),
                "ok":      ok,
                "skip":    skip,
                "errors":  err,
            }
            print(f"\n  ✓ {lyr.upper()} complete in {elapsed:.1f}s  "
                  f"({ok} OK, {skip} skipped, {err} errors)")

        except Exception as exc:
            elapsed = time.time() - t0
            log.error("Layer %s FAILED: %s", lyr, exc, exc_info=True)
            results[lyr] = {
                "status":  f"FAILED: {exc}",
                "elapsed": elapsed,
                "tables":  0,
                "ok":      0,
                "errors":  1,
            }
            print(f"\n  ✗ {lyr.upper()} FAILED after {elapsed:.1f}s: {exc}")
            print("  Pipeline halted.")
            break

    # ── Final summary ─────────────────────────────────────────────────────────
    total_elapsed = time.time() - t_pipeline_start
    print()
    print("╔" + "═" * 68 + "╗")
    print("║  PIPELINE COMPLETE" + " " * 49 + "║")
    print("╠" + "═" * 68 + "╣")
    for lyr, r in results.items():
        icon = "✓" if r["status"].startswith("OK") else "✗"
        line = f"  {icon}  {lyr.upper():<8}  {r['status']:<30}  {r['elapsed']:.1f}s"
        print(f"║{line:<68}║")
    print("╠" + "═" * 68 + "╣")
    print(f"║  Total elapsed: {total_elapsed:.1f}s" + " " * 50 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(
        description="Medical Billing Medallion Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full reload of everything
  python run_pipeline.py

  # Incremental load for all layers
  python run_pipeline.py --mode INCREMENTAL

  # Only bronze + silver, CDC mode
  python run_pipeline.py --layers bronze,silver --mode CDC

  # Different mode per layer
  python run_pipeline.py --bronze-mode FULL --silver-mode CDC --gold-mode INCREMENTAL

  # Rolling 7-day window
  python run_pipeline.py --mode WINDOW --window-days 7

  # Specific tables only
  python run_pipeline.py --tables claims,claim_lines,payments

  # Dry run
  python run_pipeline.py --mode INCREMENTAL --dry-run
        """,
    )
    p.add_argument("--layers",       default=None,
                   help="Comma-separated layers: bronze,silver,gold (default: all)")
    p.add_argument("--mode",         default=FULL,
                   choices=list(VALID_MODES),
                   help="Default load mode for all layers (default: FULL)")
    p.add_argument("--bronze-mode",  default=None, choices=list(VALID_MODES))
    p.add_argument("--silver-mode",  default=None, choices=list(VALID_MODES))
    p.add_argument("--gold-mode",    default=None, choices=list(VALID_MODES))
    p.add_argument("--tables",       default=None,
                   help="Comma-separated table names (default: all)")
    p.add_argument("--window-days",  type=int, default=7,
                   help="Days for WINDOW mode (default: 7)")
    p.add_argument("--dry-run",      action="store_true",
                   help="Print plan without executing")
    return p.parse_args()


if __name__ == "__main__":
    args   = _parse_args()
    layers = [l.strip() for l in args.layers.split(",")] if args.layers else None
    tables = [t.strip() for t in args.tables.split(",")] if args.tables else None

    results = run(
        layers      = layers,
        mode        = args.mode,
        bronze_mode = args.bronze_mode,
        silver_mode = args.silver_mode,
        gold_mode   = args.gold_mode,
        tables      = tables,
        window_days = args.window_days,
        dry_run     = args.dry_run,
    )

    failed = [l for l, r in results.items()
              if not r["status"].startswith("OK") and not r["status"].startswith("PARTIAL")]
    sys.exit(1 if failed else 0)
