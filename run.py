"""
End-to-end runner.

Usage:
    python run.py init        # create DB schema
    python run.py backfill    # full FRED history (run once)
    python run.py daily       # incremental pull + rebuild composite
    python run.py rebuild     # recompute signals + composite (no fetch)
    python run.py status      # latest reading
"""

from __future__ import annotations
import sys

from storage.db import init_db
from ingestion.fred_pull import pull_all as fred_pull_all
from ingestion.ecb_pull import pull_all as ecb_pull_all
from ingestion.eurostat_pull import pull_all as eurostat_pull_all
from compute.composite import (
    rebuild_signals, rebuild_composite, latest_snapshot,
    latest_inflation_levels, FED_INFLATION_TARGET,
)


def cmd_init():
    init_db()


def cmd_backfill():
    init_db()
    fred_pull_all(full=True)
    ecb_pull_all(full=True)
    eurostat_pull_all()
    rebuild_signals()
    rebuild_composite()


def cmd_daily():
    fred_pull_all(full=False)
    ecb_pull_all(full=False)
    eurostat_pull_all()
    rebuild_signals()
    rebuild_composite()


def cmd_rebuild():
    rebuild_signals()
    rebuild_composite()


def cmd_status():
    any_data = False
    for region in ["US", "EU", "GLOBAL"]:
        snap = latest_snapshot(region)
        if not snap:
            print(f"[{region}] No data yet.")
            continue
        any_data = True
        print(f"\n=== {region} Stress Snapshot — {snap['obs_date']} ===")
        print(f"  {'OVERALL':<18} {snap['overall']:+.2f}")
        print(f"  {'30-day ROC':<18} {snap['overall_roc_30d']:+.2f}  <-- leading signal")
        print("  By vector:")
        for v in ["liquidity", "credit", "labor", "consumer",
                  "housing", "banking", "inflation", "early_warning"]:
            val = snap.get(v)
            if val is None:
                continue
            flag = "  ALERT" if val > 1.5 else ""
            print(f"    {v:<16} {val:+.2f}{flag}")
    if not any_data:
        print("No composite data yet. Run `python run.py backfill` first.")
        return

    _print_inflation_alert()


def _print_inflation_alert():
    """Absolute-level inflation check vs the Fed's 2% target (US series)."""
    levels = latest_inflation_levels()
    if not levels:
        return
    worst = max(v["severity"] for v in levels.values())
    head = ("INFLATION ALERT" if worst >= 3
            else "INFLATION ABOVE TARGET" if worst >= 1
            else "INFLATION AT TARGET")
    as_of = max(v["date"] for v in levels.values())
    print(f"\n=== {head} (vs {FED_INFLATION_TARGET:.0f}% target) — {as_of} ===")
    for sid, v in levels.items():
        flag = "  <-- " + v["tier"] if v["severity"] >= 1 else ""
        print(f"    {v['label']:<14} {v['yoy']:+.1f}% YoY{flag}")


COMMANDS = {
    "init":     cmd_init,
    "backfill": cmd_backfill,
    "daily":    cmd_daily,
    "rebuild":  cmd_rebuild,
    "status":   cmd_status,
}


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(1)
    COMMANDS[sys.argv[1]]()
