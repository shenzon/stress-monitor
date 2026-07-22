"""
FRED ingestion.

Two pulls:
  - backfill(): full history, run once at setup
  - incremental(): pull only since last observation in DB, run daily

Rate limit: FRED allows 120 req/min on free tier. We have ~45 series.
A full run takes <30 seconds. Add throttle if expanded.

Auth: FRED_API_KEY env var. Get free key at https://fred.stlouisfed.org/docs/api/api_key.html
"""

from __future__ import annotations
import os
import time
from datetime import date, timedelta

import pandas as pd
from fredapi import Fred

from config.series_catalog import US_FRED_SERIES, Series
from storage.db import upsert_observations, log_pull, get_conn


def _client() -> Fred:
    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise RuntimeError(
            "FRED_API_KEY not set. Get one free at "
            "https://fred.stlouisfed.org/docs/api/api_key.html"
        )
    return Fred(api_key=key)


def _latest_date(series_id: str) -> date | None:
    """Most recent obs_date already stored, or None if series is empty."""
    with get_conn(read_only=True) as conn:
        row = conn.execute(
            "SELECT MAX(obs_date) FROM observations WHERE series_id = ?",
            (series_id,),
        ).fetchone()
    return row[0] if row and row[0] else None


def _fetch_series(fred: Fred, s: Series, start: date | None) -> pd.Series:
    """Pull a single series from FRED with optional start date."""
    kwargs = {}
    if start:
        kwargs["observation_start"] = start.isoformat()
    return fred.get_series(s.series_id, **kwargs)


def pull_series(s: Series, fred: Fred, full: bool = False) -> int:
    """Pull one series. Returns number of rows written."""
    try:
        if full:
            start = None
        else:
            last = _latest_date(s.series_id)
            # Re-pull last 14 days to catch any revisions
            start = (last - timedelta(days=14)) if last else None

        data = _fetch_series(fred, s, start)
        if data is None or len(data) == 0:
            log_pull("fred", s.series_id, 0, success=True)
            return 0

        # Drop NaN, convert to row tuples
        data = data.dropna()
        rows = [
            (s.series_id, idx.date(), float(val), "fred")
            for idx, val in data.items()
        ]
        n = upsert_observations(rows)
        log_pull("fred", s.series_id, n, success=True)
        return n

    except Exception as e:  # noqa: BLE001
        log_pull("fred", s.series_id, 0, success=False, error_msg=str(e))
        print(f"  [ERROR] {s.series_id}: {e}")
        return 0


def pull_all(full: bool = False, throttle: float = 0.3) -> dict:
    """Pull every FRED series in catalog. throttle = seconds between requests."""
    fred = _client()
    results = {}
    total = len(US_FRED_SERIES)

    print(f"Pulling {total} FRED series ({'FULL backfill' if full else 'incremental'})...")
    for i, s in enumerate(US_FRED_SERIES, 1):
        n = pull_series(s, fred, full=full)
        results[s.series_id] = n
        print(f"  [{i:>2}/{total}] {s.series_id:<20} {n:>6,} rows  ({s.description})")
        time.sleep(throttle)

    print(f"\nDone. Total rows written: {sum(results.values()):,}")
    return results


if __name__ == "__main__":
    import sys
    full = "--full" in sys.argv
    pull_all(full=full)
