"""
ECB Data Portal ingestion via direct REST API.

Endpoint: https://data-api.ecb.europa.eu/service/data/{dataflow}/{key}
Format:   CSV (cleaner than SDMX-XML, no SDMX library needed)

Series IDs in our catalog are dot-separated:
    'CISS.D.U2.Z0Z.4F.EC.SS_CIN.IDX'
The first segment is the dataflow ID, the rest is the key.

Why direct REST not pandasdmx:
  pandasdmx 1.6 still defaults to ECB's legacy `sdw-wsrest` endpoint which
  is being deprecated. The modern `data-api` endpoint is stable, returns
  clean CSV, and removes a dependency.

Auth: none required.
"""

from __future__ import annotations
import time
import io
from datetime import date, timedelta

import pandas as pd
import requests

from config.eu_series_catalog import EU_ECB_SERIES
from config.series_catalog import Series
from storage.db import upsert_observations, log_pull, get_conn


ECB_BASE = "https://data-api.ecb.europa.eu/service/data"
HEADERS = {"Accept": "text/csv"}
TIMEOUT = 30


def _latest_date(series_id: str) -> date | None:
    with get_conn(read_only=True) as conn:
        row = conn.execute(
            "SELECT MAX(obs_date) FROM observations WHERE series_id = ?",
            (series_id,),
        ).fetchone()
    return row[0] if row and row[0] else None


def _split_key(series_id: str) -> tuple[str, str]:
    """'CISS.D.U2.Z0Z.4F.EC.SS_CIN.IDX' -> ('CISS', 'D.U2.Z0Z.4F.EC.SS_CIN.IDX')."""
    parts = series_id.split(".", 1)
    if len(parts) != 2:
        raise ValueError(f"Bad ECB series id: {series_id}")
    return parts[0], parts[1]


def _fetch_series(s: Series, start: date | None) -> pd.Series:
    """Fetch one ECB series as CSV, return as date-indexed Series."""
    dataflow, key = _split_key(s.series_id)
    url = f"{ECB_BASE}/{dataflow}/{key}"
    params = {"format": "csvdata"}
    if start:
        params["startPeriod"] = start.isoformat()

    r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()

    if not r.text.strip():
        return pd.Series(dtype=float)

    df = pd.read_csv(io.StringIO(r.text))

    # ECB CSV always has TIME_PERIOD and OBS_VALUE columns
    if "TIME_PERIOD" not in df.columns or "OBS_VALUE" not in df.columns:
        raise RuntimeError(
            f"Unexpected ECB CSV columns for {s.series_id}: {list(df.columns)[:10]}"
        )

    df = df[["TIME_PERIOD", "OBS_VALUE"]].dropna()
    if df.empty:
        return pd.Series(dtype=float)

    df["TIME_PERIOD"] = df["TIME_PERIOD"].astype(str)

    # ECB period codes can be: '2024-03-15' (daily), '2024-03' (monthly),
    # '2024-W12' (weekly), '2024-Q1' (quarterly), '2024' (annual)
    df["date"] = df["TIME_PERIOD"].apply(_parse_ecb_period)
    df = df.dropna(subset=["date"]).sort_values("date")

    return pd.Series(
        data=pd.to_numeric(df["OBS_VALUE"], errors="coerce").values,
        index=pd.DatetimeIndex(df["date"].values),
    ).dropna()


def _parse_ecb_period(p: str) -> pd.Timestamp | None:
    """Parse ECB period codes into end-of-period timestamps."""
    p = p.strip()
    try:
        if "W" in p:  # weekly: '2024-W12'
            year, week = p.split("-W")
            # ISO week -> Friday (end of business week)
            return pd.Timestamp.fromisocalendar(int(year), int(week), 5)
        if "Q" in p:  # quarterly: '2024-Q1'
            year, q = p.split("-Q")
            month = int(q) * 3
            return pd.Timestamp(int(year), month, 1) + pd.offsets.MonthEnd(0)
        # Daily, monthly, or yearly — pandas handles them
        ts = pd.to_datetime(p)
        if len(p) == 7:  # 'YYYY-MM' -> month-end
            ts = ts + pd.offsets.MonthEnd(0)
        elif len(p) == 4:  # 'YYYY' -> year-end
            ts = pd.Timestamp(int(p), 12, 31)
        return ts
    except Exception:
        return None


def pull_series(s: Series, full: bool = False) -> int:
    """Pull one ECB series. Returns rows written."""
    try:
        if full:
            start = None
        else:
            last = _latest_date(s.series_id)
            start = (last - timedelta(days=30)) if last else None

        data = _fetch_series(s, start)
        if data is None or len(data) == 0:
            log_pull("ecb", s.series_id, 0, success=True)
            return 0

        rows = [
            (s.series_id, idx.date(), float(val), "ecb")
            for idx, val in data.items()
            if pd.notna(val)
        ]
        n = upsert_observations(rows)
        log_pull("ecb", s.series_id, n, success=True)
        return n

    except Exception as e:  # noqa: BLE001
        log_pull("ecb", s.series_id, 0, success=False, error_msg=str(e))
        print(f"  [ERROR] {s.series_id}: {type(e).__name__}: {str(e)[:120]}")
        return 0


def pull_all(full: bool = False, throttle: float = 0.5) -> dict:
    """Pull every ECB series in the EU catalog."""
    results = {}
    total = len(EU_ECB_SERIES)
    print(f"Pulling {total} ECB series ({'FULL' if full else 'incremental'})...")

    for i, s in enumerate(EU_ECB_SERIES, 1):
        n = pull_series(s, full=full)
        results[s.series_id] = n
        short = s.series_id if len(s.series_id) <= 28 else s.series_id[:25] + "..."
        print(f"  [{i:>2}/{total}] {short:<28} {n:>6,} rows  ({s.description[:40]})")
        time.sleep(throttle)

    print(f"\nECB pull complete. Rows written: {sum(results.values()):,}")
    return results


if __name__ == "__main__":
    import sys
    full = "--full" in sys.argv
    pull_all(full=full)
