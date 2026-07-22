"""
Eurostat ingestion.

Eurostat exposes a free REST API. The `eurostat` Python package wraps it.
Each dataset is identified by a code like 'une_rt_m' (unemployment rate, monthly).
Datasets contain multiple dimensions; we filter to the euro-area aggregate.

Filter conventions used here:
  - geo='EA20'   : euro area, 20 countries (current composition)
                   fallback to 'EA19' or 'EA' if EA20 not in the dataset
  - unit='PC_ACT' or 'I15' or 'RCH_A1' depending on dataset
  - s_adj='SCA' : seasonally adjusted, calendar adjusted (where applicable)

Rather than encoding all filters per series, we use a per-dataset adapter
function. New datasets = add an adapter, no other changes.
"""

from __future__ import annotations
import time
from datetime import date

import pandas as pd
import eurostat

from config.eu_series_catalog import EU_EUROSTAT_SERIES
from config.series_catalog import Series
from storage.db import upsert_observations, log_pull


# Preferred euro-area aggregate codes, in order. We try each until one matches
# the dataset's geo dimension.
EA_CODES_PREFERRED = ["EA21", "EA20", "EA19", "EA"]


def _pick_euro_area(df: pd.DataFrame) -> str | None:
    """Return the euro-area geo code present in the dataset, or None."""
    if "geo\\TIME_PERIOD" in df.columns:
        col = "geo\\TIME_PERIOD"
    elif "geo" in df.columns:
        col = "geo"
    else:
        return None
    geo_values = set(df[col].unique())
    for code in EA_CODES_PREFERRED:
        if code in geo_values:
            return code
    return None


def _melt_to_series(df: pd.DataFrame, geo: str) -> pd.Series:
    """
    Eurostat returns wide format: dimension columns + one column per period.
    Pick the euro-area row, melt period columns to long, return as time-indexed Series.
    """
    geo_col = "geo\\TIME_PERIOD" if "geo\\TIME_PERIOD" in df.columns else "geo"
    # Subset to euro-area rows
    sub = df[df[geo_col] == geo].copy()
    if sub.empty:
        return pd.Series(dtype=float)

    # Identify time-period columns (anything that parses as a date or YYYY-MM / YYYY)
    period_cols = []
    for c in sub.columns:
        if c == geo_col:
            continue
        # Period columns look like '2024M03', '2024Q1', '2024-03', or '2024'
        if any(ch.isdigit() for ch in str(c)) and len(str(c)) <= 8:
            period_cols.append(c)

    if not period_cols:
        return pd.Series(dtype=float)

    # If multiple rows match (multiple unit/freq/etc), prefer the one with most data
    if len(sub) > 1:
        sub["_nonnull"] = sub[period_cols].notna().sum(axis=1)
        sub = sub.sort_values("_nonnull", ascending=False).head(1)

    row = sub.iloc[0]
    values = {}
    for c in period_cols:
        v = row[c]
        if pd.isna(v):
            continue
        # Parse the period string into a date
        try:
            ts = _parse_period(str(c))
            values[ts] = float(v)
        except Exception:
            continue

    if not values:
        return pd.Series(dtype=float)

    s = pd.Series(values).sort_index()
    s.index = pd.to_datetime(s.index)
    return s


def _parse_period(p: str) -> pd.Timestamp:
    """Parse Eurostat period codes: '2024M03', '2024Q1', '2024-03', '2024'."""
    p = p.strip()
    if "M" in p:  # monthly: 2024M03
        year, month = p.split("M")
        return pd.Timestamp(int(year), int(month), 1) + pd.offsets.MonthEnd(0)
    if "Q" in p:  # quarterly: 2024Q1 or 2024-Q1
        clean = p.replace("-", "")  # normalise 2024-Q1 -> 2024Q1
        year, q = clean.split("Q")
        month = int(q) * 3
        return pd.Timestamp(int(year), month, 1) + pd.offsets.MonthEnd(0)
    if "-" in p:  # ISO-ish: 2024-03
        return pd.to_datetime(p) + pd.offsets.MonthEnd(0)
    # Yearly: 2024
    return pd.Timestamp(int(p), 12, 31)


def _fetch_dataset(code: str) -> pd.Series:
    """Fetch and reduce a Eurostat dataset to a single euro-area time series."""
    df = eurostat.get_data_df(code)
    if df is None or df.empty:
        raise RuntimeError(f"Eurostat returned empty dataset {code}")

    geo = _pick_euro_area(df)
    if geo is None:
        raise RuntimeError(f"No euro-area aggregate found in {code}")

    series = _melt_to_series(df, geo)
    if series.empty:
        raise RuntimeError(f"No usable observations in {code}")

    return series


def pull_series(s: Series) -> int:
    """
    Pull one Eurostat series. We always do a full fetch — Eurostat
    datasets are small enough (a few KB to a few MB) that incremental
    pulls aren't worth the complexity.
    """
    try:
        data = _fetch_dataset(s.series_id)
        rows = [
            (s.series_id, idx.date(), float(val), "eurostat")
            for idx, val in data.items()
            if pd.notna(val)
        ]
        n = upsert_observations(rows)
        log_pull("eurostat", s.series_id, n, success=True)
        return n
    except Exception as e:  # noqa: BLE001
        log_pull("eurostat", s.series_id, 0, success=False, error_msg=str(e))
        print(f"  [ERROR] {s.series_id}: {e}")
        return 0


def pull_all(throttle: float = 0.5) -> dict:
    """Pull every Eurostat series in the EU catalog."""
    results = {}
    total = len(EU_EUROSTAT_SERIES)
    print(f"Pulling {total} Eurostat datasets...")

    for i, s in enumerate(EU_EUROSTAT_SERIES, 1):
        n = pull_series(s)
        results[s.series_id] = n
        print(f"  [{i:>2}/{total}] {s.series_id:<18} {n:>6,} rows  ({s.description})")
        time.sleep(throttle)

    print(f"\nEurostat pull complete. Rows written: {sum(results.values()):,}")
    return results


if __name__ == "__main__":
    pull_all()
