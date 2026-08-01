"""
Composite stress scoring — region-aware.

Now produces composites per region:
  - US     (FRED series only)
  - EU     (ECB + Eurostat series only)
  - GLOBAL (combined, weighted: 60% US / 40% EU)

The 60/40 weighting reflects that US capital markets remain the dominant
shock transmitter even though EU has comparable economic mass — most
stress originating in EU still hits US risk assets via the same channels,
but US-originated stress propagates harder.

Per series:
  raw value -> transform (yoy/mom/level/...) -> rolling z-score (5y)
            -> direction-adjusted (positive = stress) -> stored in `signals`

Per (date, region):
  weighted average of series z-scores within each vector
  -> vector scores
  -> weighted vector avg = overall stress score
  -> 30-day rate-of-change of overall = the leading signal

Why a separate region key not just stuff them in one global pool:
  Regimes can diverge. EU recession with US strength is real (2011-13,
  partly 2024). Separating regions surfaces the divergence explicitly
  rather than averaging it away.
"""

from __future__ import annotations
import pandas as pd
import numpy as np

from config.series_catalog import (
    VECTORS, Series, all_series, by_vector,
)
from storage.db import get_conn


# Vector weights — same across regions for now. Tune per-region after observation.
VECTOR_WEIGHTS: dict[str, float] = {
    "liquidity":     0.18,
    "credit":        0.18,
    "labor":         0.14,
    "consumer":      0.09,
    "housing":       0.09,
    "banking":       0.14,
    "inflation":     0.10,
    "early_warning": 0.08,
}
assert abs(sum(VECTOR_WEIGHTS.values()) - 1.0) < 1e-9


# Region weights for the GLOBAL composite
GLOBAL_REGION_WEIGHTS: dict[str, float] = {
    "US": 0.60,
    "EU": 0.40,
}


# =============================================================================
# Frequency inference
# =============================================================================

def infer_periods_per_year(index: pd.DatetimeIndex) -> int:
    """Infer periods-per-year from median gap between observations."""
    if len(index) < 2:
        return 252
    median_days = pd.Series(index).diff().dropna().dt.days.median()
    if median_days <= 3:   return 252  # daily
    if median_days <= 10:  return 52   # weekly
    if median_days <= 35:  return 12   # monthly
    if median_days <= 100: return 4    # quarterly
    return 1                           # annual


# =============================================================================
# Transforms
# =============================================================================

def apply_transform(s: pd.Series, transform: str, ppy: int = 252) -> pd.Series:
    if transform == "level":
        return s
    if transform == "inverse_level":
        return -s
    if transform == "yoy":
        return s.pct_change(ppy) * 100
    if transform == "mom":
        return s.pct_change(max(1, ppy // 12)) * 100
    if transform == "roc_3m":
        return s.pct_change(max(1, ppy // 4)) * 100
    if transform == "diff":
        return s.diff()
    raise ValueError(f"Unknown transform: {transform}")


def rolling_zscore(s: pd.Series, lookback_years: int = 5, ppy: int = 252) -> pd.Series:
    window = lookback_years * ppy
    min_periods = max(20, window // 4)
    mean = s.rolling(window, min_periods=min_periods).mean()
    std = s.rolling(window, min_periods=min_periods).std()
    z = (s - mean) / std
    return z.replace([np.inf, -np.inf], np.nan)


# =============================================================================
# Per-series signal computation
# =============================================================================

def compute_signal_for_series(spec: Series) -> pd.DataFrame | None:
    with get_conn(read_only=True) as conn:
        df = conn.execute(
            "SELECT obs_date, value FROM observations "
            "WHERE series_id = ? ORDER BY obs_date",
            (spec.series_id,),
        ).df()

    if df.empty:
        return None

    df["obs_date"] = pd.to_datetime(df["obs_date"])
    df = df.set_index("obs_date").sort_index()

    ppy = infer_periods_per_year(df.index)
    raw = df["value"]
    transformed = apply_transform(raw, spec.transform, ppy=ppy)
    z = rolling_zscore(transformed, spec.lookback_years, ppy=ppy)
    z_adj = z * spec.direction

    out = pd.DataFrame({
        "series_id":   spec.series_id,
        "raw_value":   raw,
        "transformed": transformed,
        "z_score":     z_adj,
        "vector":      spec.vector,
        "region":      spec.region,
    })
    out.index.name = "obs_date"
    return out.reset_index().dropna(subset=["z_score"])


def rebuild_signals() -> int:
    """Recompute signals across all regions."""
    print("Computing per-series signals (US + EU)...")
    frames = []
    for spec in all_series():
        df = compute_signal_for_series(spec)
        if df is not None and not df.empty:
            frames.append(df)
            print(f"  [{spec.region}] {spec.series_id[:30]:<30} {len(df):>6,} obs")

    if not frames:
        print("No signals to write.")
        return 0

    all_signals = pd.concat(frames, ignore_index=True)
    all_signals["obs_date"] = pd.to_datetime(all_signals["obs_date"]).dt.date

    with get_conn() as conn:
        conn.execute("DELETE FROM signals")
        conn.register("df_signals", all_signals)
        conn.execute("""
            INSERT INTO signals
                (series_id, obs_date, raw_value, transformed, z_score, vector, region)
            SELECT series_id, obs_date, raw_value, transformed, z_score, vector, region
            FROM df_signals
        """)
    print(f"Wrote {len(all_signals):,} signal rows.")
    return len(all_signals)


# =============================================================================
# Composite per region
# =============================================================================

def _vector_weights_map(region: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for v in VECTORS:
        members = by_vector(v, region=region)
        total = sum(s.weight for s in members)
        if total <= 0:
            out[v] = {}
            continue
        out[v] = {s.series_id: (s.weight / total) for s in members}
    return out


def _build_region_composite(region: str) -> pd.DataFrame:
    with get_conn(read_only=True) as conn:
        sig = conn.execute(
            "SELECT obs_date, series_id, z_score, vector "
            "FROM signals WHERE region = ?",
            (region,),
        ).df()

    if sig.empty:
        return pd.DataFrame()

    sig["obs_date"] = pd.to_datetime(sig["obs_date"])
    weights = _vector_weights_map(region)

    wide = sig.pivot_table(
        index="obs_date", columns="series_id", values="z_score", aggfunc="last"
    ).sort_index()

    full_idx = pd.date_range(wide.index.min(), wide.index.max(), freq="D")
    wide = wide.reindex(full_idx).ffill()

    vector_scores = pd.DataFrame(index=wide.index)
    for v in VECTORS:
        members = weights.get(v, {})
        cols = [c for c in members if c in wide.columns]
        if not cols:
            vector_scores[v] = np.nan
            continue
        w = np.array([members[c] for c in cols])
        sub = wide[cols]
        mask = sub.notna().astype(float)
        w_matrix = mask.values * w
        row_sums = w_matrix.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            w_norm = np.where(row_sums > 0, w_matrix / row_sums, 0)
        weighted_vals = np.nansum(sub.values * w_norm, axis=1)
        weighted_vals = np.where(row_sums.flatten() > 0, weighted_vals, np.nan)
        vector_scores[v] = weighted_vals

    overall = pd.Series(0.0, index=wide.index)
    weight_sum = pd.Series(0.0, index=wide.index)
    for v in VECTORS:
        w = VECTOR_WEIGHTS[v]
        overall += vector_scores[v].fillna(0) * w
        weight_sum += vector_scores[v].notna().astype(float) * w
    overall = overall / weight_sum.replace(0, np.nan)

    overall_roc_30d = overall.diff(30)

    df = pd.DataFrame({
        "obs_date": vector_scores.index.date,
        "region": region,
        **{v: vector_scores[v].values for v in VECTORS},
        "overall": overall.values,
        "overall_roc_30d": overall_roc_30d.values,
    })
    return df.dropna(subset=["overall"])


def _build_global_composite(us_df: pd.DataFrame, eu_df: pd.DataFrame) -> pd.DataFrame:
    if us_df.empty and eu_df.empty:
        return pd.DataFrame()

    us = us_df.set_index("obs_date") if not us_df.empty else pd.DataFrame()
    eu = eu_df.set_index("obs_date") if not eu_df.empty else pd.DataFrame()

    if not us.empty and not eu.empty:
        idx = us.index.intersection(eu.index)
        if len(idx) == 0:
            # Fall back to union if no overlap
            idx = us.index.union(eu.index)
    elif not us.empty:
        idx = us.index
    else:
        idx = eu.index

    w_us = GLOBAL_REGION_WEIGHTS["US"]
    w_eu = GLOBAL_REGION_WEIGHTS["EU"]

    cols = list(VECTORS) + ["overall"]
    out = pd.DataFrame(index=idx)
    for c in cols:
        us_vals = us[c].reindex(idx) if not us.empty else pd.Series(np.nan, index=idx)
        eu_vals = eu[c].reindex(idx) if not eu.empty else pd.Series(np.nan, index=idx)

        us_mask = us_vals.notna().astype(float) * w_us
        eu_mask = eu_vals.notna().astype(float) * w_eu
        denom = us_mask + eu_mask
        with np.errstate(invalid="ignore", divide="ignore"):
            combined = (us_vals.fillna(0) * us_mask + eu_vals.fillna(0) * eu_mask) / denom.replace(0, np.nan)
        out[c] = combined

    out["region"] = "GLOBAL"
    out = out.reset_index().rename(columns={"index": "obs_date"})
    out = out.sort_values("obs_date").reset_index(drop=True)
    out["overall_roc_30d"] = out["overall"].diff(30)
    return out.dropna(subset=["overall"])


def rebuild_composite() -> int:
    print("\nBuilding region composites...")

    us_df = _build_region_composite("US")
    print(f"  US:     {len(us_df):,} rows")

    eu_df = _build_region_composite("EU")
    print(f"  EU:     {len(eu_df):,} rows")

    global_df = _build_global_composite(us_df, eu_df)
    print(f"  GLOBAL: {len(global_df):,} rows")

    parts = [df for df in [us_df, eu_df, global_df] if not df.empty]
    if not parts:
        print("Nothing to write.")
        return 0
    combined = pd.concat(parts, ignore_index=True)
    combined["obs_date"] = pd.to_datetime(combined["obs_date"]).dt.date

    combined = combined[[
        "obs_date", "region",
        "liquidity", "credit", "labor", "consumer",
        "housing", "banking", "inflation", "early_warning",
        "overall", "overall_roc_30d",
    ]]

    with get_conn() as conn:
        conn.execute("DELETE FROM composite")
        conn.register("df_comp", combined)
        conn.execute("""
            INSERT INTO composite
                (obs_date, region, liquidity, credit, labor, consumer,
                 housing, banking, inflation, early_warning, overall, overall_roc_30d)
            SELECT obs_date, region, liquidity, credit, labor, consumer,
                   housing, banking, inflation, early_warning, overall, overall_roc_30d
            FROM df_comp
        """)
    print(f"Wrote {len(combined):,} composite rows across {combined['region'].nunique()} regions.")
    return len(combined)


# =============================================================================
# Absolute inflation levels (independent of z-score normalization)
# =============================================================================

# The 5yr z-score normalizes against the 2021-23 inflation era, so it can read
# "easing" while YoY prints are still above the Fed's 2% target. These helpers
# alert on the ABSOLUTE level. Shared by the dashboard and `run.py status`.
ABS_INFLATION_SERIES: dict[str, str] = {
    "PCEPILFE": "Core PCE",
    "CPILFESL": "Core CPI",
    "CPIAUCSL": "Headline CPI",
}
FED_INFLATION_TARGET = 2.0


def inflation_tier(yoy: float | None) -> tuple[str, int]:
    """(label, severity) for an absolute YoY % print. Higher severity = worse."""
    if yoy is None or (isinstance(yoy, float) and np.isnan(yoy)):
        return ("N/A", -1)
    if yoy >= 4.0:                    return ("SEVERE", 4)
    if yoy >= 3.0:                    return ("HOT", 3)
    if yoy >= 2.5:                    return ("ELEVATED", 2)
    if yoy >= FED_INFLATION_TARGET:   return ("ABOVE TARGET", 1)
    return ("AT/BELOW TARGET", 0)


def latest_inflation_levels() -> dict[str, dict]:
    """Latest absolute YoY % for each tracked core/headline inflation series.

    Reads `transformed` (the YoY transform output) from `signals`, independent
    of the direction-adjusted z-score.
    """
    out: dict[str, dict] = {}
    with get_conn(read_only=True) as conn:
        for sid, label in ABS_INFLATION_SERIES.items():
            row = conn.execute(
                "SELECT obs_date, transformed FROM signals "
                "WHERE series_id = ? AND transformed IS NOT NULL "
                "ORDER BY obs_date DESC LIMIT 1",
                (sid,),
            ).fetchone()
            if row:
                label_, sev = inflation_tier(row[1])
                out[sid] = {"label": label, "date": row[0], "yoy": row[1],
                            "tier": label_, "severity": sev}
    return out


def latest_snapshot(region: str = "US") -> dict:
    with get_conn(read_only=True) as conn:
        result = conn.execute(
            "SELECT * FROM composite WHERE region = ? "
            "ORDER BY obs_date DESC LIMIT 1",
            (region,),
        )
        row = result.fetchone()
        cols = [d[0] for d in result.description] if row else []
    return dict(zip(cols, row)) if row else {}


def rebuild_all() -> None:
    rebuild_signals()
    rebuild_composite()
    for region in ["US", "EU", "GLOBAL"]:
        snap = latest_snapshot(region)
        if snap:
            print(f"\n=== {region} latest reading ===")
            for k, v in snap.items():
                if isinstance(v, float):
                    print(f"  {k:<18} {v:+.2f}")
                else:
                    print(f"  {k:<18} {v}")


if __name__ == "__main__":
    rebuild_all()
