"""
Series Catalog — Master registry of every economic series tracked.

Each entry defines:
  - source: which API to pull from (fred for now; ecb/eurostat added later)
  - vector: which stress category it belongs to
  - direction: +1 = higher value means more stress, -1 = lower means more stress
  - transform: 'level' | 'yoy' | 'mom' | 'roc_3m' | 'diff' — how to convert raw to stress signal
  - lookback_years: window for z-score normalization
  - weight: relative weight inside its vector (vector weights are in compute/composite.py)

Adding a series = adding one row here. The pipeline picks it up automatically.
"""

from dataclasses import dataclass
from typing import Literal

Vector = Literal[
    "liquidity",
    "credit",
    "labor",
    "consumer",
    "housing",
    "banking",
    "inflation",
    "early_warning",
]

Transform = Literal["level", "yoy", "mom", "roc_3m", "diff", "inverse_level"]


@dataclass(frozen=True)
class Series:
    series_id: str
    source: str            # 'fred', 'ecb', 'eurostat', 'yfinance'
    vector: Vector
    description: str
    direction: int         # +1 = stress when high, -1 = stress when low
    transform: Transform
    lookback_years: int = 5
    weight: float = 1.0
    region: str = "US"


# =============================================================================
# US — FRED
# =============================================================================

US_FRED_SERIES: list[Series] = [
    # --- LIQUIDITY ---
    Series("WALCL",        "fred", "liquidity", "Fed Balance Sheet (Total Assets)",          direction=-1, transform="level",   weight=1.0),
    Series("RRPONTSYD",    "fred", "liquidity", "Overnight Reverse Repo",                     direction=+1, transform="level",   weight=0.8),
    Series("WTREGEN",      "fred", "liquidity", "Treasury General Account",                   direction=+1, transform="level",   weight=0.8),
    Series("FEDFUNDS",     "fred", "liquidity", "Federal Funds Effective Rate",               direction=+1, transform="level",   weight=0.4),
    Series("SOFR",         "fred", "liquidity", "Secured Overnight Financing Rate",           direction=+1, transform="level",   weight=0.6),

    # --- CREDIT ---
    Series("BAA10Y",         "fred", "credit", "Moody's Baa Spread over 10Y Treasury",        direction=+1, transform="level",   weight=1.0),
    Series("BUSLOANS",       "fred", "credit", "C&I Loans, All Commercial Banks",             direction=-1, transform="yoy",     weight=0.8),
    Series("DRCCLACBS",      "fred", "credit", "Credit Card Delinquency Rate",                direction=+1, transform="level",   weight=0.9),
    Series("DRTSCILM",       "fred", "credit", "SLOOS: Tightening C&I to Large/Medium Firms", direction=+1, transform="level",   weight=1.0),
    Series("DRTSCLCC",       "fred", "credit", "SLOOS: Tightening Credit Card Standards",     direction=+1, transform="level",   weight=0.7),
    Series("DRSDCIS",        "fred", "credit", "SLOOS: Stronger Demand for C&I Loans",        direction=-1, transform="level",   weight=0.5),

    # --- LABOR ---
    Series("IC4WSA",       "fred", "labor", "Initial Claims, 4-week MA",                      direction=+1, transform="level",   weight=1.0),
    Series("CCSA",         "fred", "labor", "Continuing Claims",                              direction=+1, transform="level",   weight=0.9),
    Series("SAHMREALTIME", "fred", "labor", "Sahm Rule Recession Indicator",                  direction=+1, transform="level",   weight=1.2),
    Series("JTSQUR",       "fred", "labor", "JOLTS Quits Rate",                               direction=-1, transform="level",   weight=0.8),
    Series("JTSJOR",       "fred", "labor", "JOLTS Job Openings Rate",                        direction=-1, transform="level",   weight=0.7),
    Series("TEMPHELPS",    "fred", "labor", "Temporary Help Services Employment",             direction=-1, transform="yoy",     weight=0.9),
    Series("AWHMAN",       "fred", "labor", "Avg Weekly Hours, Manufacturing",                direction=-1, transform="level",   weight=0.6),
    Series("UNRATE",       "fred", "labor", "Unemployment Rate",                              direction=+1, transform="level",   weight=0.7),

    # --- CONSUMER ---
    Series("RSXFS",        "fred", "consumer", "Retail Sales ex Food Services",               direction=-1, transform="yoy",     weight=1.0),
    Series("W875RX1",      "fred", "consumer", "Real Personal Income ex Transfers",           direction=-1, transform="yoy",     weight=0.9),
    Series("PSAVERT",      "fred", "consumer", "Personal Savings Rate",                       direction=-1, transform="level",   weight=0.6),
    Series("REVOLSL",      "fred", "consumer", "Revolving Credit Outstanding",                direction=+1, transform="yoy",     weight=0.7),
    Series("UMCSENT",      "fred", "consumer", "U Michigan Consumer Sentiment",               direction=-1, transform="level",   weight=0.7),
    Series("CSCICP03USM665S", "fred", "consumer", "Conference Board Consumer Confidence",     direction=-1, transform="level",   weight=0.7),

    # --- HOUSING ---
    Series("HSN1F",        "fred", "housing", "New One Family Houses Sold",                   direction=-1, transform="yoy",     weight=1.0),
    Series("MSACSR",       "fred", "housing", "Monthly Supply of New Houses",                 direction=+1, transform="level",   weight=0.8),
    Series("MORTGAGE30US", "fred", "housing", "30-Year Fixed Mortgage Rate",                  direction=+1, transform="level",   weight=0.5),
    Series("HOUST",        "fred", "housing", "Housing Starts",                               direction=-1, transform="yoy",     weight=0.8),
    Series("PERMIT",       "fred", "housing", "Building Permits",                             direction=-1, transform="yoy",     weight=0.7),
    Series("DRSREACBS",    "fred", "housing", "Single-Family Mortgage Delinquency",           direction=+1, transform="level",   weight=1.0),
    Series("DRCRELEXFACBS","fred", "housing", "CRE Delinquency Rate (CMBS proxy)",            direction=+1, transform="level",   weight=1.1),

    # --- BANKING ---
    Series("DPSACBW027SBOG", "fred", "banking", "Deposits, All Commercial Banks",             direction=-1, transform="yoy",     weight=1.0),
    Series("WLCFLPCL",       "fred", "banking", "Discount Window Primary Credit",             direction=+1, transform="level",   weight=0.8),
    Series("DRALACBS",       "fred", "banking", "All Loans Delinquency Rate",                 direction=+1, transform="level",   weight=0.9),
    Series("DRBLACBS",       "fred", "banking", "Charge-Off Rate, All Loans",                 direction=+1, transform="level",   weight=0.7),

    # --- INFLATION ---
    # Price-level stress. Closes the structural blind spot: the conditions
    # vectors above can read "easing" while CPI runs hot (e.g. an oil shock).
    # All direction=+1 — higher inflation / expectations = more stress.
    Series("CPIAUCSL",   "fred", "inflation", "CPI All Urban Consumers (headline, YoY)",     direction=+1, transform="yoy",   weight=1.0),
    Series("CPILFESL",   "fred", "inflation", "Core CPI ex Food & Energy (YoY)",             direction=+1, transform="yoy",   weight=1.0),
    Series("PCEPILFE",   "fred", "inflation", "Core PCE Price Index (Fed's target, YoY)",    direction=+1, transform="yoy",   weight=1.1),
    Series("CPIENGSL",   "fred", "inflation", "CPI Energy (oil-shock pass-through, YoY)",     direction=+1, transform="yoy",   weight=0.6),
    Series("CES0500000003","fred","inflation", "Avg Hourly Earnings, Private (wage infl, YoY)",direction=+1, transform="yoy",   weight=0.7),
    Series("T10YIE",     "fred", "inflation", "10Y Breakeven Inflation Expectations",        direction=+1, transform="level", weight=0.7),
    Series("MICH",       "fred", "inflation", "UMich 1Y Inflation Expectations",             direction=+1, transform="level", weight=0.6),

    # --- EARLY WARNING / COMPOSITE ---
    Series("T10Y3M",       "fred", "early_warning", "10Y-3M Yield Curve",                     direction=-1, transform="level",   weight=1.0),
    Series("T10Y2Y",       "fred", "early_warning", "10Y-2Y Yield Curve",                     direction=-1, transform="level",   weight=0.7),
    Series("NFCI",         "fred", "early_warning", "Chicago Fed NFCI",                       direction=+1, transform="level",   weight=1.0),
    Series("ANFCI",        "fred", "early_warning", "Chicago Fed Adjusted NFCI",              direction=+1, transform="level",   weight=0.8),
    Series("STLFSI4",      "fred", "early_warning", "St Louis Fed Financial Stress Index",    direction=+1, transform="level",   weight=1.0),
    Series("USSLIND",      "fred", "early_warning", "Conference Board Leading Index (state-level proxy)", direction=-1, transform="yoy", weight=0.8),
]


# =============================================================================
# Helpers
# =============================================================================

def all_series(region: str | None = None) -> list[Series]:
    from config.eu_series_catalog import all_eu_series  # lazy to avoid circular import
    series = list(US_FRED_SERIES) + all_eu_series()
    if region is not None:
        series = [s for s in series if s.region == region]
    return series


def by_vector(vector: Vector, region: str | None = None) -> list[Series]:
    return [s for s in all_series(region=region) if s.vector == vector]


def by_source(source: str) -> list[Series]:
    return [s for s in all_series() if s.source == source]


VECTORS: list[Vector] = [
    "liquidity", "credit", "labor", "consumer",
    "housing", "banking", "inflation", "early_warning",
]
