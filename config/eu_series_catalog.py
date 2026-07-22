"""
EU series catalog — ECB Data Portal + Eurostat.

Same Series structure as US catalog, but region='EU' so composite math
can produce US-only, EU-only, and combined readings.

Source notes:
  - ECB series IDs from data.ecb.europa.eu (SDMX REST API)
  - Eurostat dataset codes from ec.europa.eu/eurostat/web/main/data/database
  - All free, no API key required for either

Verifying series IDs:
  ECB:      https://data.ecb.europa.eu/  (search, copy series key)
  Eurostat: https://ec.europa.eu/eurostat/databrowser/  (find dataset code)
"""

from config.series_catalog import Series


# =============================================================================
# ECB — Composite stress, banking, liquidity, rates
# =============================================================================

EU_ECB_SERIES: list[Series] = [
    # --- EARLY WARNING (composites) ---
    # CISS — Composite Indicator of Systemic Stress (the EU NFCI equivalent)
    Series("CISS.D.U2.Z0Z.4F.EC.SS_CIN.IDX", "ecb", "early_warning",
           "Composite Indicator of Systemic Stress (ECB CISS)",
           direction=+1, transform="level", weight=1.2, region="EU"),

    # --- LIQUIDITY ---
    # ECB total assets (the EU WALCL equivalent) — weekly
    Series("ILM.W.U2.C.T000000.Z5.Z01", "ecb", "liquidity",
           "ECB Balance Sheet Total Assets",
           direction=-1, transform="level", weight=1.0, region="EU"),

    # ECB Deposit Facility Rate (DFR) — policy rate
    Series("FM.B.U2.EUR.4F.KR.DFR.LEV", "ecb", "liquidity",
           "ECB Deposit Facility Rate",
           direction=+1, transform="level", weight=0.5, region="EU"),

    # €STR (Euro Short-Term Rate) — overnight unsecured rate
    Series("EST.B.EU000A2X2A25.WT", "ecb", "liquidity",
           "Euro Short-Term Rate (€STR)",
           direction=+1, transform="level", weight=0.6, region="EU"),

    # --- CREDIT ---
    # MFI loans to non-financial corporations — YoY growth
    Series("BSI.M.U2.N.A.A20.A.1.U2.2240.Z01.E", "ecb", "credit",
           "MFI Loans to Non-Financial Corporations",
           direction=-1, transform="yoy", weight=0.8, region="EU"),

    # --- BANKING ---
    # M3 broad money — YoY (M3 contraction is a real banking stress signal)
    Series("BSI.M.U2.Y.V.M30.X.1.U2.2300.Z01.E", "ecb", "banking",
           "Eurozone M3 Money Supply",
           direction=-1, transform="yoy", weight=0.9, region="EU"),

    # MFI deposits from euro-area residents — YoY (deposit flight signal)
    Series("BSI.M.U2.N.A.L20.A.1.U2.2240.Z01.E", "ecb", "banking",
           "MFI Deposits from Euro Area Residents",
           direction=-1, transform="yoy", weight=1.0, region="EU"),

    # --- EARLY WARNING / rates ---
    # Euro area 10Y AAA yield curve (replaces Bund/BTP which are 404 on ECB API)
    Series("YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y", "ecb", "early_warning",
           "Euro Area 10Y AAA Yield Curve Spot Rate",
           direction=+1, transform="level", weight=0.7, region="EU"),
]


# =============================================================================
# Eurostat — Real economy: labor, consumer, housing, industry
# =============================================================================
#
# Eurostat dataset codes (the package uses these directly):
#   une_rt_m      — Unemployment rate, monthly
#   sts_trtu_m    — Retail trade volume, monthly
#   sts_inpr_m    — Industrial production, monthly
#   prc_hicp_manr — HICP, monthly annual rate
#   ei_bsco_m     — Consumer confidence indicator, monthly
#   prc_hpi_q     — House price index, quarterly
#   ei_bsfs_m     — Financial situation expectations, monthly
#
# For each Eurostat dataset we'll filter to euro-area aggregate (EA20 or EA)
# and the relevant transformation/series. The fetcher handles filtering.
# =============================================================================

EU_EUROSTAT_SERIES: list[Series] = [
    # --- LABOR ---
    Series("une_rt_m", "eurostat", "labor",
           "Eurozone Unemployment Rate (monthly)",
           direction=+1, transform="level", weight=1.0, region="EU"),

    # --- CONSUMER ---
    Series("ei_bsco_m", "eurostat", "consumer",
           "Eurozone Consumer Confidence Indicator",
           direction=-1, transform="level", weight=1.0, region="EU"),

    Series("sts_trtu_m", "eurostat", "consumer",
           "Eurozone Retail Trade Volume",
           direction=-1, transform="yoy", weight=0.8, region="EU"),

    # --- HOUSING ---
    Series("prc_hpi_q", "eurostat", "housing",
           "Eurozone House Price Index (quarterly)",
           direction=-1, transform="yoy", weight=0.8, region="EU"),

    # --- EARLY WARNING (industrial production, inflation surprise) ---
    Series("sts_inpr_m", "eurostat", "early_warning",
           "Eurozone Industrial Production",
           direction=-1, transform="yoy", weight=0.7, region="EU"),

    Series("prc_hicp_manr", "eurostat", "early_warning",
           "Eurozone HICP Annual Rate",
           direction=+1, transform="level", weight=0.4, region="EU"),
]


# =============================================================================
# Aggregator
# =============================================================================

def all_eu_series() -> list[Series]:
    return list(EU_ECB_SERIES) + list(EU_EUROSTAT_SERIES)
