"""
Real-Economy Stress Dashboard — Streamlit UI
Run: streamlit run dashboard/app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from storage.db import get_conn

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Economic Stress Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── theme / CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* dark background */
  .stApp { background: #0d1117; color: #e6edf3; }
  .block-container { padding-top: 2.75rem; padding-bottom: 1rem; }

  /* collapse Streamlit's empty default header band (keep toolbar reachable) */
  header[data-testid="stHeader"] {
    background: transparent;
    height: 2.5rem;
  }
  [data-testid="stToolbar"] { right: 0.5rem; }
  #MainMenu, footer { visibility: hidden; }

  /* metric cards */
  .metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 8px;
  }
  .metric-label { font-size: 0.72rem; color: #8b949e; text-transform: uppercase;
                  letter-spacing: 0.08em; margin-bottom: 4px; }
  .metric-value { font-size: 2rem; font-weight: 700; line-height: 1; }
  .metric-sub   { font-size: 0.8rem; color: #8b949e; margin-top: 4px; }

  /* alert banner */
  .alert-banner {
    display: block;
    width: 100%;
    box-sizing: border-box;
    background: #2d1b1b;
    border: 1.5px solid #f85149;
    border-radius: 8px;
    padding: 14px 20px;
    margin: 0 0 16px 0;
    color: #f85149;
    font-weight: 600;
    font-size: 0.95rem;
    line-height: 1.5;
    overflow: visible;
  }
  .ok-banner {
    display: block;
    width: 100%;
    box-sizing: border-box;
    background: #0d2818;
    border: 1.5px solid #3fb950;
    border-radius: 8px;
    padding: 14px 20px;
    margin: 0 0 16px 0;
    color: #3fb950;
    font-weight: 600;
    font-size: 0.95rem;
    line-height: 1.5;
    overflow: visible;
  }

  /* vector row */
  .vec-row {
    display: flex; align-items: center; gap: 12px;
    padding: 8px 0; border-bottom: 1px solid #21262d;
  }
  .vec-name  { width: 130px; font-size: 0.82rem; color: #c9d1d9; }
  .vec-bar   { flex: 1; height: 8px; border-radius: 4px; background: #21262d; }
  .vec-fill  { height: 100%; border-radius: 4px; }
  .vec-val   { width: 50px; text-align: right; font-size: 0.85rem; font-weight: 600; }

  /* section headers */
  h2 { color: #e6edf3 !important; font-size: 1rem !important;
       text-transform: uppercase; letter-spacing: 0.06em;
       border-bottom: 1px solid #21262d; padding-bottom: 6px; }

  /* rename the auto "app" nav entry (derived from app.py filename) */
  [data-testid="stSidebarNav"] a span[label="app"] p { font-size: 0; }
  [data-testid="stSidebarNav"] a span[label="app"] p::before {
    content: "Overview";
    font-size: 0.9rem;
  }

  /* sidebar */
  [data-testid="stSidebar"] { background: #161b22 !important; }
  [data-testid="stSidebar"] * { color: #c9d1d9 !important; }
  div[data-testid="stSelectbox"] label,
  div[data-testid="stSlider"] label { color: #8b949e !important; font-size: 0.75rem; }
</style>
""", unsafe_allow_html=True)

# ── helpers ───────────────────────────────────────────────────────────────────
VECTORS = ["liquidity","credit","labor","consumer","housing","banking","inflation","early_warning"]
VECTOR_WEIGHTS = {"liquidity":0.18,"credit":0.18,"labor":0.14,"consumer":0.09,
                  "housing":0.09,"banking":0.14,"inflation":0.10,"early_warning":0.08}

VECTOR_DESC = {
    "US": {
        "liquidity":     "Fed balance sheet, reserves, SOFR, repo",
        "credit":        "HY spreads, IG spreads, leveraged loans",
        "labor":         "Jobless claims, SAHM, unemployment, hours",
        "consumer":      "Savings rate, revolving credit, sentiment, confidence",
        "housing":       "Mortgage rates, starts, permits",
        "banking":       "Deposits, commercial loans, loan loss provisions",
        "inflation":     "CPI, core CPI/PCE, energy, wages, breakevens, UMich",
        "early_warning": "Yield curve, NFCI, ANFCI, St. Louis FSI",
    },
    "EU": {
        "liquidity":     "ECB balance sheet, DFR, €STR overnight rate",
        "credit":        "MFI loans to non-financial corporations",
        "labor":         "EA21 unemployment rate (Eurostat)",
        "consumer":      "Consumer confidence, retail trade volume",
        "housing":       "House price index, 10Y government bond yield",
        "banking":       "M3 money supply, MFI deposits YoY",
        "inflation":     "— no EU inflation series yet (US-only vector)",
        "early_warning": "ECB CISS systemic stress index",
    },
    "GLOBAL": {
        "liquidity":     "60% Fed / 40% ECB balance sheet & rates",
        "credit":        "60% US spreads / 40% EU MFI lending",
        "labor":         "60% US jobless claims / 40% EA unemployment",
        "consumer":      "60% US confidence / 40% EU consumer confidence",
        "housing":       "60% US mortgage & starts / 40% EU HPI & yields",
        "banking":       "60% US deposits / 40% EU M3 & deposits",
        "inflation":     "US only — CPI/PCE, energy, wages, breakevens",
        "early_warning": "60% US yield curve & FSI / 40% ECB CISS",
    },
}

def stress_color(val):
    if val is None or np.isnan(val): return "#8b949e"
    if val >= 2.0:  return "#f85149"
    if val >= 1.5:  return "#ff7b72"
    if val >= 1.0:  return "#e3b341"
    if val >= 0.5:  return "#f0883e"
    if val >= 0.0:  return "#3fb950"
    return "#58a6ff"

def stress_label(val):
    if val is None or np.isnan(val): return "N/A"
    if val >= 2.0:  return "SEVERE"
    if val >= 1.5:  return "ALERT"
    if val >= 1.0:  return "ELEVATED"
    if val >= 0.5:  return "MILD"
    if val >= 0.0:  return "NORMAL"
    return "EASING"

# ── absolute inflation thresholds ─────────────────────────────────────────────
# Independent of the 5yr z-score. The z-score normalizes against the 2021-23
# inflation era, so it can read "easing" while YoY prints are still well above
# the Fed's 2% target. These tiers alert on the ABSOLUTE level of pain.
# Tiers are distance above the 2% target; same scheme for all three series.
ABS_INFLATION = {
    "PCEPILFE": "Core PCE",
    "CPILFESL": "Core CPI",
    "CPIAUCSL": "Headline CPI",
}
FED_TARGET = 2.0

def inflation_tier(yoy):
    """Return (label, color, severity) for an absolute YoY % print. Higher severity = worse."""
    if yoy is None or np.isnan(yoy):        return ("N/A",            "#8b949e", -1)
    if yoy >= 4.0:                          return ("SEVERE",         "#f85149",  4)
    if yoy >= 3.0:                          return ("HOT",            "#ff7b72",  3)
    if yoy >= 2.5:                          return ("ELEVATED",       "#f0883e",  2)
    if yoy >= FED_TARGET:                   return ("ABOVE TARGET",   "#e3b341",  1)
    return ("AT/BELOW TARGET",              "#3fb950",  0)

@st.cache_data(ttl=300)
def load_composite(region: str = "US") -> pd.DataFrame:
    with get_conn(read_only=True) as conn:
        df = conn.execute(
            "SELECT * FROM composite WHERE region = ? ORDER BY obs_date",
            (region,),
        ).df()
    df["obs_date"] = pd.to_datetime(df["obs_date"])
    return df

@st.cache_data(ttl=300)
def load_signals(region: str = "US") -> pd.DataFrame:
    with get_conn(read_only=True) as conn:
        df = conn.execute(
            "SELECT series_id, obs_date, z_score, vector, region FROM signals "
            "WHERE region = ? ORDER BY obs_date",
            (region,),
        ).df()
    df["obs_date"] = pd.to_datetime(df["obs_date"])
    return df

@st.cache_data(ttl=300)
def load_inflation_levels() -> dict:
    """Latest absolute YoY % for the tracked core/headline inflation series.
    Reads `transformed` (the YoY transform output), independent of z-scores."""
    out = {}
    with get_conn(read_only=True) as conn:
        for sid in ABS_INFLATION:
            row = conn.execute(
                "SELECT obs_date, transformed FROM signals "
                "WHERE series_id = ? AND transformed IS NOT NULL "
                "ORDER BY obs_date DESC LIMIT 1",
                (sid,),
            ).fetchone()
            if row:
                out[sid] = {"date": row[0], "yoy": row[1]}
    return out

@st.cache_data(ttl=300)
def load_pull_log() -> pd.DataFrame:
    with get_conn(read_only=True) as conn:
        df = conn.execute(
            "SELECT * FROM pull_log ORDER BY pulled_at DESC LIMIT 200"
        ).df()
    return df

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Stress Monitor")
    st.markdown("---")
    region = st.radio(
        "REGION",
        ["US", "EU", "GLOBAL"],
        horizontal=True,
        index=0,
    )
    st.markdown("---")
    lookback = st.selectbox(
        "CHART WINDOW",
        ["3 Months","6 Months","1 Year","2 Years","5 Years","All History"],
        index=2,
    )
    alert_threshold = st.slider("ALERT THRESHOLD (σ)", 0.5, 3.0, 1.5, 0.25)
    show_roc = st.toggle("Show 30-day ROC", value=True)
    show_vectors = st.toggle("Show Vector Lines", value=False)
    st.markdown("---")
    if st.button("🔄 Refresh Data", width="stretch"):
        st.cache_data.clear()
        st.rerun()
    st.markdown("---")
    _src = {"US": "FRED", "EU": "ECB · Eurostat", "GLOBAL": "FRED · ECB · Eurostat"}
    st.markdown(
        f"<div style='font-size:0.7rem;color:#484f58'>Data: {_src[region]} · Updated daily<br>"
        "Z-score vs 5yr rolling window</div>"
        "<div style='font-size:0.7rem;color:#6e7681;margin-top:10px'>"
        "Developed by Mustapa Osman<br>"
        "<a href='mailto:qmjtnkwq9@mozmail.com' "
        "style='color:#58a6ff;text-decoration:none'>qmjtnkwq9@mozmail.com</a></div>",
        unsafe_allow_html=True
    )

# ── load data ─────────────────────────────────────────────────────────────────
try:
    df = load_composite(region)
except Exception as e:
    st.error(f"Cannot load data: {e}\nRun: python run.py backfill")
    st.stop()

if df.empty:
    st.warning("No data yet. Run `python run.py backfill` first.")
    st.stop()

# apply lookback window
lookback_map = {
    "3 Months": 90, "6 Months": 180, "1 Year": 365,
    "2 Years": 730, "5 Years": 1825, "All History": 99999
}
cutoff = df["obs_date"].max() - timedelta(days=lookback_map[lookback])
df_view = df[df["obs_date"] >= cutoff].copy()

latest = df.iloc[-1]
target_date = latest["obs_date"] - pd.Timedelta(days=30)
prev_idx = (df["obs_date"] - target_date).abs().idxmin()
prev = df.loc[prev_idx]

# ── alert banner ──────────────────────────────────────────────────────────────
alert_vectors = [v for v in VECTORS
                 if latest[v] is not None and not np.isnan(latest[v])
                 and latest[v] >= alert_threshold]

if latest["overall"] >= alert_threshold or alert_vectors:
    vlist = ", ".join(v.upper() for v in alert_vectors)
    st.markdown(
        f'<div style="width:100%;box-sizing:border-box">'
        f'<div class="alert-banner">⚠️&nbsp; [{region}] STRESS ALERT &nbsp;—&nbsp; '
        f'Overall: {latest["overall"]:+.2f}σ &nbsp;|&nbsp; '
        f'Elevated vectors: {vlist or "overall"} &nbsp;|&nbsp; '
        f'As of {latest["obs_date"].date()}</div></div>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        f'<div style="width:100%;box-sizing:border-box">'
        f'<div class="ok-banner">✅&nbsp; [{region}] No stress alerts &nbsp;|&nbsp; '
        f'Overall: {latest["overall"]:+.2f}σ &nbsp;|&nbsp; '
        f'As of {latest["obs_date"].date()}</div></div>',
        unsafe_allow_html=True
    )

# ── absolute inflation alert banner ───────────────────────────────────────────
# US series only. The stress vector above is z-scored; this banner reads the
# raw YoY level vs the Fed's 2% target — surfacing inflation pain the z-score
# normalizes away.
if region in ("US", "GLOBAL"):
    levels = load_inflation_levels()
    if levels:
        worst = max(
            (inflation_tier(v["yoy"])[2] for v in levels.values()),
            default=-1,
        )
        chips = []
        for sid, label in ABS_INFLATION.items():
            if sid not in levels:
                continue
            yoy = levels[sid]["yoy"]
            tlabel, tcolor, _ = inflation_tier(yoy)
            chips.append(
                f'<span style="margin-right:18px">{label}: '
                f'<b style="color:{tcolor}">{yoy:+.1f}% {tlabel}</b></span>'
            )
        as_of = max(v["date"] for v in levels.values())
        if worst >= 3:      # any series HOT or worse
            cls, icon, head = "alert-banner", "🔥", "INFLATION ALERT"
        elif worst >= 1:    # above target
            cls, icon, head = "alert-banner", "⚠️", "INFLATION ABOVE TARGET"
        else:
            cls, icon, head = "ok-banner", "✅", "INFLATION AT TARGET"
        st.markdown(
            f'<div style="width:100%;box-sizing:border-box">'
            f'<div class="{cls}">{icon}&nbsp; {head} &nbsp;—&nbsp; '
            f'{"".join(chips)}&nbsp;|&nbsp; Fed target {FED_TARGET:.0f}% &nbsp;|&nbsp; '
            f'As of {as_of}</div></div>',
            unsafe_allow_html=True
        )

# ── top KPI row ───────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)

def kpi(col, label, val, prev_val=None, suffix="σ", desc=""):
    delta = val - prev_val if prev_val is not None else 0
    arrow = "▲" if delta > 0.01 else ("▼" if delta < -0.01 else "—")
    delta_color = "#f85149" if delta > 0.01 else ("#3fb950" if delta < -0.01 else "#8b949e")
    color = stress_color(val)
    col.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value" style="color:{color}">{val:+.2f}{suffix}</div>
      <div class="metric-sub" style="color:{delta_color}">{arrow} {abs(delta):.2f} 30d &nbsp; {desc}</div>
    </div>""", unsafe_allow_html=True)

kpi(c1, "Overall Stress",   latest["overall"],          prev["overall"],         desc=stress_label(latest["overall"]))
kpi(c2, "30-day ROC",       latest["overall_roc_30d"],  prev["overall_roc_30d"], desc="Leading signal")
kpi(c3, "Banking",          latest["banking"],           prev["banking"],          desc=stress_label(latest["banking"]))
kpi(c4, "Credit",           latest["credit"],            prev["credit"],           desc=stress_label(latest["credit"]))
kpi(c5, "Inflation",        latest["inflation"],         prev["inflation"],        desc=stress_label(latest["inflation"]))
kpi(c6, "Early Warning",    latest["early_warning"],     prev["early_warning"],    desc=stress_label(latest["early_warning"]))

st.markdown("<br>", unsafe_allow_html=True)

# ── main chart — subplots so ROC has its own independent y-axis ───────────────
from plotly.subplots import make_subplots

st.markdown(f"## {region} Overall Stress Index")

vc_colors = {
    "liquidity":"#58a6ff","credit":"#bc8cff","labor":"#3fb950",
    "consumer":"#e3b341","housing":"#f0883e","banking":"#f85149",
    "inflation":"#ff7b72","early_warning":"#79c0ff"
}

row_heights = [0.72, 0.28] if show_roc else [1.0]
n_rows      = 2 if show_roc else 1

fig = make_subplots(
    rows=n_rows, cols=1,
    shared_xaxes=True,
    row_heights=row_heights,
    vertical_spacing=0.04,
)

# alert band on main panel
fig.add_hrect(
    y0=alert_threshold, y1=5,
    fillcolor="rgba(248,81,73,0.06)", line_width=0,
    annotation_text=f"Alert ({alert_threshold}σ)",
    annotation_position="top left",
    annotation_font=dict(color="#f85149", size=10),
    row=1, col=1,
)
fig.add_hline(y=alert_threshold, line_dash="dot", line_color="#f85149",
              line_width=1, opacity=0.5, row=1, col=1)
fig.add_hline(y=0, line_color="#30363d", line_width=1, row=1, col=1)

# optional vector lines — step mode for monthly series
if show_vectors:
    monthly_series = {"labor","consumer","housing"}
    for v in VECTORS:
        line_shape = "hv" if v in monthly_series else "linear"
        fig.add_trace(go.Scatter(
            x=df_view["obs_date"], y=df_view[v],
            mode="lines", name=v.replace("_"," ").title(),
            line=dict(color=vc_colors[v], width=1, shape=line_shape),
            opacity=0.45,
            hovertemplate=f"{v}: %{{y:.2f}}σ<extra></extra>",
        ), row=1, col=1)

# overall stress — filled area
fig.add_trace(go.Scatter(
    x=df_view["obs_date"], y=df_view["overall"],
    mode="lines", name="Overall Stress",
    line=dict(color="#58a6ff", width=2.5),
    fill="tozeroy", fillcolor="rgba(88,166,255,0.08)",
    hovertemplate="<b>%{x|%b %d, %Y}</b><br>Overall: %{y:.2f}σ<extra></extra>",
), row=1, col=1)

# ROC in its own panel — independent scale
if show_roc:
    roc_vals = df_view["overall_roc_30d"]
    roc_color = ["#f85149" if v > 0 else "#3fb950" for v in roc_vals]
    fig.add_trace(go.Bar(
        x=df_view["obs_date"], y=roc_vals,
        name="30-day ROC", marker_color=roc_color,
        opacity=0.75,
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>ROC: %{y:.3f}<extra></extra>",
    ), row=2, col=1)
    fig.add_hline(y=0, line_color="#484f58", line_width=1, row=2, col=1)

_bg = "#0d1117"
fig.update_layout(
    paper_bgcolor=_bg, plot_bgcolor=_bg,
    font=dict(color="#8b949e", size=11),
    margin=dict(l=10, r=10, t=10, b=10),
    height=400 if show_roc else 300,
    legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1,
                orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    hovermode="x unified", bargap=0.1,
)
fig.update_xaxes(showgrid=False, zeroline=False, tickfont=dict(size=10),
                 linecolor="#21262d")
fig.update_yaxes(showgrid=True, gridcolor="#21262d", zeroline=False,
                 tickfont=dict(size=10))
fig.update_yaxes(title_text="Stress (σ)", row=1, col=1)
if show_roc:
    fig.update_yaxes(title_text="ROC", row=2, col=1)
    fig.update_xaxes(showticklabels=True, row=2, col=1)

st.plotly_chart(fig, width="stretch")

# ── vector breakdown ──────────────────────────────────────────────────────────
st.markdown("## Vector Breakdown")

left, right = st.columns([1, 1])

# left: bar chart latest reading
with left:
    vec_vals = {v: latest[v] for v in VECTORS}
    colors   = [stress_color(v) for v in vec_vals.values()]

    fig_bar = go.Figure(go.Bar(
        x=list(vec_vals.values()),
        y=[v.replace("_"," ").title() for v in vec_vals.keys()],
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.2f}σ" for v in vec_vals.values()],
        textposition="auto",
        hovertemplate="%{y}: %{x:.2f}σ<extra></extra>",
    ))
    fig_bar.add_vline(x=0, line_color="#484f58", line_width=1)
    fig_bar.add_vline(x=alert_threshold, line_dash="dot",
                      line_color="#f85149", line_width=1, opacity=0.6)
    fig_bar.update_layout(
        paper_bgcolor="#161b22", plot_bgcolor="#161b22",
        font=dict(color="#8b949e", size=11),
        margin=dict(l=10, r=10, t=10, b=10),
        height=280, showlegend=False,
        xaxis=dict(showgrid=True, gridcolor="#21262d", zeroline=False,
                   title="Z-score (σ)", tickfont=dict(size=10)),
        yaxis=dict(showgrid=False, tickfont=dict(size=11)),
    )
    st.plotly_chart(fig_bar, width="stretch")

# right: vector time series (last 6 months)
with right:
    df_6m = df[df["obs_date"] >= df["obs_date"].max() - timedelta(days=180)]
    fig_v = go.Figure()
    vc = {"liquidity":"#58a6ff","credit":"#bc8cff","labor":"#3fb950",
          "consumer":"#e3b341","housing":"#f0883e","banking":"#f85149",
          "inflation":"#ff7b72","early_warning":"#79c0ff"}
    monthly_vecs = {"labor","consumer","housing","inflation"}
    for v in VECTORS:
        shape = "hv" if v in monthly_vecs else "linear"
        fig_v.add_trace(go.Scatter(
            x=df_6m["obs_date"], y=df_6m[v],
            mode="lines", name=v.replace("_"," ").title(),
            line=dict(color=vc[v], width=1.5, shape=shape),
            hovertemplate=f"{v}: %{{y:.2f}}σ<extra></extra>"
        ))
    fig_v.add_hline(y=alert_threshold, line_dash="dot",
                    line_color="#f85149", line_width=1, opacity=0.5)
    fig_v.add_hline(y=0, line_color="#30363d", line_width=1)
    fig_v.update_layout(
        paper_bgcolor="#161b22", plot_bgcolor="#161b22",
        font=dict(color="#8b949e", size=11),
        margin=dict(l=10, r=10, t=10, b=10),
        height=280,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=9),
                    orientation="v", x=1.01),
        hovermode="x unified",
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor="#21262d", zeroline=False,
                   tickfont=dict(size=10)),
    )
    st.plotly_chart(fig_v, width="stretch")

# ── vector detail cards ───────────────────────────────────────────────────────
st.markdown("## Vector Status")

cols = st.columns(4)
for i, v in enumerate(VECTORS):
    val   = latest[v]
    p_val = prev[v]
    delta = val - p_val if not np.isnan(val) and not np.isnan(p_val) else 0
    color = stress_color(val)
    label = stress_label(val)
    arrow = "▲" if delta > 0.01 else ("▼" if delta < -0.01 else "—")
    dcol  = "#f85149" if delta > 0.01 else ("#3fb950" if delta < -0.01 else "#8b949e")
    weight = int(VECTOR_WEIGHTS[v] * 100)

    cols[i % 4].markdown(f"""
    <div class="metric-card">
      <div class="metric-label">{v.replace("_"," ").upper()} &nbsp;
        <span style="color:#484f58">({weight}%)</span></div>
      <div class="metric-value" style="color:{color};font-size:1.6rem">{val:+.2f}σ</div>
      <div style="font-size:0.75rem;color:{color};font-weight:600;margin-top:2px">{label}</div>
      <div class="metric-sub" style="color:{dcol}">{arrow} {abs(delta):.2f} 30d</div>
      <div style="font-size:0.68rem;color:#484f58;margin-top:6px">{VECTOR_DESC[region][v]}</div>
    </div>""", unsafe_allow_html=True)

# ── historical heatmap ────────────────────────────────────────────────────────
st.markdown("## Stress Heatmap — Monthly")

df_monthly = df.set_index("obs_date").resample("ME")[VECTORS + ["overall"]].mean()
df_monthly.index = df_monthly.index.strftime("%Y-%m")
df_monthly = df_monthly.tail(36)  # last 3 years

fig_heat = go.Figure(go.Heatmap(
    z=df_monthly[VECTORS].T.values,
    x=df_monthly.index,
    y=[v.replace("_"," ").title() for v in VECTORS],
    colorscale=[
        [0.000, "#0d2818"],   # −2σ extreme easing
        [0.375, "#3fb950"],   # −0.5σ easing
        [0.500, "#21262d"],   # 0σ neutral (zmid)
        [0.625, "#3fb950"],   # +0.5σ still NORMAL
        [0.750, "#f0883e"],   # +1.0σ MILD
        [0.875, "#e3b341"],   # +1.5σ ELEVATED
        [1.000, "#f85149"],   # +2.0σ ALERT / SEVERE
    ],
    zmid=0,
    zmin=-2,
    zmax=2,
    colorbar=dict(tickfont=dict(color="#8b949e", size=10),
                  tickvals=[-2, -1, 0, 1, 2],
                  ticktext=["-2σ", "-1σ", "0", "+1σ", "+2σ"],
                  title=dict(text="σ", font=dict(color="#8b949e"))),
    hovertemplate="%{y}<br>%{x}: %{z:.2f}σ<extra></extra>",
))
fig_heat.update_layout(
    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
    font=dict(color="#8b949e", size=11),
    margin=dict(l=10, r=10, t=10, b=10),
    height=240,
    xaxis=dict(showgrid=False, tickangle=-45, tickfont=dict(size=9)),
    yaxis=dict(showgrid=False, tickfont=dict(size=11)),
)
st.plotly_chart(fig_heat, width="stretch")

# ── recent readings table ─────────────────────────────────────────────────────
with st.expander("📋  Recent Readings (last 30 days)"):
    display_cols = ["obs_date","overall","overall_roc_30d"] + VECTORS
    df_table = df[display_cols].tail(30).sort_values("obs_date", ascending=False).copy()
    df_table["obs_date"] = df_table["obs_date"].dt.strftime("%Y-%m-%d")
    for col in display_cols[1:]:
        df_table[col] = df_table[col].map(lambda x: f"{x:+.2f}" if pd.notna(x) else "—")
    st.dataframe(df_table, width="stretch", hide_index=True)

# ── last update info ──────────────────────────────────────────────────────────
with st.expander("⚙️  System Info"):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Last data date:** {latest['obs_date'].date()}")
        st.markdown(f"**Total composite rows:** {len(df):,}")
        st.markdown(f"**Dashboard refreshes every:** 5 minutes")
    with c2:
        try:
            log_df = load_pull_log()
            if not log_df.empty:
                st.markdown(f"**Last pull:** {log_df['pulled_at'].iloc[0]}")
                ok  = log_df["success"].sum()
                err = (~log_df["success"]).sum()
                st.markdown(f"**Pull log:** {ok} OK · {err} errors (last 200)")
        except Exception:
            st.markdown("Pull log unavailable")
