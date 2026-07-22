"""
Vector Drill-Down — three clicks to bedrock.
Click a vector → see every constituent series → click a series → raw values + z-score.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import timedelta

from storage.db import get_conn
from config.series_catalog import US_FRED_SERIES
from config.eu_series_catalog import all_eu_series

st.set_page_config(
    page_title="Vector Drill-Down",
    page_icon="🔍",
    layout="wide",
)

st.markdown("""
<style>
  .stApp { background: #0d1117; color: #e6edf3; }
  .block-container { padding-top: 1.5rem; }
  .metric-card {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 10px; padding: 14px 18px; margin-bottom: 8px;
  }
  .metric-label { font-size: 0.7rem; color: #8b949e; text-transform: uppercase;
                  letter-spacing: 0.08em; margin-bottom: 4px; }
  .metric-value { font-size: 1.6rem; font-weight: 700; line-height: 1; }
  .metric-sub   { font-size: 0.75rem; color: #8b949e; margin-top: 4px; }
  .series-card {
    background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    padding: 12px 16px; margin-bottom: 6px; cursor: pointer;
  }
  .series-card:hover { border-color: #58a6ff; }
  h2 { color: #e6edf3 !important; font-size: 0.9rem !important;
       text-transform: uppercase; letter-spacing: 0.06em;
       border-bottom: 1px solid #21262d; padding-bottom: 6px; }
  [data-testid="stSidebar"] { background: #161b22 !important; }
  [data-testid="stSidebar"] * { color: #c9d1d9 !important; }
  /* rename the auto "app" nav entry (derived from app.py filename) */
  [data-testid="stSidebarNav"] a span[label="app"] p { font-size: 0; }
  [data-testid="stSidebarNav"] a span[label="app"] p::before {
    content: "Overview";
    font-size: 0.9rem;
  }
  /* collapse Streamlit's empty default header band */
  header[data-testid="stHeader"] { background: transparent; height: 2.5rem; }
  #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── helpers ───────────────────────────────────────────────────────────────────
VECTORS = ["liquidity","credit","labor","consumer","housing","banking","early_warning"]
VECTOR_WEIGHTS = {"liquidity":0.20,"credit":0.20,"labor":0.15,"consumer":0.10,
                  "housing":0.10,"banking":0.15,"early_warning":0.10}

# Auto-derive from catalog so adding a series in one place is enough.
# Override specific entries where a shorter display name is warranted.
SERIES_META: dict[str, tuple[str, str, str]] = {
    s.series_id: (s.description, s.vector, s.description)
    for s in list(US_FRED_SERIES) + all_eu_series()
}
SERIES_META.update({
    "WALCL":     ("Fed Balance Sheet",    "liquidity",     "Total Assets of the Federal Reserve"),
    "RRPONTSYD": ("Overnight Repo (RRP)", "liquidity",     "Reverse Repo — excess reserves parked at Fed"),
    "SOFR":      ("SOFR",                 "liquidity",     "Secured Overnight Financing Rate"),
    "IC4WSA":    ("Initial Jobless Claims","labor",         "Weekly initial unemployment claims (4wk MA)"),
    "CCSA":      ("Continuing Claims",    "labor",         "Continuing unemployment insurance claims"),
    "UNRATE":    ("Unemployment Rate",    "labor",         "U-3 unemployment rate"),
    "UMCSENT":   ("UMich Sentiment",      "consumer",      "University of Michigan consumer sentiment"),
    "T10Y3M":    ("10Y-3M Yield Curve",   "early_warning", "Treasury spread — best recession predictor"),
    "T10Y2Y":    ("10Y-2Y Yield Curve",   "early_warning", "Treasury 10y minus 2y spread"),
    "NFCI":      ("NFCI",                 "early_warning", "Chicago Fed National Financial Conditions Index"),
    "ANFCI":     ("Adj NFCI",             "early_warning", "Adjusted NFCI (removes economic cycle)"),
    "EST.B.EU000A2X2A25.WT": ("€STR",    "liquidity",     "Euro Short-Term Rate (overnight unsecured)"),
    "CISS.D.U2.Z0Z.4F.EC.SS_CIN.IDX": ("ECB CISS", "early_warning",
                                        "Composite Indicator of Systemic Stress (daily)"),
})

def stress_color(val):
    if val is None or (isinstance(val, float) and np.isnan(val)): return "#8b949e"
    if val >= 2.0:  return "#f85149"
    if val >= 1.5:  return "#ff7b72"
    if val >= 1.0:  return "#e3b341"
    if val >= 0.5:  return "#f0883e"
    if val >= 0.0:  return "#3fb950"
    return "#58a6ff"

def stress_label(val):
    if val is None or (isinstance(val, float) and np.isnan(val)): return "N/A"
    if val >= 2.0:  return "SEVERE"
    if val >= 1.5:  return "ALERT"
    if val >= 1.0:  return "ELEVATED"
    if val >= 0.5:  return "MILD"
    if val >= 0.0:  return "NORMAL"
    return "EASING"

@st.cache_data(ttl=300)
def load_latest_signals(region: str = "US") -> pd.DataFrame:
    """Latest z-score per series (last available date per series) for a region."""
    with get_conn(read_only=True) as conn:
        df = conn.execute("""
            SELECT s.series_id, s.obs_date, s.z_score, s.vector
            FROM signals s
            INNER JOIN (
                SELECT series_id, MAX(obs_date) AS max_date
                FROM signals WHERE region = ? GROUP BY series_id
            ) m ON s.series_id = m.series_id AND s.obs_date = m.max_date
            WHERE s.region = ?
        """, [region, region]).df()
    return df

@st.cache_data(ttl=300)
def load_series_history(series_id: str, region: str = "US", days: int = 1825) -> pd.DataFrame:
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
    with get_conn(read_only=True) as conn:
        df = conn.execute("""
            SELECT o.obs_date, o.value AS raw_value, s.z_score,
                   s.transformed
            FROM observations o
            LEFT JOIN signals s
                   ON o.series_id = s.series_id
                  AND o.obs_date  = s.obs_date
                  AND s.region    = ?
            WHERE o.series_id = ?
              AND o.obs_date >= ?
            ORDER BY o.obs_date
        """, [region, series_id, cutoff.date()]).df()
    df["obs_date"] = pd.to_datetime(df["obs_date"])
    return df

@st.cache_data(ttl=300)
def load_vector_history(vector: str, region: str = "US", days: int = 730) -> pd.DataFrame:
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
    with get_conn(read_only=True) as conn:
        df = conn.execute("""
            SELECT obs_date, series_id, z_score
            FROM signals
            WHERE vector = ? AND region = ? AND obs_date >= ?
            ORDER BY obs_date
        """, [vector, region, cutoff.date()]).df()
    df["obs_date"] = pd.to_datetime(df["obs_date"])
    return df

_BG = "#0d1117"
_BG2 = "#161b22"

def chart_cfg():
    return dict(paper_bgcolor=_BG2, plot_bgcolor=_BG2,
                font=dict(color="#8b949e", size=11),
                margin=dict(l=10, r=10, t=30, b=10),
                hovermode="x unified")

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 Drill-Down")
    st.markdown("---")
    region = st.radio("REGION", ["US", "EU", "GLOBAL"], horizontal=True, index=0)
    st.markdown("---")
    if st.button("🔄 Refresh", width="stretch"):
        st.cache_data.clear()
        st.rerun()
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.7rem;color:#6e7681'>"
        "Developed by Mustapa Osman<br>"
        "<a href='mailto:qmjtnkwq9@mozmail.com' "
        "style='color:#58a6ff;text-decoration:none'>qmjtnkwq9@mozmail.com</a></div>",
        unsafe_allow_html=True
    )

# ── layout ────────────────────────────────────────────────────────────────────
st.markdown(f"# 🔍 {region} Vector Drill-Down")
st.markdown("<div style='color:#8b949e;font-size:0.85rem;margin-bottom:16px'>"
            "Select a vector → inspect constituent series → click any series for raw data</div>",
            unsafe_allow_html=True)

latest = load_latest_signals(region)

# ── LEVEL 1: vector selector ──────────────────────────────────────────────────
st.markdown("## Select Vector")

vec_cols = st.columns(len(VECTORS))
selected_vector = st.session_state.get("selected_vector", "banking")

for i, v in enumerate(VECTORS):
    row = latest[latest["vector"] == v]
    score = row["z_score"].mean() if not row.empty else float("nan")
    color = stress_color(score)
    label = stress_label(score)
    weight = int(VECTOR_WEIGHTS[v] * 100)
    border = "#58a6ff" if v == selected_vector else "#30363d"

    with vec_cols[i]:
        clicked = st.button(
            f"{'🔴' if score>=1.5 else '🟡' if score>=0.5 else '🟢'} "
            f"{v.replace('_',' ').title()}\n{score:+.2f}σ · {label}",
            key=f"vec_{v}",
            width="stretch",
        )
        if clicked:
            st.session_state["selected_vector"] = v
            st.session_state.pop("selected_series", None)
            st.rerun()

st.markdown("---")

# ── LEVEL 2: series in selected vector ────────────────────────────────────────
v = st.session_state.get("selected_vector", "banking")
st.markdown(f"## {v.replace('_',' ').title()} — Constituent Series")

# vector history chart (stacked lines)
vec_hist = load_vector_history(v, region)
if not vec_hist.empty:
    fig_vh = go.Figure()
    palette = ["#58a6ff","#bc8cff","#3fb950","#e3b341","#f0883e","#f85149","#79c0ff"]
    for idx, sid in enumerate(vec_hist["series_id"].unique()):
        sub = vec_hist[vec_hist["series_id"] == sid]
        name = SERIES_META.get(sid, (sid,))[0]
        fig_vh.add_trace(go.Scatter(
            x=sub["obs_date"], y=sub["z_score"],
            mode="lines", name=name,
            line=dict(color=palette[idx % len(palette)], width=1.5),
            hovertemplate=f"{name}: %{{y:.2f}}σ<extra></extra>",
        ))
    fig_vh.add_hline(y=1.5, line_dash="dot", line_color="#f85149",
                     line_width=1, opacity=0.5)
    fig_vh.add_hline(y=0, line_color="#30363d", line_width=1)
    fig_vh.update_layout(**chart_cfg(), height=200,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=9),
                    orientation="h", y=1.12))
    fig_vh.update_xaxes(showgrid=False, zeroline=False)
    fig_vh.update_yaxes(showgrid=True, gridcolor="#21262d", title="Z-score (σ)")
    st.plotly_chart(fig_vh, width="stretch")

# series cards — discovered dynamically from signals for the active region
series_in_vector = sorted(latest[latest["vector"] == v]["series_id"].unique())
series_cols = st.columns(3)

selected_series = st.session_state.get("selected_series", None)

for idx, sid in enumerate(series_in_vector):
    meta = SERIES_META.get(sid, (sid, v, ""))
    row = latest[latest["series_id"] == sid]
    score = float(row["z_score"].iloc[0]) if not row.empty else float("nan")
    last_date = str(row["obs_date"].iloc[0].date()) if not row.empty else "—"
    color = stress_color(score)
    label = stress_label(score)
    border = "2px solid #58a6ff" if sid == selected_series else "1px solid #30363d"

    with series_cols[idx % 3]:
        clicked = st.button(
            f"**{meta[0]}** `{sid}`\n{score:+.2f}σ — {label} · {last_date}",
            key=f"ser_{sid}",
            width="stretch",
        )
        if clicked:
            st.session_state["selected_series"] = sid
            st.rerun()

st.markdown("---")

# ── LEVEL 3: series raw data ──────────────────────────────────────────────────
sid = st.session_state.get("selected_series")
if sid:
    meta = SERIES_META.get(sid, (sid, v, ""))
    row  = latest[latest["series_id"] == sid]
    score = float(row["z_score"].iloc[0]) if not row.empty else float("nan")

    st.markdown(f"## {meta[0]}  `{sid}`")
    st.markdown(f"<div style='color:#8b949e;font-size:0.83rem;margin-bottom:12px'>"
                f"{meta[2]}</div>", unsafe_allow_html=True)

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    color = stress_color(score)
    with k1:
        st.markdown(f"""<div class="metric-card">
          <div class="metric-label">Current Z-score</div>
          <div class="metric-value" style="color:{color}">{score:+.2f}σ</div>
          <div class="metric-sub">{stress_label(score)}</div>
        </div>""", unsafe_allow_html=True)

    hist = load_series_history(sid, region)
    if not hist.empty:
        latest_raw = hist["raw_value"].dropna().iloc[-1]
        latest_z   = hist["z_score"].dropna().iloc[-1] if hist["z_score"].notna().any() else float("nan")
        z_pct      = (hist["z_score"].dropna() < latest_z).mean() * 100 if not np.isnan(latest_z) else float("nan")

        with k2:
            st.markdown(f"""<div class="metric-card">
              <div class="metric-label">Latest Raw Value</div>
              <div class="metric-value" style="font-size:1.4rem">{latest_raw:,.3g}</div>
              <div class="metric-sub">as of {hist["obs_date"].iloc[-1].date()}</div>
            </div>""", unsafe_allow_html=True)
        with k3:
            st.markdown(f"""<div class="metric-card">
              <div class="metric-label">Historical Percentile</div>
              <div class="metric-value" style="font-size:1.4rem;color:{stress_color(score)}">{z_pct:.0f}th</div>
              <div class="metric-sub">5yr rolling window</div>
            </div>""", unsafe_allow_html=True)
        with k4:
            obs_count = hist["z_score"].notna().sum()
            st.markdown(f"""<div class="metric-card">
              <div class="metric-label">Observations</div>
              <div class="metric-value" style="font-size:1.4rem">{obs_count:,}</div>
              <div class="metric-sub">{hist["obs_date"].min().date()} → {hist["obs_date"].max().date()}</div>
            </div>""", unsafe_allow_html=True)

        # dual chart: raw + z-score
        fig_s = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.5, 0.5], vertical_spacing=0.06,
            subplot_titles=["Raw Value", "Z-Score (5yr rolling)"],
        )

        # raw value
        fig_s.add_trace(go.Scatter(
            x=hist["obs_date"], y=hist["raw_value"],
            mode="lines", name="Raw",
            line=dict(color="#58a6ff", width=1.8),
            hovertemplate="%{x|%b %d %Y}: %{y:,.4g}<extra></extra>",
        ), row=1, col=1)

        # z-score with colour fill
        z = hist["z_score"]
        pos_mask = z >= 0
        fig_s.add_trace(go.Scatter(
            x=hist["obs_date"], y=z.where(pos_mask),
            mode="lines", name="Z-score (+)",
            line=dict(color="#f85149", width=0.5),
            fill="tozeroy", fillcolor="rgba(248,81,73,0.15)",
            hovertemplate="Z: %{y:.2f}σ<extra></extra>",
            showlegend=False,
        ), row=2, col=1)
        fig_s.add_trace(go.Scatter(
            x=hist["obs_date"], y=z.where(~pos_mask),
            mode="lines", name="Z-score (−)",
            line=dict(color="#3fb950", width=0.5),
            fill="tozeroy", fillcolor="rgba(63,185,80,0.15)",
            hovertemplate="Z: %{y:.2f}σ<extra></extra>",
            showlegend=False,
        ), row=2, col=1)
        fig_s.add_hline(y=1.5, line_dash="dot", line_color="#f85149",
                        line_width=1, opacity=0.5, row=2, col=1)
        fig_s.add_hline(y=0, line_color="#484f58", line_width=1, row=2, col=1)

        fig_s.update_layout(**chart_cfg(), height=420, showlegend=False)
        fig_s.update_xaxes(showgrid=False, zeroline=False, tickfont=dict(size=10))
        fig_s.update_yaxes(showgrid=True, gridcolor="#21262d",
                           zeroline=False, tickfont=dict(size=10))
        for ann in fig_s.layout.annotations:
            ann.font.color = "#8b949e"
            ann.font.size  = 11
        st.plotly_chart(fig_s, width="stretch")

        # raw data table
        with st.expander("📋 Raw data (last 60 obs)"):
            tbl = hist[["obs_date","raw_value","transformed","z_score"]].tail(60) \
                      .sort_values("obs_date", ascending=False).copy()
            tbl["obs_date"]    = tbl["obs_date"].dt.strftime("%Y-%m-%d")
            tbl["raw_value"]   = tbl["raw_value"].map(lambda x: f"{x:,.4g}" if pd.notna(x) else "—")
            tbl["transformed"] = tbl["transformed"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
            tbl["z_score"]     = tbl["z_score"].map(lambda x: f"{x:+.3f}" if pd.notna(x) else "—")
            tbl.columns        = ["Date","Raw Value","Transformed","Z-score"]
            st.dataframe(tbl, width="stretch", hide_index=True)
    else:
        st.warning(f"No history found for {sid}.")
else:
    st.info("👆 Click any series card above to see raw values and z-score history.")
