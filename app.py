"""
ML-Based Flood Prediction for Ungauged Rivers
Reviewer-Facing Interactive Demo — Inference & Evaluation Visualization

Streamlit application for supervisors and reviewers to inspect final model predictions,
uncertainty intervals, and benchmark comparisons across the 37 held-out unseen test catchments.
"""

from datetime import datetime, date
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import streamlit as st

from src.visualization.demo_data import (
    get_repo_root,
    get_test_gauge_ids,
    load_catchment_metadata,
    load_option_b_predictions,
    load_xgboost_predictions,
    load_per_catchment_metrics,
    load_overall_model_summary,
    load_summary_json,
)

# Page configuration
st.set_page_config(
    page_title="Flood Predictor — Reviewer Demo",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.15rem;
        font-weight: 500;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 12px 16px;
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    .metric-val {
        font-size: 1.4rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .disclaimer-box {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 14px 18px;
        margin: 20px 0;
        border-radius: 4px;
        color: #92400e;
        font-size: 0.95rem;
    }
    .period-box {
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 14px 18px;
        margin: 20px 0;
        border-radius: 4px;
        color: #1e40af;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_cached_gauge_ids():
    return get_test_gauge_ids()


@st.cache_data
def get_cached_metadata(gauge_id: str):
    return load_catchment_metadata(gauge_id)


@st.cache_data
def get_cached_option_b_preds(gauge_id: str):
    return load_option_b_predictions(gauge_id)


@st.cache_data
def get_cached_xgboost_preds(gauge_id: str):
    return load_xgboost_predictions(gauge_id)


@st.cache_data
def get_cached_metrics(gauge_id: str):
    return load_per_catchment_metrics(gauge_id)


@st.cache_data
def get_cached_overall_summary():
    return load_overall_model_summary()


@st.cache_data
def get_cached_summary_json():
    return load_summary_json()


# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.title("Catchment Selection")
test_gauges = get_cached_gauge_ids()

# Format selector label with basin and site name if available
def format_gauge_option(gid: str) -> str:
    try:
        meta = get_cached_metadata(gid)
        site = meta.get("site_name", "")
        basin = meta.get("basin", "")
        if site and basin:
            return f"{gid} — {basin} ({site})"
    except Exception:
        pass
    return gid

# Default selection: 08013 (good representative site)
default_index = test_gauges.index("08013") if "08013" in test_gauges else 0

selected_gauge_str = st.sidebar.selectbox(
    "Select unseen test catchment:",
    options=test_gauges,
    index=default_index,
    format_func=format_gauge_option,
    help="Strictly contains the 37 held-out unseen test catchments (19 Mahanadi, 18 Narmada).",
)
selected_gauge = selected_gauge_str

# Sidebar quick filters
st.sidebar.markdown("---")
st.sidebar.subheader("Date Range Zoom")

# Load data for selected catchment
df_opt_b = get_cached_option_b_preds(selected_gauge)
min_date = df_opt_b["date"].min().date()
max_date = df_opt_b["date"].max().date()

# Quick presets
date_preset = st.sidebar.radio(
    "Zoom Presets:",
    options=[
        "Full 10-Year Record (1999–2009)",
        "2003 Monsoon (Peak Season)",
        "2006 Monsoon (High Flow)",
        "Custom Date Range",
    ],
    index=0,
)

if date_preset == "2003 Monsoon (Peak Season)":
    start_date = date(2003, 6, 1)
    end_date = date(2003, 11, 30)
elif date_preset == "2006 Monsoon (High Flow)":
    start_date = date(2006, 6, 1)
    end_date = date(2006, 11, 30)
elif date_preset == "Custom Date Range":
    selected_range = st.sidebar.slider(
        "Select custom window:",
        min_value=min_date,
        max_value=max_date,
        value=(date(2002, 1, 1), date(2005, 12, 31)),
        format="YYYY-MM-DD",
    )
    start_date, end_date = selected_range
else:
    start_date = min_date
    end_date = max_date

show_xgb = st.sidebar.checkbox("Overlay XGBoost Baseline", value=False)
show_q90_line = st.sidebar.checkbox("Show 90th percentile high-flow threshold", value=True)

# -----------------------------------------------------------------------------
# MAIN APP HEADER
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">ML-Based Flood Prediction for Ungauged Rivers</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Unseen-Catchment Hydrological Evaluation & Uncertainty Demonstration</div>', unsafe_allow_html=True)

# Catchment metadata panel
catchment_meta = get_cached_metadata(selected_gauge)
catchment_metrics = get_cached_metrics(selected_gauge)

area_display = f"{catchment_meta['drainage_area_km2']:,.1f} km²" if catchment_meta["drainage_area_km2"] else "Not available"

st.subheader(f"Catchment Information — Gauge {selected_gauge}")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Catchment ID</div><div class="metric-val">{selected_gauge}</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><div class="metric-label">River Basin</div><div class="metric-val">{catchment_meta["basin"]}</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-card"><div class="metric-label">River / Site</div><div class="metric-val">{catchment_meta["site_name"]}</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Drainage Area</div><div class="metric-val">{area_display}</div></div>', unsafe_allow_html=True)

st.write("")

# Filter data to active window
mask = (df_opt_b["date"].dt.date >= start_date) & (df_opt_b["date"].dt.date <= end_date)
df_sub = df_opt_b[mask].copy()

# Optional XGBoost overlay
df_xgb = get_cached_xgboost_preds(selected_gauge)
if df_xgb is not None:
    mask_xgb = (df_xgb["date"].dt.date >= start_date) & (df_xgb["date"].dt.date <= end_date)
    df_xgb_sub = df_xgb[mask_xgb].copy()
else:
    df_xgb_sub = None

# High-flow threshold
q90_thresh = catchment_metrics["option_b_uncertainty"]["q90_threshold_m3s"]

# -----------------------------------------------------------------------------
# SECTION 1: OBSERVED VS PREDICTED HYDROGRAPH
# -----------------------------------------------------------------------------
st.markdown("---")
st.header("1. Observed vs Predicted Streamflow")
st.markdown("*Interactive hydrograph comparing observed daily streamflow against Option B Q50 deterministic point predictions.*")

fig1, ax1 = plt.subplots(figsize=(13, 4.8), dpi=150)

# Observed flow (missing y_true remains missing/unconnected)
ax1.plot(
    df_sub["date"],
    df_sub["y_true"],
    label="Observed Flow ($y_{\\mathrm{true}}$)",
    color="#0f172a",
    linewidth=1.7,
    alpha=0.9,
    zorder=3,
)

# Option B Q50
ax1.plot(
    df_sub["date"],
    df_sub["q50"],
    label="Option B Q50 (Median Predicted Flow)",
    color="#0284c7",
    linewidth=1.6,
    alpha=0.95,
    zorder=4,
)

# Optional XGBoost overlay
if show_xgb and df_xgb_sub is not None and not df_xgb_sub.empty:
    ax1.plot(
        df_xgb_sub["date"],
        df_xgb_sub["y_pred"],
        label="XGBoost Baseline",
        color="#dc2626",
        linestyle="--",
        linewidth=1.2,
        alpha=0.8,
        zorder=2,
    )

if show_q90_line and q90_thresh is not None:
    ax1.axhline(
        q90_thresh,
        color="#f97316",
        linestyle=":",
        linewidth=1.3,
        label=f"Observed 90th %ile Threshold ({q90_thresh:.1f} m³/s)",
        zorder=1,
    )

ax1.set_xlabel("Date", fontsize=11, fontweight="medium")
ax1.set_ylabel("Discharge (m³/s)", fontsize=11, fontweight="medium")
ax1.set_title(
    f"Gauge {selected_gauge} ({catchment_meta['site_name']}) — Observed vs Option B Q50 Simulated Flow",
    fontsize=12,
    fontweight="bold",
    pad=10,
)
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(loc="upper right", framealpha=0.92, fontsize=9.5)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
fig1.tight_layout()

st.pyplot(fig1)
st.caption("Caption: Q50 is the model's median predicted discharge (m³/s). Missing observations are masked and not imputed.")

# -----------------------------------------------------------------------------
# SELECTED-DAY DISCHARGE PANEL
# -----------------------------------------------------------------------------
st.subheader("Selected-Day Discharge")
st.markdown("*Inspect exact numerical streamflow observations and predictive discharge quantiles for an individual date.*")

default_insp_date = date(2003, 8, 28) if (min_date <= date(2003, 8, 28) <= max_date) else min_date

insp_col1, insp_col2 = st.columns([1, 3])
with insp_col1:
    selected_day = st.date_input(
        "Select date:",
        value=default_insp_date,
        min_value=min_date,
        max_value=max_date,
        key=f"insp_date_{selected_gauge}",
        help="Restricted to available prediction dates for the selected catchment.",
    )

day_match = df_opt_b[df_opt_b["date"].dt.date == selected_day]

if not day_match.empty:
    day_row = day_match.iloc[0]
    obs_val = day_row["y_true"]
    q10_val = day_row["q10"]
    q50_val = day_row["q50"]
    q90_val = day_row["q90"]
    formatted_date_str = selected_day.strftime("%d-%b-%Y")

    with insp_col2:
        st.markdown(f"**Selected date:** `{formatted_date_str}`")
        dc1, dc2, dc3, dc4 = st.columns(4)
        with dc1:
            if pd.notna(obs_val):
                st.metric("Observed Flow", f"{obs_val:,.1f} m³/s")
            else:
                st.metric("Observed Flow", "Not available")
        with dc2:
            st.metric("Q10", f"{q10_val:,.1f} m³/s")
        with dc3:
            st.metric("Q50", f"{q50_val:,.1f} m³/s")
        with dc4:
            st.metric("Q90", f"{q90_val:,.1f} m³/s")

    if pd.notna(obs_val):
        in_band = (q10_val <= obs_val <= q90_val)
        band_text = "Inside 80% interval [Q10, Q90]" if in_band else ("Above Q90 bound" if obs_val > q90_val else "Below Q10 bound")
        diff = q50_val - obs_val
        diff_text = f"Q50 is {abs(diff):,.1f} m³/s {'above' if diff > 0 else 'below'} observed flow" if abs(diff) >= 0.1 else "Q50 matches observed flow"
        st.caption(f"Inspection Summary ({formatted_date_str}): {band_text} | {diff_text}. (Daily inspection only; benchmark metrics below reflect the full period).")
    else:
        st.caption(f"Inspection Summary ({formatted_date_str}): Observed discharge: Not available (missing observation). Forward predictive discharge quantiles are shown above.")

# -----------------------------------------------------------------------------
# SECTION 2: QUANTILE UNCERTAINTY PLOT
# -----------------------------------------------------------------------------
st.markdown("---")
st.header("2. Prediction Uncertainty (Predictive Discharge Quantiles)")
st.markdown("*80% predictive discharge interval ($Q_{10}$–$Q_{90}$) encompassing median simulated discharge ($Q_{50}$).*")

fig2, ax2 = plt.subplots(figsize=(13, 5.0), dpi=150)

# Shaded uncertainty band between Q10 and Q90
ax2.fill_between(
    df_sub["date"],
    df_sub["q10"],
    df_sub["q90"],
    color="#38bdf8",
    alpha=0.35,
    label="80% Predictive Interval: Q10–Q90",
    zorder=2,
)

# Quantile bounds
ax2.plot(
    df_sub["date"],
    df_sub["q90"],
    color="#0284c7",
    linestyle="--",
    linewidth=1.0,
    alpha=0.7,
    label="Q90 (90th Predictive Discharge Quantile)",
    zorder=3,
)
ax2.plot(
    df_sub["date"],
    df_sub["q10"],
    color="#0284c7",
    linestyle=":",
    linewidth=1.0,
    alpha=0.7,
    label="Q10 (10th Predictive Discharge Quantile)",
    zorder=3,
)

# Q50 median line
ax2.plot(
    df_sub["date"],
    df_sub["q50"],
    color="#0369a1",
    linewidth=1.7,
    label="Q50 (Median Predictive Discharge)",
    zorder=4,
)

# Observed flow
ax2.plot(
    df_sub["date"],
    df_sub["y_true"],
    color="#0f172a",
    linewidth=1.5,
    label="Observed Flow ($y_{\\mathrm{true}}$)",
    zorder=5,
)

if show_q90_line and q90_thresh is not None:
    ax2.axhline(
        q90_thresh,
        color="#ea580c",
        linestyle=":",
        linewidth=1.3,
        label=f"Observed 90th %ile Threshold ({q90_thresh:.1f} m³/s)",
        zorder=1,
    )

ax2.set_xlabel("Date", fontsize=11, fontweight="medium")
ax2.set_ylabel("Discharge (m³/s)", fontsize=11, fontweight="medium")
ax2.set_title(
    f"Gauge {selected_gauge} — Multi-Quantile Uncertainty Interval ($Q_{10}, Q_{50}, Q_{90}$)",
    fontsize=12,
    fontweight="bold",
    pad=10,
)
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend(loc="upper right", framealpha=0.92, fontsize=9.5)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
fig2.tight_layout()

st.pyplot(fig2)

st.markdown("""
**Quantile Definitions:**
- **Q10:** 10th predictive discharge quantile (lower 80% interval bound in m³/s)
- **Q50:** Median predictive discharge (deterministic point forecast in m³/s)
- **Q90:** 90th predictive discharge quantile (upper 80% interval bound in m³/s)
""")

# -----------------------------------------------------------------------------
# SECTION 3: CATCHMENT-LEVEL METRICS
# -----------------------------------------------------------------------------
st.markdown("---")
st.header(f"3. Performance Metrics — Gauge {selected_gauge}")

det = catchment_metrics["option_b_deterministic"]
unc = catchment_metrics["option_b_uncertainty"]

col_det, col_unc = st.columns(2)

with col_det:
    st.subheader("Main Prediction Quality")
    m1, m2, m3 = st.columns(3)
    m1.metric("NSE", f"{det['nse']:.3f}")
    m2.metric("Pearson r", f"{det['pearson_r']:.3f}")
    m3.metric("High-Flow NSE", f"{det['high_flow_nse']:.3f}")

with col_unc:
    st.subheader("Prediction Uncertainty")
    u1, u2 = st.columns(2)
    u1.metric("PICP 80%", f"{unc['picp_80']*100:.1f}%", help="Nominal target: 80.0%")
    u2.metric("MPIW", f"{unc['mpiw']:.1f} m³/s")

    u3, u4 = st.columns(2)
    u3.metric("High-Flow PICP 80%", f"{unc['hf_picp_80']*100:.1f}%", help="Empirical coverage during top 10% flow days.")
    u4.metric("High-Flow MPIW", f"{unc['hf_mpiw']:.1f} m³/s", help="Mean interval width during top 10% flow days.")

# -----------------------------------------------------------------------------
# SECTION 4: EXTREME / HIGH-FLOW PERFORMANCE
# -----------------------------------------------------------------------------
st.markdown("---")
st.header("4. Extreme / High-Flow Performance")
st.info("High-flow evaluation focuses on the top 10% of observed discharge values (observations exceeding the 90th percentile).")

hfc1, hfc2 = st.columns(2)
with hfc1:
    st.metric("High-Flow NSE", f"{det['high_flow_nse']:.3f}")
with hfc2:
    st.metric("High-Flow PICP 80%", f"{unc['hf_picp_80']*100:.1f}%", help="Empirical coverage during top 10% flow days.")

st.caption("Note: High-flow discharge quantiles reflect physical volume uncertainty during monsoon flood surges; they do not represent a formal probability of flood disaster.")

# -----------------------------------------------------------------------------
# SECTION 5: OVERALL MODEL COMPARISON
# -----------------------------------------------------------------------------
st.markdown("---")
st.header("5. Multi-Model Benchmark Comparison (37 Unseen Test Catchments)")

overall_summary_df = get_cached_overall_summary()

# Render model comparison table
display_cols = [
    "Model",
    "Evaluation Period",
    "Median NSE",
    "Mean NSE",
    "Median KGE",
    "Median Pearson r",
    "Median RMSE (m3/s)",
    "Median MAE (m3/s)",
    "Median PBIAS (%)",
    "Median High-Flow NSE",
    "Fraction NSE > 0 (%)",
    "Fraction NSE > 0.5 (%)",
]

available_cols = [c for c in display_cols if c in overall_summary_df.columns]
st.dataframe(
    overall_summary_df[available_cols].style.format({
        "Median NSE": "{:.3f}",
        "Mean NSE": "{:.3f}",
        "Median KGE": "{:.3f}",
        "Median Pearson r": "{:.3f}",
        "Median RMSE (m3/s)": "{:.1f}",
        "Median MAE (m3/s)": "{:.1f}",
        "Median PBIAS (%)": "{:.1f}%",
        "Median High-Flow NSE": "{:.3f}",
        "Fraction NSE > 0 (%)": "{:.1f}%",
        "Fraction NSE > 0.5 (%)": "{:.1f}%",
    }),
    use_container_width=True,
)

# Comparison Bar Chart
fig_bar, (ax_b1, ax_b2) = plt.subplots(1, 2, figsize=(11, 4.0), dpi=150)

models_plot = [
    "XGBoost\n(Matched)",
    "Base\nTCN",
    "Regional\nTCN",
    "Option B\n(Q50)",
]
# Extract medians matching row names
def get_val(df, m_name, col):
    m = df[df["Model"].str.contains(m_name, case=False, na=False)]
    return float(m[col].iloc[0]) if not m.empty else 0.0

nse_vals = [
    get_val(overall_summary_df, "Matched", "Median NSE"),
    get_val(overall_summary_df, "Base TCN", "Median NSE"),
    get_val(overall_summary_df, "Regional TCN", "Median NSE"),
    get_val(overall_summary_df, "Option B", "Median NSE"),
]

kge_vals = [
    get_val(overall_summary_df, "Matched", "Median KGE"),
    get_val(overall_summary_df, "Base TCN", "Median KGE"),
    get_val(overall_summary_df, "Regional TCN", "Median KGE"),
    get_val(overall_summary_df, "Option B", "Median KGE"),
]

colors = ["#dc2626", "#f59e0b", "#2563eb", "#10b981"]

# Bar chart 1: Median NSE
ax_b1.bar(models_plot, nse_vals, color=colors, width=0.55, edgecolor="#334155")
ax_b1.axhline(0, color="gray", linestyle="--", linewidth=1.0)
ax_b1.set_ylabel("Median NSE", fontsize=10, fontweight="medium")
ax_b1.set_title("Median Regional NSE (37 Test Basins)", fontsize=11, fontweight="bold")
ax_b1.set_ylim(-3.5, 0.4)
ax_b1.grid(True, linestyle=":", alpha=0.5, axis="y")
for i, v in enumerate(nse_vals):
    ax_b1.text(i, v + 0.1 if v < 0 else v + 0.03, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")

# Bar chart 2: Median KGE
ax_b2.bar(models_plot, kge_vals, color=colors, width=0.55, edgecolor="#334155")
ax_b2.axhline(0, color="gray", linestyle="--", linewidth=1.0)
ax_b2.set_ylabel("Median KGE", fontsize=10, fontweight="medium")
ax_b2.set_title("Median Regional KGE (37 Test Basins)", fontsize=11, fontweight="bold")
ax_b2.set_ylim(-6.0, 0.45)
ax_b2.grid(True, linestyle=":", alpha=0.5, axis="y")
for i, v in enumerate(kge_vals):
    ax_b2.text(i, v + 0.2 if v < 0 else v + 0.03, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")

fig_bar.tight_layout()
st.pyplot(fig_bar)

# Mandatory Evaluation-Period Warning Callout
st.markdown("""
<div class="period-box">
<strong>Evaluation-Period Note:</strong><br>
Option B and period-matched XGBoost use the identical 1999–2009 evaluation period. Base TCN and Regional TCN use their existing committed benchmark metrics over their available historical test records. Therefore, the four-model table is contextual rather than a fully common-period apples-to-apples comparison.
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SECTION 6: OVERALL PROJECT RESULTS
# -----------------------------------------------------------------------------
st.markdown("---")
st.header("6. Overall Project Results — 37 Unseen Test Catchments")

sum_json = get_cached_summary_json()
opt_b_sum = sum_json.get("Option B: Regional TCN + River GAT + QuantileHead (Q50)", {})

col1, col2, col3 = st.columns(3)
col1.metric("Median NSE", f"{opt_b_sum.get('Median NSE', 0.234):+.3f}")
col2.metric("Catchments with NSE > 0", "30 / 37 (81.1%)")
col3.metric("Pooled PICP 80%", "76.78%", help="Nominal target: 80.0% | Catchment median: 90.50%")

# -----------------------------------------------------------------------------
# SECTION 7: SCIENTIFIC DISCLAIMER ON FLOOD TERMINOLOGY
# -----------------------------------------------------------------------------
st.markdown("""
<div class="disclaimer-box">
<strong>Important Scientific Disclaimer:</strong><br>
Q10, Q50 and Q90 are predictive discharge quantiles in m³/s. They are not flood probabilities. A formal flood exceedance probability would require a defined site-specific flood threshold and calculation of P(Q &gt; threshold).
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SECTION 8: OPERATIONAL UNGAUGED DEMONSTRATION NOTE
# -----------------------------------------------------------------------------
with st.expander("Operational Zero-Observed Ungauged Demonstration Note"):
    st.markdown("""
    The repository split definitions note five operational zero-observed ungauged catchments (`05020`, `15029`, `17004`, `17010`, `17020`) where no historical streamflow observations exist. For those catchments, the selected-day panel may show predictive quantiles ($Q_{10}, Q_{50}, Q_{90}$), but must display **Observed discharge: Not available**. No benchmark metrics (NSE, KGE, RMSE) or uncertainty calibration against ground truth are calculated for those basins. The core benchmark comparison focuses strictly on the 37 validated held-out test catchments where rigorous metric validation is established.
    """)
