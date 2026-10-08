"""
Final 4-Model Benchmark Comparison Protocol (Member 5).

Models Evaluated:
1. XGBoost Baseline (results/baseline/xgboost_test_metrics.csv & xgboost_test_predictions.csv.gz)
2. Base TCN Baseline (results/regional_tcn/test_metrics.csv)
3. Regional TCN (results/regional_tcn/regional_tcn_test_metrics.csv)
4. Option B: Regional TCN + River GAT + QuantileHead (results/quantile/quantile_test_predictions.csv.gz)
   - Evaluated as deterministic point forecast via Q50
   - Evaluated for uncertainty quantification via Q10/Q50/Q90, PICP, MPIW, and Winkler score.

Strictly evaluates across the 37 held-out unseen test catchments.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.metrics import compute_all_metrics
from src.evaluation.peak_metrics import peak_flow_error, peak_timing_error, high_flow_nse
from src.uncertainty.calibration import evaluate_uncertainty_forecasts


def run_final_model_comparison() -> Dict[str, Any]:
    root = REPO_ROOT
    comp_dir = root / "results" / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = comp_dir / "representative_hydrographs"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("STARTING FINAL 4-MODEL BENCHMARK COMPARISON PROTOCOL")
    print("Models: XGBoost Baseline | Base TCN | Regional TCN | Option B (Q50 + Uncertainty)")
    print("=" * 90)

    # 1. Load Catchment Metadata
    meta_file = root / "data" / "processed" / "candidate_unseen_test_catchments.csv"
    assert meta_file.exists(), f"Missing metadata: {meta_file}"
    df_meta = pd.read_csv(meta_file)
    df_meta["gauge_id"] = df_meta["gauge_id"].astype(str).str.zfill(5)
    df_meta = df_meta.set_index("gauge_id").sort_index()
    test_ids = sorted(list(df_meta.index))
    assert len(test_ids) == 37, f"Expected 37 catchments, got {len(test_ids)}"

    # 2. Load Existing Benchmark Metrics
    xgb_file = root / "results" / "baseline" / "xgboost_test_metrics.csv"
    base_file = root / "results" / "regional_tcn" / "test_metrics.csv"
    reg_file = root / "results" / "regional_tcn" / "regional_tcn_test_metrics.csv"

    assert xgb_file.exists(), f"Missing: {xgb_file}"
    assert base_file.exists(), f"Missing: {base_file}"
    assert reg_file.exists(), f"Missing: {reg_file}"

    df_xgb_full = pd.read_csv(xgb_file)
    df_base = pd.read_csv(base_file)
    df_reg = pd.read_csv(reg_file)

    df_xgb_full["gauge_id"] = df_xgb_full["gauge_id"].astype(str).str.zfill(5)
    df_base["gauge_id"] = df_base["gauge_id"].astype(str).str.zfill(5)
    df_reg["gauge_id"] = df_reg["gauge_id"].astype(str).str.zfill(5)

    df_xgb_full = df_xgb_full.set_index("gauge_id").sort_index()
    df_base = df_base.set_index("gauge_id").sort_index()
    df_reg = df_reg.set_index("gauge_id").sort_index()

    # 3. Load Continuous Option B Predictions and Evaluate
    q_pred_file = root / "results" / "quantile" / "quantile_test_predictions.csv.gz"
    assert q_pred_file.exists(), f"Missing Option B predictions: {q_pred_file}"
    print(f"Loading Option B predictions from {q_pred_file}...")
    df_q = pd.read_csv(q_pred_file)
    df_q["gauge_id"] = df_q["gauge_id"].astype(str).str.zfill(5)

    option_b_records = []
    for gid in test_ids:
        grp = df_q[df_q["gauge_id"] == gid].sort_values("date")
        obs = grp["y_true"].to_numpy()
        p10 = grp["q10"].to_numpy()
        p50 = grp["q50"].to_numpy()
        p90 = grp["q90"].to_numpy()

        # Deterministic Q50 metrics
        m = compute_all_metrics(obs, p50)
        p_err = peak_flow_error(obs, p50)
        t_err = peak_timing_error(obs, p50)
        hf = high_flow_nse(obs, p50)

        # Uncertainty metrics
        u = evaluate_uncertainty_forecasts(obs, p10, p50, p90)

        # High-Flow Uncertainty (top 10% observed flows)
        obs_clean = obs[~np.isnan(obs)]
        if len(obs_clean) > 0:
            q90_th = np.percentile(obs_clean, 90.0)
            valid_mask = ~np.isnan(obs)
            hf_mask = valid_mask & (obs >= q90_th)
            u_hf = evaluate_uncertainty_forecasts(obs[hf_mask], p10[hf_mask], p50[hf_mask], p90[hf_mask])
        else:
            u_hf = {"picp_80": np.nan, "mpiw": np.nan}

        option_b_records.append({
            "gauge_id": gid,
            **m,
            "high_flow_nse": hf,
            "peak_obs_m3s": p_err["obs_peak"],
            "peak_sim_m3s": p_err["sim_peak"],
            "peak_diff_m3s": p_err["diff_m3s"],
            "peak_rel_error_pct": p_err["rel_error_pct"],
            "peak_timing_days": t_err,
            "picp_80": u["picp_80"],
            "mpiw": u["mpiw"],
            "coverage_p10": u["coverage_p10"],
            "coverage_p50": u["coverage_p50"],
            "coverage_p90": u["coverage_p90"],
            "winkler_score_80": u["winkler_score_80"],
            "crossing_violations_pct": u["crossing_violations_pct"],
            "hf_picp_80": u_hf["picp_80"],
            "hf_mpiw": u_hf["mpiw"],
        })

    df_option_b = pd.DataFrame(option_b_records).set_index("gauge_id").sort_index()

    # 4. Load XGBoost Predictions and Compute Period-Matched (1999–2009) Metrics
    xgb_pred_file = root / "results" / "baseline" / "xgboost_test_predictions.csv.gz"
    assert xgb_pred_file.exists(), f"Missing XGBoost predictions: {xgb_pred_file}"
    print(f"Loading XGBoost predictions from {xgb_pred_file} for period matching...")
    df_xgb_preds = pd.read_csv(xgb_pred_file)
    df_xgb_preds["gauge_id"] = df_xgb_preds["gauge_id"].astype(str).str.zfill(5)

    test_dates = set(df_q["date"].unique())
    sub_xgb_preds = df_xgb_preds[df_xgb_preds["date"].isin(test_dates)]

    xgb_matched_records = []
    for gid in test_ids:
        grp = sub_xgb_preds[sub_xgb_preds["gauge_id"] == gid].sort_values("date")
        obs = grp["y_true"].to_numpy()
        pred = grp["y_pred"].to_numpy()
        m = compute_all_metrics(obs, pred)
        p_err = peak_flow_error(obs, pred)
        t_err = peak_timing_error(obs, pred)
        hf = high_flow_nse(obs, pred)
        xgb_matched_records.append({
            "gauge_id": gid,
            **m,
            "high_flow_nse": hf,
            "peak_obs_m3s": p_err["obs_peak"],
            "peak_sim_m3s": p_err["sim_peak"],
            "peak_diff_m3s": p_err["diff_m3s"],
            "peak_rel_error_pct": p_err["rel_error_pct"],
            "peak_timing_days": t_err,
        })

    df_xgb_matched = pd.DataFrame(xgb_matched_records).set_index("gauge_id").sort_index()

    # 5. Build Summary Comparison Table
    model_evaluations = {
        "Option B: Regional TCN + River GAT + QuantileHead (Q50)": {
            "df": df_option_b,
            "period": "1999-10-01 to 2009-09-30 (10 Hydrological Years)",
            "is_probabilistic": True,
        },
        "Regional TCN": {
            "df": df_reg,
            "period": "1980–2020 Available Record (Committed Benchmark)",
            "is_probabilistic": False,
        },
        "Base TCN": {
            "df": df_base,
            "period": "1980–2020 Available Record (Committed Benchmark)",
            "is_probabilistic": False,
        },
        "XGBoost Baseline (Full Period)": {
            "df": df_xgb_full,
            "period": "1980–2020 Available Record (Full Benchmark)",
            "is_probabilistic": False,
        },
        "XGBoost Baseline (Period-Matched)": {
            "df": df_xgb_matched,
            "period": "1999-10-01 to 2009-09-30 (Strict Period Match)",
            "is_probabilistic": False,
        },
    }

    summary_rows = []
    summary_dict = {}

    for name, info in model_evaluations.items():
        df = info["df"]
        nse = df["NSE"].dropna()
        kge = df["KGE"].dropna()
        rmse = df["RMSE"].dropna()
        mae = df["MAE"].dropna()
        pbias = df["PBIAS"].dropna()
        r = df["Pearson_r"].dropna()
        hf_nse = df["high_flow_nse"].dropna()
        peak_err = df["peak_rel_error_pct"].dropna()
        peak_time = df["peak_timing_days"].dropna()

        row = {
            "Model": name,
            "Evaluation Period": info["period"],
            "Median NSE": float(nse.median()),
            "Q25 NSE": float(nse.quantile(0.25)),
            "Q75 NSE": float(nse.quantile(0.75)),
            "Mean NSE": float(nse.mean()),
            "Median KGE": float(kge.median()),
            "Mean KGE": float(kge.mean()),
            "Median Pearson r": float(r.median()),
            "Mean Pearson r": float(r.mean()),
            "Median RMSE (m3/s)": float(rmse.median()),
            "Mean RMSE (m3/s)": float(rmse.mean()),
            "Median MAE (m3/s)": float(mae.median()),
            "Mean MAE (m3/s)": float(mae.mean()),
            "Median PBIAS (%)": float(pbias.median()),
            "Mean PBIAS (%)": float(pbias.mean()),
            "Median High-Flow NSE": float(hf_nse.median()),
            "Mean High-Flow NSE": float(hf_nse.mean()),
            "Median Peak Rel Err (%)": float(peak_err.median()),
            "Median Peak Timing (days)": float(peak_time.median()),
            "Count NSE > 0": int((nse > 0.0).sum()),
            "Fraction NSE > 0 (%)": float((nse > 0.0).mean() * 100.0),
            "Count NSE > 0.5": int((nse > 0.5).sum()),
            "Fraction NSE > 0.5 (%)": float((nse > 0.5).mean() * 100.0),
            "Count NSE > 0.7": int((nse > 0.7).sum()),
            "Fraction NSE > 0.7 (%)": float((nse > 0.7).mean() * 100.0),
        }

        if info["is_probabilistic"]:
            row["Median PICP 80% (%)"] = float(df["picp_80"].median() * 100.0)
            row["Median MPIW (m3/s)"] = float(df["mpiw"].median())
            row["Median Winkler Score"] = float(df["winkler_score_80"].median())
            row["Quantile Crossing Violations (%)"] = float(df["crossing_violations_pct"].mean())
            row["High-Flow PICP 80% (%)"] = float(df["hf_picp_80"].median() * 100.0)
            row["High-Flow MPIW (m3/s)"] = float(df["hf_mpiw"].median())
        else:
            row["Median PICP 80% (%)"] = np.nan
            row["Median MPIW (m3/s)"] = np.nan
            row["Median Winkler Score"] = np.nan
            row["Quantile Crossing Violations (%)"] = np.nan
            row["High-Flow PICP 80% (%)"] = np.nan
            row["High-Flow MPIW (m3/s)"] = np.nan

        summary_rows.append(row)
        summary_dict[name] = row

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(comp_dir / "final_model_comparison_summary.csv", index=False)
    with open(comp_dir / "final_comparison_summary.json", "w") as f:
        json.dump(summary_dict, f, indent=2)

    print("\n" + "=" * 100)
    print("FINAL 4-MODEL REGIONAL BENCHMARK COMPARISON (37 UNSEEN TEST BASINS):")
    print("=" * 100)
    disp_cols = [
        "Model", "Median NSE", "Mean NSE", "Median KGE", "Median Pearson r",
        "Median RMSE (m3/s)", "Median PBIAS (%)", "Median High-Flow NSE",
        "Count NSE > 0", "Count NSE > 0.5"
    ]
    print(summary_df[disp_cols].to_string(index=False))
    print("=" * 100 + "\n")

    # 6. Build Per-Catchment Comparison Table
    catchment_rows = []
    for gid in test_ids:
        meta = df_meta.loc[gid] if gid in df_meta.index else {}
        xgb_m = df_xgb_full.loc[gid]
        base_m = df_base.loc[gid]
        reg_m = df_reg.loc[gid]
        opt_b_m = df_option_b.loc[gid]

        c_row = {
            "gauge_id": gid,
            "river_basin": meta.get("river_basin", "Unknown"),
            "cwc_river": meta.get("cwc_river", "Unknown"),
            "cwc_site_name": meta.get("cwc_site_name", "Unknown"),
            "cwc_area_km2": meta.get("cwc_area", np.nan),
            # NSE
            "xgboost_nse": xgb_m["NSE"],
            "base_tcn_nse": base_m["NSE"],
            "regional_tcn_nse": reg_m["NSE"],
            "option_b_q50_nse": opt_b_m["NSE"],
            # KGE
            "xgboost_kge": xgb_m["KGE"],
            "base_tcn_kge": base_m["KGE"],
            "regional_tcn_kge": reg_m["KGE"],
            "option_b_q50_kge": opt_b_m["KGE"],
            # Pearson r
            "xgboost_pearson_r": xgb_m["Pearson_r"],
            "base_tcn_pearson_r": base_m["Pearson_r"],
            "regional_tcn_pearson_r": reg_m["Pearson_r"],
            "option_b_q50_pearson_r": opt_b_m["Pearson_r"],
            # RMSE
            "xgboost_rmse": xgb_m["RMSE"],
            "base_tcn_rmse": base_m["RMSE"],
            "regional_tcn_rmse": reg_m["RMSE"],
            "option_b_q50_rmse": opt_b_m["RMSE"],
            # MAE
            "xgboost_mae": xgb_m["MAE"],
            "base_tcn_mae": base_m["MAE"],
            "regional_tcn_mae": reg_m["MAE"],
            "option_b_q50_mae": opt_b_m["MAE"],
            # PBIAS
            "xgboost_pbias": xgb_m["PBIAS"],
            "base_tcn_pbias": base_m["PBIAS"],
            "regional_tcn_pbias": reg_m["PBIAS"],
            "option_b_q50_pbias": opt_b_m["PBIAS"],
            # High-Flow NSE
            "xgboost_high_flow_nse": xgb_m["high_flow_nse"],
            "base_tcn_high_flow_nse": base_m["high_flow_nse"],
            "regional_tcn_high_flow_nse": reg_m["high_flow_nse"],
            "option_b_q50_high_flow_nse": opt_b_m["high_flow_nse"],
            # Uncertainty Metrics (Option B)
            "option_b_picp_80": opt_b_m["picp_80"],
            "option_b_mpiw": opt_b_m["mpiw"],
            "option_b_winkler_80": opt_b_m["winkler_score_80"],
            "option_b_coverage_p10": opt_b_m["coverage_p10"],
            "option_b_coverage_p50": opt_b_m["coverage_p50"],
            "option_b_coverage_p90": opt_b_m["coverage_p90"],
            "option_b_crossing_violations_pct": opt_b_m["crossing_violations_pct"],
            "option_b_hf_picp_80": opt_b_m["hf_picp_80"],
            "option_b_hf_mpiw": opt_b_m["hf_mpiw"],
        }
        catchment_rows.append(c_row)

    per_catchment_df = pd.DataFrame(catchment_rows)
    per_catchment_df.to_csv(comp_dir / "final_per_catchment_model_comparison.csv", index=False)
    print(f"Per-catchment comparison saved to: {comp_dir / 'final_per_catchment_model_comparison.csv'}")

    # 7. Figure 1: 4-Model Empirical CDF of NSE
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    cdf_models = {
        "XGBoost Baseline": {"scores": df_xgb_full["NSE"].dropna(), "color": "#dc2626", "ls": "--"},
        "Base TCN": {"scores": df_base["NSE"].dropna(), "color": "#f59e0b", "ls": "-."},
        "Regional TCN": {"scores": df_reg["NSE"].dropna(), "color": "#2563eb", "ls": "-"},
        "Option B (Q50)": {"scores": df_option_b["NSE"].dropna(), "color": "#10b981", "ls": "-"},
    }

    for name, c_info in cdf_models.items():
        sorted_scores = np.sort(c_info["scores"].to_numpy())
        y_vals = np.linspace(0, 1, len(sorted_scores))
        ax.plot(
            sorted_scores, y_vals,
            label=f"{name} (Median NSE = {c_info['scores'].median():.2f})",
            color=c_info["color"],
            linestyle=c_info["ls"],
            linewidth=2.2,
        )

    ax.axvline(0, color="gray", linestyle=":", linewidth=1.2, label="NSE = 0 Baseline")
    ax.axvline(0.5, color="#059669", linestyle=":", linewidth=1.2, label="NSE = 0.5 Satisfactory")
    ax.set_xlabel("Nash-Sutcliffe Efficiency (NSE)", fontsize=11)
    ax.set_ylabel("Empirical Cumulative Frequency", fontsize=11)
    ax.set_title("Empirical CDF of NSE on 37 Unseen Test Catchments (4 Models)", fontsize=12, fontweight="bold")
    ax.set_xlim(-2.0, 1.0)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(comp_dir / "final_cumulative_nse_comparison.png", dpi=300)
    plt.close(fig)
    print(f"Saved: {comp_dir / 'final_cumulative_nse_comparison.png'}")

    # 8. Figure 2: Model Performance Bar Chart
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    names_short = ["XGBoost", "Base TCN", "Reg TCN", "Option B"]
    b_colors = ["#dc2626", "#f59e0b", "#2563eb", "#10b981"]

    # Median NSE
    median_nses = [
        float(df_xgb_full["NSE"].median()),
        float(df_base["NSE"].median()),
        float(df_reg["NSE"].median()),
        float(df_option_b["NSE"].median()),
    ]
    axes[0].bar(names_short, median_nses, color=b_colors, width=0.55)
    axes[0].set_title("Median NSE (Higher = Better)", fontweight="bold", fontsize=11)
    axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    for b, val in zip(axes[0].patches, median_nses):
        axes[0].text(b.get_x() + b.get_width() / 2, (val + 0.05 if val >= 0 else val - 0.25), f"{val:.2f}", ha="center", fontsize=9.5, fontweight="bold")

    # Median KGE
    median_kges = [
        float(df_xgb_full["KGE"].median()),
        float(df_base["KGE"].median()),
        float(df_reg["KGE"].median()),
        float(df_option_b["KGE"].median()),
    ]
    axes[1].bar(names_short, median_kges, color=b_colors, width=0.55)
    axes[1].set_title("Median KGE (Higher = Better)", fontweight="bold", fontsize=11)
    axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    for b, val in zip(axes[1].patches, median_kges):
        axes[1].text(b.get_x() + b.get_width() / 2, (val + 0.05 if val >= 0 else val - 0.3), f"{val:.2f}", ha="center", fontsize=9.5, fontweight="bold")

    # High-Flow NSE
    median_hfs = [
        float(df_xgb_full["high_flow_nse"].median()),
        float(df_base["high_flow_nse"].median()),
        float(df_reg["high_flow_nse"].median()),
        float(df_option_b["high_flow_nse"].median()),
    ]
    axes[2].bar(names_short, median_hfs, color=b_colors, width=0.55)
    axes[2].set_title("High-Flow NSE (Top 10% Floods)", fontweight="bold", fontsize=11)
    axes[2].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[2].grid(axis="y", linestyle="--", alpha=0.5)
    for b, val in zip(axes[2].patches, median_hfs):
        axes[2].text(b.get_x() + b.get_width() / 2, (val + 0.02 if val >= 0 else val - 0.04), f"{val:.2f}", ha="center", fontsize=9.5, fontweight="bold")

    # Catchments NSE > 0 & NSE > 0.5
    pos_counts = [
        int((df_xgb_full["NSE"] > 0).sum()),
        int((df_base["NSE"] > 0).sum()),
        int((df_reg["NSE"] > 0).sum()),
        int((df_option_b["NSE"] > 0).sum()),
    ]
    axes[3].bar(names_short, pos_counts, color=b_colors, width=0.55)
    axes[3].set_title("Catchments with NSE > 0 (out of 37)", fontweight="bold", fontsize=11)
    axes[3].axhline(37, color="gray", linestyle=":", label="Total Basins (37)")
    axes[3].set_ylim(0, 40)
    axes[3].grid(axis="y", linestyle="--", alpha=0.5)
    for b, val in zip(axes[3].patches, pos_counts):
        axes[3].text(b.get_x() + b.get_width() / 2, val + 1.0, f"{val}/37", ha="center", fontsize=9.5, fontweight="bold")

    fig.tight_layout()
    fig.savefig(comp_dir / "final_model_metric_comparison.png", dpi=300)
    plt.close(fig)
    print(f"Saved: {comp_dir / 'final_model_metric_comparison.png'}")

    # 9. Representative Hydrographs with Option B Q50 and P10-P90 Uncertainty Band
    rep_catchments = [
        ("08013", "Good Performance (Mahanadi | Area: 19,600 km2)", "final_representative_good_08013.png"),
        ("08029", "Average Performance (Mahanadi | Area: 8,760 km2)", "final_representative_average_08029.png"),
        ("12016", "Challenging Headwater (Narmada | Area: 2,292 km2)", "final_representative_difficult_12016.png"),
        ("08001", "Small Flashy Basin (Mahanadi | Area: 2,210 km2)", "final_representative_difficult_08001.png"),
    ]

    for gid, label, filename in rep_catchments:
        sub_q_gid = df_q[df_q["gauge_id"] == gid].sort_values("date")
        sub_xgb_gid = df_xgb_preds[(df_xgb_preds["gauge_id"] == gid) & (df_xgb_preds["date"].isin(test_dates))].sort_values("date")

        # Focus on a representative 2-year window (2007-06-01 to 2009-09-30) for visual clarity of peaks
        sub_q_window = sub_q_gid[(sub_q_gid["date"] >= "2007-06-01") & (sub_q_gid["date"] <= "2009-09-30")]
        sub_xgb_window = sub_xgb_gid[(sub_xgb_gid["date"] >= "2007-06-01") & (sub_xgb_gid["date"] <= "2009-09-30")]

        dates = pd.to_datetime(sub_q_window["date"])
        obs = sub_q_window["y_true"].values
        q10 = sub_q_window["q10"].values
        q50 = sub_q_window["q50"].values
        q90 = sub_q_window["q90"].values
        xgb_sim = sub_xgb_window["y_pred"].values if len(sub_xgb_window) == len(sub_q_window) else None

        fig, ax = plt.subplots(figsize=(13, 5))
        ax.plot(dates, obs, label="Observed Streamflow (Ground Truth)", color="black", linewidth=1.5, alpha=0.9)
        if xgb_sim is not None:
            ax.plot(dates, xgb_sim, label=f"XGBoost Baseline (NSE={df_xgb_full.loc[gid]['NSE']:.2f})", color="#dc2626", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.plot(dates, q50, label=f"Option B Q50 Median (NSE={df_option_b.loc[gid]['NSE']:.2f})", color="#0284c7", linewidth=1.6)
        ax.fill_between(dates, q10, q90, color="#38bdf8", alpha=0.35, label=f"Option B 80% Uncertainty Band [Q10, Q90] (PICP={df_option_b.loc[gid]['picp_80']*100:.1f}%)")

        ax.set_title(f"Streamflow Forecast & Uncertainty Ribbon — Catchment {gid}\n{label}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel("Streamflow (m³/s)", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", framealpha=0.92, fontsize=10)
        fig.tight_layout()
        fig.savefig(plots_dir / filename, dpi=300)
        plt.close(fig)
        print(f"Saved representative hydrograph: {plots_dir / filename}")

    print("\nFinal comparison protocol completed successfully!")
    return summary_dict


if __name__ == "__main__":
    run_final_model_comparison()
