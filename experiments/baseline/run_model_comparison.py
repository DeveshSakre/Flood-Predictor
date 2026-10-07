"""
Multi-Model Benchmark Comparison Protocol (Member 5).

Compares:
1. XGBoost Baseline (results/baseline/xgboost_test_metrics.csv)
2. Base TCN Baseline (results/regional_tcn/test_metrics.csv)
3. Regional TCN (results/regional_tcn/regional_tcn_test_metrics.csv)

Strictly evaluates all three models across the same 37 held-out unseen test catchments
(Mahanadi & Narmada) using standardized hydrological metrics.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def run_multi_model_comparison() -> Dict[str, Any]:
    root = repo_root
    comp_dir = root / "results/comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = comp_dir / "representative_hydrographs"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load test metrics
    xgb_file = root / "results/baseline/xgboost_test_metrics.csv"
    base_file = root / "results/regional_tcn/test_metrics.csv"
    reg_file = root / "results/regional_tcn/regional_tcn_test_metrics.csv"
    meta_file = root / "data/processed/candidate_unseen_test_catchments.csv"

    assert xgb_file.exists(), f"Missing XGBoost test metrics: {xgb_file}"
    assert base_file.exists(), f"Missing Base TCN test metrics: {base_file}"
    assert reg_file.exists(), f"Missing Regional TCN test metrics: {reg_file}"
    assert meta_file.exists(), f"Missing test metadata: {meta_file}"

    df_xgb = pd.read_csv(xgb_file)
    df_base = pd.read_csv(base_file)
    df_reg = pd.read_csv(reg_file)
    df_meta = pd.read_csv(meta_file)

    # Standardize gauge IDs to 5 digits
    df_xgb["gauge_id"] = df_xgb["gauge_id"].astype(str).str.zfill(5)
    df_base["gauge_id"] = df_base["gauge_id"].astype(str).str.zfill(5)
    df_reg["gauge_id"] = df_reg["gauge_id"].astype(str).str.zfill(5)
    df_meta["gauge_id"] = df_meta["gauge_id"].astype(str).str.zfill(5)

    # Set gauge_id index for alignment
    df_xgb = df_xgb.set_index("gauge_id").sort_index()
    df_base = df_base.set_index("gauge_id").sort_index()
    df_reg = df_reg.set_index("gauge_id").sort_index()
    df_meta = df_meta.set_index("gauge_id").sort_index()

    # 2. Verify common test catchment intersection
    common_ids = sorted(list(set(df_xgb.index) & set(df_base.index) & set(df_reg.index)))
    print(f"Common test catchments: {len(common_ids)} / 37.")
    assert len(common_ids) == 37, f"Expected 37 common catchments, got {len(common_ids)}"

    models = {
        "XGBoost Baseline": df_xgb.loc[common_ids],
        "Base TCN": df_base.loc[common_ids],
        "Regional TCN": df_reg.loc[common_ids],
    }

    # 3. Calculate Regional Performance Summary Table
    summary_rows = []
    summary_dict = {}

    for name, df in models.items():
        nse = df["NSE"].dropna()
        kge = df["KGE"].dropna()
        rmse = df["RMSE"].dropna()
        mae = df["MAE"].dropna()
        pbias = df["PBIAS"].dropna()
        r = df["Pearson_r"].dropna()
        hf_nse = df["high_flow_nse"].dropna()
        peak_diff = df["peak_diff_m3s"].dropna()
        peak_err = df["peak_rel_error_pct"].dropna()
        peak_time = df["peak_timing_days"].dropna()

        row = {
            "Model": name,
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
        summary_rows.append(row)
        summary_dict[name] = row

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(comp_dir / "model_comparison_summary.csv", index=False)
    with open(comp_dir / "comparison_summary.json", "w") as f:
        json.dump(summary_dict, f, indent=2)

    print("\n" + "=" * 95)
    print("OFFICIAL THREE-MODEL REGIONAL BENCHMARK COMPARISON (37 UNSEEN TEST BASINS):")
    print("=" * 95)
    disp_cols = [
        "Model", "Median NSE", "Mean NSE", "Median KGE", "Median Pearson r",
        "Median RMSE (m3/s)", "Median PBIAS (%)", "Median High-Flow NSE",
        "Count NSE > 0", "Count NSE > 0.5"
    ]
    print(summary_df[disp_cols].to_string(index=False))
    print("=" * 95 + "\n")

    # 4. Build Per-Catchment Comparison Table
    catchment_rows = []
    for gid in common_ids:
        meta = df_meta.loc[gid] if gid in df_meta.index else {}
        xgb_m = df_xgb.loc[gid]
        base_m = df_base.loc[gid]
        reg_m = df_reg.loc[gid]

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
            # KGE
            "xgboost_kge": xgb_m["KGE"],
            "base_tcn_kge": base_m["KGE"],
            "regional_tcn_kge": reg_m["KGE"],
            # Pearson r
            "xgboost_pearson_r": xgb_m["Pearson_r"],
            "base_tcn_pearson_r": base_m["Pearson_r"],
            "regional_tcn_pearson_r": reg_m["Pearson_r"],
            # RMSE
            "xgboost_rmse": xgb_m["RMSE"],
            "base_tcn_rmse": base_m["RMSE"],
            "regional_tcn_rmse": reg_m["RMSE"],
            # MAE
            "xgboost_mae": xgb_m["MAE"],
            "base_tcn_mae": base_m["MAE"],
            "regional_tcn_mae": reg_m["MAE"],
            # PBIAS
            "xgboost_pbias": xgb_m["PBIAS"],
            "base_tcn_pbias": base_m["PBIAS"],
            "regional_tcn_pbias": reg_m["PBIAS"],
            # High-Flow NSE
            "xgboost_high_flow_nse": xgb_m["high_flow_nse"],
            "base_tcn_high_flow_nse": base_m["high_flow_nse"],
            "regional_tcn_high_flow_nse": reg_m["high_flow_nse"],
            # Peak Relative Error %
            "xgboost_peak_rel_err_pct": xgb_m["peak_rel_error_pct"],
            "base_tcn_peak_rel_err_pct": base_m["peak_rel_error_pct"],
            "regional_tcn_peak_rel_err_pct": reg_m["peak_rel_error_pct"],
            # Peak Timing Offset (days)
            "xgboost_peak_timing_days": xgb_m["peak_timing_days"],
            "base_tcn_peak_timing_days": base_m["peak_timing_days"],
            "regional_tcn_peak_timing_days": reg_m["peak_timing_days"],
        }
        catchment_rows.append(c_row)

    per_catchment_df = pd.DataFrame(catchment_rows)
    per_catchment_df.to_csv(comp_dir / "per_catchment_model_comparison.csv", index=False)
    print(f"Per-catchment comparison saved to: {comp_dir / 'per_catchment_model_comparison.csv'}")

    # 5. Figure 1: Multi-Model Cumulative NSE Empirical CDF
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = {
        "XGBoost Baseline": "#dc2626",      # Red
        "Base TCN": "#f59e0b",              # Amber
        "Regional TCN": "#2563eb",          # Blue
    }
    linestyles = {
        "XGBoost Baseline": "--",
        "Base TCN": "-.",
        "Regional TCN": "-",
    }

    for name, df in models.items():
        scores = np.sort(df["NSE"].dropna().to_numpy())
        y_vals = np.linspace(0, 1, len(scores))
        ax.plot(
            scores, y_vals,
            label=f"{name} (Median NSE = {df['NSE'].median():.2f})",
            color=colors[name],
            linestyle=linestyles[name],
            linewidth=2.2,
        )

    ax.axvline(0, color="gray", linestyle=":", linewidth=1.2, label="NSE = 0 Baseline")
    ax.axvline(0.5, color="#10b981", linestyle=":", linewidth=1.2, label="NSE = 0.5 Satisfactory")
    ax.set_xlabel("Nash-Sutcliffe Efficiency (NSE)", fontsize=11)
    ax.set_ylabel("Empirical Cumulative Frequency", fontsize=11)
    ax.set_title("Empirical CDF of NSE on 37 Unseen Test Catchments", fontsize=12, fontweight="bold")
    ax.set_xlim(-2.0, 1.0)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=10)
    fig.tight_layout()
    fig.savefig(comp_dir / "cumulative_nse_comparison.png", dpi=300)
    plt.close(fig)
    print(f"Cumulative NSE CDF saved to: {comp_dir / 'cumulative_nse_comparison.png'}")

    # 6. Figure 2: Model Performance Bar/Summary Plot
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))

    model_names_short = ["XGBoost", "Base TCN", "Regional TCN"]
    bar_colors = ["#dc2626", "#f59e0b", "#2563eb"]

    # Metric 1: Median NSE
    median_nses = [summary_dict[m]["Median NSE"] for m in models.keys()]
    axes[0].bar(model_names_short, median_nses, color=bar_colors, width=0.55)
    axes[0].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_title("Median NSE (Higher is Better)", fontweight="bold", fontsize=11)
    axes[0].set_ylabel("NSE")
    for i, v in enumerate(median_nses):
        axes[0].text(i, v + (0.1 if v < 0 else 0.03), f"{v:.2f}", ha="center", fontweight="bold", fontsize=10)

    # Metric 2: Median KGE
    median_kges = [summary_dict[m]["Median KGE"] for m in models.keys()]
    axes[1].bar(model_names_short, median_kges, color=bar_colors, width=0.55)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_title("Median KGE (Higher is Better)", fontweight="bold", fontsize=11)
    axes[1].set_ylabel("KGE")
    for i, v in enumerate(median_kges):
        axes[1].text(i, v + (0.2 if v < 0 else 0.03), f"{v:.2f}", ha="center", fontweight="bold", fontsize=10)

    # Metric 3: Median High-Flow NSE
    median_hfs = [summary_dict[m]["Median High-Flow NSE"] for m in models.keys()]
    axes[2].bar(model_names_short, median_hfs, color=bar_colors, width=0.55)
    axes[2].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[2].set_title("High-Flow NSE (Flood Peaks)", fontweight="bold", fontsize=11)
    axes[2].set_ylabel("High-Flow NSE")
    for i, v in enumerate(median_hfs):
        axes[2].text(i, v + (0.05 if v < 0 else 0.02), f"{v:.2f}", ha="center", fontweight="bold", fontsize=10)

    # Metric 4: Catchment Count with NSE > 0 and NSE > 0.5
    gt0 = [summary_dict[m]["Count NSE > 0"] for m in models.keys()]
    gt05 = [summary_dict[m]["Count NSE > 0.5"] for m in models.keys()]
    x_pos = np.arange(len(model_names_short))
    w = 0.35
    axes[3].bar(x_pos - w/2, gt0, width=w, color="#3b82f6", label="NSE > 0")
    axes[3].bar(x_pos + w/2, gt05, width=w, color="#10b981", label="NSE > 0.5")
    axes[3].set_xticks(x_pos)
    axes[3].set_xticklabels(model_names_short)
    axes[3].set_title("Catchments with Positive NSE (N=37)", fontweight="bold", fontsize=11)
    axes[3].set_ylabel("Number of Catchments")
    axes[3].set_ylim(0, 37)
    axes[3].legend(loc="upper left")
    for i in range(len(model_names_short)):
        axes[3].text(x_pos[i] - w/2, gt0[i] + 0.8, str(gt0[i]), ha="center", fontsize=9, fontweight="bold")
        axes[3].text(x_pos[i] + w/2, gt05[i] + 0.8, str(gt05[i]), ha="center", fontsize=9, fontweight="bold")

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    fig.suptitle("Comparative Evaluation Across 37 Unseen Test Catchments (Mahanadi & Narmada)", fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(comp_dir / "model_metric_comparison.png", dpi=300)
    plt.close(fig)
    print(f"Summary bar comparison saved to: {comp_dir / 'model_metric_comparison.png'}")

    return summary_dict


if __name__ == "__main__":
    run_multi_model_comparison()
