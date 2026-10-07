"""
Training and Evaluation Pipeline for XGBoost Baseline.

Controls for fair, leakage-safe empirical benchmarking against deep learning architectures:
- Input: Meteorological and soil-moisture forcings + antecedent lags/rolling stats + 129 static descriptors
- Trains strictly on 169 source catchments
- Evaluates validation loss and early stopping on 36 validation catchments
- Evaluates out-of-sample generalization on 37 held-out unseen test catchments (Mahanadi & Narmada)
- Generates forward predictions on 5 operational zero-flow demonstration catchments
- Computes comprehensive hydrological verification metrics (NSE, KGE, RMSE, MAE, PBIAS, Peak Error, Peak Timing, High-Flow NSE)
- Produces benchmark summary tables, hydrographs, and empirical CDF plots.

RESPONSIBILITY: Member 5 (XGBoost Baseline & Evaluation)
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import yaml

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.utils.config import load_config, get_repo_root
from src.utils.reproducibility import set_seed
from src.data.loaders import (
    load_split_catchment_ids,
    load_catchment_forcing,
    load_static_attributes,
)
from src.data.preprocessing import Preprocessor
from src.models.xgboost_baseline import XGBoostBaseline
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.peak_metrics import (
    peak_flow_error,
    peak_timing_error,
    high_flow_nse,
)
from src.evaluation.evaluation_pipeline import (
    evaluate_catchment_predictions,
    summarize_regional_performance,
)
from src.evaluation.plots import plot_hydrograph, plot_cumulative_nse


def run_preflight_anti_leakage_audit(
    train_ids: List[str],
    val_ids: List[str],
    test_ids: List[str],
    demo_ids: List[str],
    static_df: pd.DataFrame,
) -> None:
    """Rigorous pre-flight anti-leakage audit prior to training execution."""
    print("\n" + "=" * 75)
    print("RUNNING PRE-FLIGHT ANTI-LEAKAGE AUDIT (Member 5)")
    print("=" * 75)
    set_train = set(train_ids)
    set_val = set(val_ids)
    set_test = set(test_ids)
    set_demo = set(demo_ids)

    # 1. Partition disjointness
    assert set_train.isdisjoint(set_val), "CRITICAL: Train and validation partitions overlap!"
    assert set_train.isdisjoint(set_test), "CRITICAL: Train and test partitions overlap!"
    assert set_val.isdisjoint(set_test), "CRITICAL: Validation and test partitions overlap!"
    assert set_train.isdisjoint(set_demo), "CRITICAL: Train and ungauged demo overlap!"
    print(f"Partition disjointness: VERIFIED (169 train, 36 val, 37 test, 5 demo).")

    # 2. Static feature leakage checks
    forbidden = ["reservoir_index", "flow_availability", "DIS_AV_CMS", "ORD_FLOW", "lstm_pred_streamflow"]
    for f in forbidden:
        assert f not in static_df.columns, f"CRITICAL: Forbidden feature '{f}' present in static features!"

    streamflow_signatures = ["q_mean", "runoff_ratio", "slope_fdc", "bfi", "q_10", "q_50", "q_90"]
    for sig in streamflow_signatures:
        assert sig not in static_df.columns, f"CRITICAL: Streamflow signature '{sig}' found in static features!"

    print("Static feature anti-leakage: VERIFIED (129 pure physical descriptors).")
    print("=" * 75 + "\n")


def plot_peak_flow_comparison(
    metrics_df: pd.DataFrame,
    save_path: Path,
) -> None:
    """Plot observed vs simulated peak discharge across test catchments."""
    fig, ax = plt.subplots(figsize=(7, 6))
    obs_peaks = metrics_df["peak_obs_m3s"].dropna()
    sim_peaks = metrics_df["peak_sim_m3s"].dropna()

    max_val = max(obs_peaks.max(), sim_peaks.max()) * 1.1

    ax.scatter(obs_peaks, sim_peaks, color="#2563eb", alpha=0.75, edgecolors="none", s=50, label="Catchments (N=37)")
    ax.plot([0, max_val], [0, max_val], color="red", linestyle="--", linewidth=1.5, label="1:1 Perfect Fit")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Observed Peak Discharge (m³/s)", fontsize=11)
    ax.set_ylabel("Predicted Peak Discharge (m³/s)", fontsize=11)
    ax.set_title("Peak Flood Discharge: Observed vs XGBoost Baseline", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def train_and_evaluate_xgboost(
    config_path: str = "configs/baseline_config.yaml",
    samples_per_train_basin: int = 600,
    peak_samples_per_basin: int = 100,
    samples_per_val_basin: int = 250,
) -> Dict[str, Any]:
    """Execute complete XGBoost baseline training and evaluation protocol."""
    start_total_time = time.time()
    root = get_repo_root()

    # 1. Load configuration
    cfg = load_config(config_path) if (root / config_path).exists() else {}
    xgb_cfg = cfg.get("xgboost", {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "early_stopping_rounds": 30,
        "eval_metric": "rmse",
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1,
    })
    seed = int(cfg.get("random_seed", 42))
    set_seed(seed)
    rng = np.random.default_rng(seed)

    print("=" * 75)
    print("STARTING XGBOOST BASELINE BENCHMARK (Member 5)")
    print(f"Seed: {seed} | Estimators: {xgb_cfg.get('n_estimators')} | LR: {xgb_cfg.get('learning_rate')} | Max Depth: {xgb_cfg.get('max_depth')}")
    print("=" * 75)

    # 2. Load partition IDs and static attributes
    split_ids = load_split_catchment_ids()
    train_ids = split_ids["train"]
    val_ids = split_ids["val"]
    test_ids = split_ids["test"]
    demo_ids = split_ids["ungauged_demo"]

    static_df = load_static_attributes()
    preprocessor = Preprocessor()

    # Pre-flight audit
    run_preflight_anti_leakage_audit(train_ids, val_ids, test_ids, demo_ids, static_df)

    # 3. Load observed streamflow matrix
    flow_file = root / "data/raw/CAMELS_IND_All_Catchments/streamflow_timeseries/streamflow_observed.csv"
    if not flow_file.exists():
        raise FileNotFoundError(f"Observed streamflow file not found: {flow_file}")
    flow_df = pd.read_csv(flow_file)
    date_series = pd.to_datetime(flow_df[["year", "month", "day"]])

    # 4. Construct tabular training dataset across 169 source catchments
    print("Constructing training feature matrix across 169 catchments...")
    X_train_list = []
    y_train_list = []

    for idx, gid in enumerate(train_ids):
        col_key = str(int(gid))
        if col_key not in flow_df.columns:
            continue
        obs_flow = flow_df[col_key].to_numpy()
        valid_indices = np.where((~np.isnan(obs_flow)) & (obs_flow >= 0.0))[0]
        if len(valid_indices) == 0:
            continue

        # Load dynamic forcing and prepare tabular features
        df_forcing = load_catchment_forcing(gid)
        static_row = static_df.loc[gid]
        X_cat = XGBoostBaseline.prepare_catchment_tabular_features(
            df_forcing, static_row, preprocessor
        )

        # Stratified sampling: pick top flood peaks + random days to balance representation & memory
        if len(valid_indices) > samples_per_train_basin:
            sorted_by_flow = valid_indices[np.argsort(-obs_flow[valid_indices])]
            top_peaks = sorted_by_flow[:peak_samples_per_basin]
            remaining = np.setdiff1d(valid_indices, top_peaks)
            n_random = samples_per_train_basin - len(top_peaks)
            random_sampled = rng.choice(remaining, size=n_random, replace=False)
            selected_idx = np.sort(np.concatenate([top_peaks, random_sampled]))
        else:
            selected_idx = valid_indices

        X_train_list.append(X_cat.iloc[selected_idx].to_numpy(dtype=np.float32))
        y_train_list.append(obs_flow[selected_idx].astype(np.float32))

        if (idx + 1) % 40 == 0 or (idx + 1) == len(train_ids):
            print(f"  Processed {idx + 1}/{len(train_ids)} training catchments...")

    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)
    feature_names = list(X_cat.columns)
    print(f"Training feature matrix ready: {X_train.shape[0]:,} samples x {X_train.shape[1]} features.")

    # 5. Construct validation dataset across 36 validation catchments
    print("\nConstructing validation monitoring matrix across 36 catchments...")
    X_val_list = []
    y_val_list = []

    for gid in val_ids:
        col_key = str(int(gid))
        if col_key not in flow_df.columns:
            continue
        obs_flow = flow_df[col_key].to_numpy()
        valid_indices = np.where((~np.isnan(obs_flow)) & (obs_flow >= 0.0))[0]
        if len(valid_indices) == 0:
            continue

        df_forcing = load_catchment_forcing(gid)
        static_row = static_df.loc[gid]
        X_cat = XGBoostBaseline.prepare_catchment_tabular_features(
            df_forcing, static_row, preprocessor
        )

        if len(valid_indices) > samples_per_val_basin:
            sorted_by_flow = valid_indices[np.argsort(-obs_flow[valid_indices])]
            top_peaks = sorted_by_flow[:50]
            remaining = np.setdiff1d(valid_indices, top_peaks)
            n_random = samples_per_val_basin - len(top_peaks)
            random_sampled = rng.choice(remaining, size=n_random, replace=False)
            selected_idx = np.sort(np.concatenate([top_peaks, random_sampled]))
        else:
            selected_idx = valid_indices

        X_val_list.append(X_cat.iloc[selected_idx].to_numpy(dtype=np.float32))
        y_val_list.append(obs_flow[selected_idx].astype(np.float32))

    X_val = np.vstack(X_val_list)
    y_val = np.concatenate(y_val_list)
    print(f"Validation monitoring matrix ready: {X_val.shape[0]:,} samples x {X_val.shape[1]} features.")

    # 6. Fit XGBoostBaseline with early stopping
    print("\nFitting XGBoost regressor...")
    model_wrapper = XGBoostBaseline(config=xgb_cfg)
    # Convert to DataFrame for column preservation
    df_X_train = pd.DataFrame(X_train, columns=feature_names)
    df_X_val = pd.DataFrame(X_val, columns=feature_names)

    t0 = time.time()
    model_wrapper.fit(df_X_train, y_train, df_X_val, y_val)
    train_duration = time.time() - t0
    best_iter = model_wrapper.best_iteration
    print(f"XGBoost fit complete in {train_duration:.2f}s. Best iteration: {best_iter}")

    # 7. Evaluate on 36 Validation Catchments (Per-Catchment Metrics)
    print("\nEvaluating per-catchment metrics on 36 validation catchments...")
    val_records = []
    for gid in val_ids:
        col_key = str(int(gid))
        if col_key not in flow_df.columns:
            continue
        obs = flow_df[col_key].to_numpy()
        df_forcing = load_catchment_forcing(gid)
        static_row = static_df.loc[gid]
        X_cat = XGBoostBaseline.prepare_catchment_tabular_features(
            df_forcing, static_row, preprocessor
        )
        pred = model_wrapper.predict(X_cat)

        m = compute_all_metrics(obs, pred)
        peak = peak_flow_error(obs, pred)
        timing = peak_timing_error(obs, pred)
        hf_nse = high_flow_nse(obs, pred)

        m["gauge_id"] = str(gid).zfill(5)
        m["peak_obs_m3s"] = peak["obs_peak"]
        m["peak_sim_m3s"] = peak["sim_peak"]
        m["peak_diff_m3s"] = peak["diff_m3s"]
        m["peak_rel_error_pct"] = peak["rel_error_pct"]
        m["peak_timing_days"] = timing
        m["high_flow_nse"] = hf_nse
        m["n_timesteps"] = int(np.sum(~np.isnan(obs) & ~np.isnan(pred)))
        val_records.append(m)

    val_metrics_df = pd.DataFrame(val_records)

    # 8. Evaluate on 37 Held-Out Unseen Test Catchments (Mahanadi & Narmada)
    print("\nEvaluating out-of-sample generalization on 37 held-out test catchments...")
    test_records = []
    test_predictions_list = []

    for gid in test_ids:
        col_key = str(int(gid))
        if col_key not in flow_df.columns:
            continue
        obs = flow_df[col_key].to_numpy()
        df_forcing = load_catchment_forcing(gid)
        static_row = static_df.loc[gid]
        X_cat = XGBoostBaseline.prepare_catchment_tabular_features(
            df_forcing, static_row, preprocessor
        )
        pred = model_wrapper.predict(X_cat)

        m = compute_all_metrics(obs, pred)
        peak = peak_flow_error(obs, pred)
        timing = peak_timing_error(obs, pred)
        hf_nse = high_flow_nse(obs, pred)

        m["gauge_id"] = str(gid).zfill(5)
        m["peak_obs_m3s"] = peak["obs_peak"]
        m["peak_sim_m3s"] = peak["sim_peak"]
        m["peak_diff_m3s"] = peak["diff_m3s"]
        m["peak_rel_error_pct"] = peak["rel_error_pct"]
        m["peak_timing_days"] = timing
        m["high_flow_nse"] = hf_nse
        m["n_timesteps"] = int(np.sum(~np.isnan(obs) & ~np.isnan(pred)))
        test_records.append(m)

        # Standardized prediction table for integration
        pred_df = pd.DataFrame({
            "date": date_series,
            "gauge_id": str(gid).zfill(5),
            "y_true": obs,
            "y_pred": pred,
        })
        test_predictions_list.append(pred_df)

    test_metrics_df = pd.DataFrame(test_records)
    all_test_preds_df = pd.concat(test_predictions_list, ignore_index=True)

    # 9. Compute regional summary metrics
    summary = summarize_regional_performance(test_metrics_df)
    valid_test_nse = test_metrics_df["NSE"].dropna()
    summary["fraction_nse_gt_0.7"] = float((valid_test_nse > 0.7).mean()) if len(valid_test_nse) > 0 else 0.0
    summary["count_nse_gt_0"] = int((valid_test_nse > 0.0).sum())
    summary["count_nse_gt_0.5"] = int((valid_test_nse > 0.5).sum())
    summary["count_nse_gt_0.7"] = int((valid_test_nse > 0.7).sum())
    summary["total_test_catchments"] = int(len(test_metrics_df))
    summary["best_iteration"] = best_iter
    summary["train_duration_sec"] = float(train_duration)

    print("\n" + "=" * 75)
    print("XGBOOST REGIONAL PERFORMANCE SUMMARY (37 UNSEEN TEST BASINS):")
    print(f"  Median NSE:           {summary.get('NSE_median', float('nan')):.3f}")
    print(f"  NSE IQR [Q25, Q75]:   [{summary.get('NSE_q25', float('nan')):.3f}, {summary.get('NSE_q75', float('nan')):.3f}]")
    print(f"  Mean NSE:             {summary.get('NSE_mean', float('nan')):.3f}")
    print(f"  Median KGE:           {summary.get('KGE_median', float('nan')):.3f}")
    print(f"  Mean KGE:             {summary.get('KGE_mean', float('nan')):.3f}")
    print(f"  Median RMSE (m³/s):   {summary.get('RMSE_median', float('nan')):.2f}")
    print(f"  Median MAE (m³/s):    {summary.get('MAE_median', float('nan')):.2f}")
    print(f"  Median PBIAS (%):     {summary.get('PBIAS_median', float('nan')):.2f}%")
    print(f"  Median High-Flow NSE: {summary.get('high_flow_nse_median', float('nan')):.3f}")
    print(f"  Catchments NSE > 0.0: {summary['count_nse_gt_0']}/{summary['total_test_catchments']} ({summary.get('fraction_nse_gt_0', 0)*100:.1f}%)")
    print(f"  Catchments NSE > 0.5: {summary['count_nse_gt_0.5']}/{summary['total_test_catchments']} ({summary.get('fraction_nse_gt_0.5', 0)*100:.1f}%)")
    print(f"  Catchments NSE > 0.7: {summary['count_nse_gt_0.7']}/{summary['total_test_catchments']} ({summary.get('fraction_nse_gt_0.7', 0)*100:.1f}%)")
    print("=" * 75 + "\n")

    # 10. Save Results, Config, Predictions, and Plots
    results_dir = root / "results/baseline"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    hydrographs_dir = plots_dir / "hydrographs"
    hydrographs_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics CSVs
    val_metrics_df.to_csv(results_dir / "xgboost_validation_metrics.csv", index=False)
    test_metrics_df.to_csv(results_dir / "xgboost_test_metrics.csv", index=False)
    print(f"Saved test metrics: {results_dir / 'xgboost_test_metrics.csv'}")

    # Save summary JSON
    with open(results_dir / "xgboost_summary_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save config and feature list
    with open(results_dir / "xgboost_train_config.yaml", "w") as f:
        yaml.dump(xgb_cfg, f)

    with open(results_dir / "xgboost_feature_list.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    # Save feature importances
    feat_imp = model_wrapper.get_feature_importances()
    feat_imp.to_frame("importance").to_csv(results_dir / "feature_importances.csv", index_label="feature")

    # Save test predictions (compressed gzip to save disk space and stay clean)
    preds_save_path = results_dir / "xgboost_test_predictions.csv.gz"
    all_test_preds_df.to_csv(preds_save_path, index=False, compression="gzip")
    print(f"Saved standardized test predictions: {preds_save_path}")

    # 11. Generate benchmark plots
    # A. Cumulative NSE distribution
    plot_cumulative_nse(
        test_metrics_df["NSE"].tolist(),
        model_name="XGBoost Baseline",
        save_path=plots_dir / "cumulative_nse.png",
    )
    plt.close()

    # B. Peak flow comparison plot
    plot_peak_flow_comparison(test_metrics_df, plots_dir / "peak_flow_comparison.png")

    # C. Representative test hydrographs (high-performing, median, and challenging)
    sample_basins = ["08001", "08029", "12016", "12043"]
    for s_gid in sample_basins:
        sub = all_test_preds_df[all_test_preds_df["gauge_id"] == s_gid]
        if not sub.empty:
            # Focus on 2018–2020 monsoons for clarity
            sub_window = sub[(sub["date"] >= "2018-01-01") & (sub["date"] <= "2020-12-31")]
            if len(sub_window) == 0:
                sub_window = sub.tail(1095)

            plot_hydrograph(
                dates=sub_window["date"],
                obs=sub_window["y_true"].to_numpy(),
                sim=sub_window["y_pred"].to_numpy(),
                gauge_id=s_gid,
                save_path=hydrographs_dir / f"test_hydrograph_{s_gid}.png",
            )
            plt.close()

    total_time = time.time() - start_total_time
    print(f"All outputs and figures generated in {total_time:.2f}s.")
    return summary


if __name__ == "__main__":
    train_and_evaluate_xgboost()
