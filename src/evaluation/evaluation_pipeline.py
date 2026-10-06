"""
Evaluation pipeline for multi-catchment regional streamflow benchmarking.
Computes per-catchment performance metrics and summary statistics across unseen test basins.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_all_metrics
from src.evaluation.peak_metrics import peak_flow_error


def evaluate_catchment_predictions(
    predictions_df: pd.DataFrame,
    obs_col: str = "observed_flow",
    pred_col: str = "predicted_flow",
    group_col: str = "gauge_id",
) -> pd.DataFrame:
    """
    Compute hydrologic metrics per catchment across test set.

    Args:
        predictions_df: DataFrame containing observations and predictions.
        obs_col: Name of ground truth observed column.
        pred_col: Name of predicted column.
        group_col: Column indicating catchment identifier.

    Returns:
        DataFrame with one row per catchment and metric columns (NSE, KGE, RMSE, MAE, PBIAS, PeakError).
    """
    records = []

    for gid, group in predictions_df.groupby(group_col):
        obs = group[obs_col].to_numpy()
        pred = group[pred_col].to_numpy()

        metrics = compute_all_metrics(obs, pred)
        peak = peak_flow_error(obs, pred)

        metrics["gauge_id"] = str(gid).zfill(5)
        metrics["peak_diff_m3s"] = peak["diff_m3s"]
        metrics["peak_rel_error_pct"] = peak["rel_error_pct"]
        metrics["n_timesteps"] = len(obs)

        records.append(metrics)

    results_df = pd.DataFrame(records)
    return results_df


def summarize_regional_performance(results_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute median, interquartile range (IQR), and mean metrics across all unseen catchments.
    """
    summary = {}
    metric_cols = ["NSE", "KGE", "RMSE", "MAE", "PBIAS"]

    for col in metric_cols:
        if col in results_df.columns:
            valid_vals = results_df[col].dropna()
            summary[f"{col}_median"] = float(valid_vals.median())
            summary[f"{col}_q25"] = float(valid_vals.quantile(0.25))
            summary[f"{col}_q75"] = float(valid_vals.quantile(0.75))
            summary[f"{col}_mean"] = float(valid_vals.mean())
            summary[f"{col}_fraction_nse_gt_0"] = float((valid_vals > 0.0).mean())

    return summary
