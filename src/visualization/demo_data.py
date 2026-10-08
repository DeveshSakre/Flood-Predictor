"""
Data access and caching layer for the reviewer-facing interactive demo.
Loads validated artifacts using repository-relative paths without altering source files.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import numpy as np
import pandas as pd

from src.evaluation.peak_metrics import peak_flow_error, peak_timing_error


def get_repo_root() -> Path:
    """Return repository root path relative to this file."""
    return Path(__file__).resolve().parent.parent.parent


def get_test_gauge_ids() -> List[str]:
    """
    Return sorted list of the 37 held-out unseen test catchment gauge IDs.
    All gauge IDs are 5-character zero-padded strings.
    """
    root = get_repo_root()
    comp_file = root / "results" / "comparison" / "final_per_catchment_model_comparison.csv"
    if comp_file.exists():
        df = pd.read_csv(comp_file, usecols=["gauge_id"])
        return sorted(df["gauge_id"].astype(str).str.zfill(5).unique().tolist())

    # Fallback to candidate unseen test catchments
    cand_file = root / "data" / "processed" / "candidate_unseen_test_catchments.csv"
    if cand_file.exists():
        df = pd.read_csv(cand_file, usecols=["gauge_id"])
        return sorted(df["gauge_id"].astype(str).str.zfill(5).unique().tolist())

    raise FileNotFoundError("Could not find test catchments list in repository.")


def load_catchment_metadata(gauge_id: str) -> Dict[str, Any]:
    """
    Load metadata for a specific test catchment.
    Returns:
        Dict with keys: gauge_id, basin, river, site_name, drainage_area_km2.
    """
    gid_str = str(gauge_id).zfill(5)
    root = get_repo_root()
    comp_file = root / "results" / "comparison" / "final_per_catchment_model_comparison.csv"

    if not comp_file.exists():
        raise FileNotFoundError(f"Missing comparison file: {comp_file}")

    df = pd.read_csv(comp_file)
    df["gauge_id"] = df["gauge_id"].astype(str).str.zfill(5)
    match = df[df["gauge_id"] == gid_str]

    if match.empty:
        raise ValueError(f"Catchment {gid_str} is not one of the 37 test catchments.")

    row = match.iloc[0]
    basin = row.get("river_basin", "Not available")
    river = row.get("cwc_river", "Not available")
    site = row.get("cwc_site_name", "Not available")
    area = row.get("cwc_area_km2", None)

    return {
        "gauge_id": gid_str,
        "basin": str(basin) if pd.notna(basin) else "Not available",
        "river": str(river) if pd.notna(river) else "Not available",
        "site_name": str(site) if pd.notna(site) else "Not available",
        "drainage_area_km2": float(area) if (pd.notna(area) and area is not None) else None,
    }


def load_option_b_predictions(
    gauge_id: str,
    predictions_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load Option B continuous predictions for a specific gauge ID.
    Columns: date, gauge_id, y_true, q10, q50, q90.
    Missing y_true values are preserved as NaN (never converted to 0.0).
    """
    gid_str = str(gauge_id).zfill(5)
    root = get_repo_root()
    pred_file = predictions_path or (root / "results" / "quantile" / "quantile_test_predictions.csv.gz")

    if not pred_file.exists():
        raise FileNotFoundError(f"Predictions file not found: {pred_file}")

    # Read predictions
    df = pd.read_csv(pred_file)
    df["gauge_id"] = df["gauge_id"].astype(str).str.zfill(5)

    sub = df[df["gauge_id"] == gid_str].copy()
    if sub.empty:
        raise ValueError(f"No predictions found for catchment {gid_str} in {pred_file.name}")

    sub["date"] = pd.to_datetime(sub["date"])
    sub = sub.sort_values("date").reset_index(drop=True)

    # Ensure numeric columns
    for col in ["y_true", "q10", "q50", "q90"]:
        sub[col] = pd.to_numeric(sub[col], errors="coerce")

    return sub


def load_xgboost_predictions(
    gauge_id: str,
    predictions_path: Optional[Path] = None,
) -> Optional[pd.DataFrame]:
    """
    Load XGBoost predictions for the selected gauge ID if available.
    """
    gid_str = str(gauge_id).zfill(5)
    root = get_repo_root()
    pred_file = predictions_path or (root / "results" / "baseline" / "xgboost_test_predictions.csv.gz")

    if not pred_file.exists():
        return None

    df = pd.read_csv(pred_file)
    df["gauge_id"] = df["gauge_id"].astype(str).str.zfill(5)
    sub = df[df["gauge_id"] == gid_str].copy()

    if sub.empty:
        return None

    sub["date"] = pd.to_datetime(sub["date"])
    sub = sub.sort_values("date").reset_index(drop=True)
    return sub


def load_per_catchment_metrics(gauge_id: str) -> Dict[str, Any]:
    """
    Load deterministic and uncertainty metrics for the selected catchment from
    results/comparison/final_per_catchment_model_comparison.csv.
    """
    gid_str = str(gauge_id).zfill(5)
    root = get_repo_root()
    comp_file = root / "results" / "comparison" / "final_per_catchment_model_comparison.csv"

    if not comp_file.exists():
        raise FileNotFoundError(f"Missing comparison file: {comp_file}")

    df = pd.read_csv(comp_file)
    df["gauge_id"] = df["gauge_id"].astype(str).str.zfill(5)
    match = df[df["gauge_id"] == gid_str]

    if match.empty:
        raise ValueError(f"Catchment {gid_str} not found in comparison metrics.")

    row = match.iloc[0].to_dict()

    # Calculate peak metrics on the fly if needed, or extract
    opt_b_preds = load_option_b_predictions(gid_str)
    obs = opt_b_preds["y_true"].to_numpy()
    q50 = opt_b_preds["q50"].to_numpy()
    p_err = peak_flow_error(obs, q50)
    t_err = peak_timing_error(obs, q50)

    # Compute observed 90th percentile threshold
    obs_valid = obs[~np.isnan(obs)]
    q90_threshold = float(np.percentile(obs_valid, 90.0)) if len(obs_valid) > 0 else None

    return {
        "metadata": {
            "gauge_id": gid_str,
            "basin": row.get("river_basin", "Not available"),
            "river": row.get("cwc_river", "Not available"),
            "site_name": row.get("cwc_site_name", "Not available"),
            "drainage_area_km2": row.get("cwc_area_km2", None),
        },
        "option_b_deterministic": {
            "nse": float(row.get("option_b_q50_nse", np.nan)),
            "kge": float(row.get("option_b_q50_kge", np.nan)),
            "pearson_r": float(row.get("option_b_q50_pearson_r", np.nan)),
            "rmse": float(row.get("option_b_q50_rmse", np.nan)),
            "mae": float(row.get("option_b_q50_mae", np.nan)),
            "pbias": float(row.get("option_b_q50_pbias", np.nan)),
            "high_flow_nse": float(row.get("option_b_q50_high_flow_nse", np.nan)),
            "peak_rel_error_pct": float(p_err["rel_error_pct"]),
            "peak_timing_days": int(t_err),
            "peak_obs_m3s": float(p_err["obs_peak"]),
            "peak_sim_m3s": float(p_err["sim_peak"]),
        },
        "option_b_uncertainty": {
            "picp_80": float(row.get("option_b_picp_80", np.nan)),
            "mpiw": float(row.get("option_b_mpiw", np.nan)),
            "winkler_score_80": float(row.get("option_b_winkler_80", np.nan)),
            "coverage_p10": float(row.get("option_b_coverage_p10", np.nan)),
            "coverage_p50": float(row.get("option_b_coverage_p50", np.nan)),
            "coverage_p90": float(row.get("option_b_coverage_p90", np.nan)),
            "crossing_violations_pct": float(row.get("option_b_crossing_violations_pct", np.nan)),
            "hf_picp_80": float(row.get("option_b_hf_picp_80", np.nan)),
            "hf_mpiw": float(row.get("option_b_hf_mpiw", np.nan)),
            "q90_threshold_m3s": q90_threshold,
        },
        "all_models_nse": {
            "Option B (Q50)": float(row.get("option_b_q50_nse", np.nan)),
            "Regional TCN": float(row.get("regional_tcn_nse", np.nan)),
            "Base TCN": float(row.get("base_tcn_nse", np.nan)),
            "XGBoost Baseline": float(row.get("xgboost_nse", np.nan)),
        },
        "all_models_kge": {
            "Option B (Q50)": float(row.get("option_b_q50_kge", np.nan)),
            "Regional TCN": float(row.get("regional_tcn_kge", np.nan)),
            "Base TCN": float(row.get("base_tcn_kge", np.nan)),
            "XGBoost Baseline": float(row.get("xgboost_kge", np.nan)),
        },
        "all_models_rmse": {
            "Option B (Q50)": float(row.get("option_b_q50_rmse", np.nan)),
            "Regional TCN": float(row.get("regional_tcn_rmse", np.nan)),
            "Base TCN": float(row.get("base_tcn_rmse", np.nan)),
            "XGBoost Baseline": float(row.get("xgboost_rmse", np.nan)),
        },
    }


def load_overall_model_summary() -> pd.DataFrame:
    """
    Load the overall 4-model comparison summary table.
    """
    root = get_repo_root()
    summary_file = root / "results" / "comparison" / "final_model_comparison_summary.csv"
    if not summary_file.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_file}")
    return pd.read_csv(summary_file)


def load_summary_json() -> Dict[str, Any]:
    """
    Load the structured summary JSON.
    """
    root = get_repo_root()
    json_file = root / "results" / "comparison" / "final_comparison_summary.json"
    if not json_file.exists():
        raise FileNotFoundError(f"Missing summary JSON: {json_file}")
    with open(json_file, "r") as f:
        return json.load(f)
