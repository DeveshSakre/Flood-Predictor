"""
Continuous Daily Inference Pipeline for Member 4 Option B Quantile Regression Model.

Architecture:
CAMELS-IND Dynamic Meteorology [B, 64, 20] + Static Catchment Descriptors [B, 129]
                        ↓
        FROZEN Regional TCN (mode='embedding') → [B, 128]
                        +
HydroRIVERS Graph Topology [Nodes, Edges]
                        ↓
        FROZEN River GAT (models/gat/river_gat_encoder.pt) → [B, 64]
                        ↓
        Catchment-Aligned Concatenation → [B, 192]
                        ↓
        TRAINED QuantileHead (models/quantile/option_b_quantile_head_best.pt)
                        ↓
        Normalized [P10, P50, P90]
                        ↓
        Physical Rescaling (y * target_std + target_mean) & Non-negative clipping
                        ↓
        Physical Streamflow Quantiles [Q10, Q50, Q90] in m3/s

Target Outputs:
- results/quantile/quantile_test_predictions.csv.gz
- results/quantile/continuous_inference_report.json
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import get_repo_root
from src.utils.reproducibility import set_seed
from src.data.loaders import (
    load_catchment_forcing,
    load_static_attributes,
)
from src.data.preprocessing import Preprocessor
from src.models.regional_tcn import RegionalTCN
from src.models.gat_config import GATConfig
from src.models.river_gat import RiverGAT
from src.uncertainty.quantile_regression import QuantileHead

FEATURE_COLS = [
    "prcp(mm/day)", "tmax(C)", "tmin(C)", "srad_lw(w/m2)", "srad_sw(w/m2)",
    "wind_u(m/s)", "wind_v(m/s)", "rel_hum(%)", "pet_gleam(mm/day)", "aet_gleam(mm/day)",
    "evap_canopy(mm/day)", "evap_surface(mm/day)", "sm_lvl1(kg/m2)", "sm_lvl2(kg/m2)",
    "sm_lvl3(kg/m2)", "sm_lvl4(kg/m2)", "sin_month", "cos_month", "sin_doy", "cos_doy"
]


def extract_catchment_matrix(df_raw: pd.DataFrame, preprocessor: Preprocessor) -> np.ndarray:
    """Transform dynamic forcings and add cyclical calendar encodings."""
    df_norm = preprocessor.transform_dynamic(df_raw)
    dt = pd.to_datetime(df_raw[["year", "month", "day"]])
    df_norm["sin_month"] = np.sin(2 * np.pi * dt.dt.month / 12.0)
    df_norm["cos_month"] = np.cos(2 * np.pi * dt.dt.month / 12.0)
    df_norm["sin_doy"] = np.sin(2 * np.pi * dt.dt.dayofyear / 365.25)
    df_norm["cos_doy"] = np.cos(2 * np.pi * dt.dt.dayofyear / 365.25)
    return df_norm[FEATURE_COLS].to_numpy().astype(np.float32)


def extract_test_gat_embeddings(
    gat_model: RiverGAT,
    node_scaler: Any,
    edge_scaler: Any,
    data_dir: Path,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Extract spatial embeddings [37, 64] from River-Network test graph."""
    nodes_csv = data_dir / "graph" / "nodes_test.csv"
    edges_csv = data_dir / "graph" / "edges_test.csv"

    nodes_df = pd.read_csv(nodes_csv)
    edges_df = pd.read_csv(edges_csv)

    node_cols = ["UPLAND_SKM", "CATCH_SKM", "DIST_DN_KM", "DIST_UP_KM", "ORD_STRA", "ORD_CLAS"]
    node_raw = nodes_df[node_cols].to_numpy().astype(np.float32)
    node_scaled = node_scaler.transform(node_raw)
    x_nodes = torch.tensor(node_scaled, dtype=torch.float32)

    src = torch.tensor(edges_df["source_idx"].to_numpy(), dtype=torch.long)
    dst = torch.tensor(edges_df["target_idx"].to_numpy(), dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)

    edge_cols = ["routing_distance_km", "steps", "drainage_area_ratio", "stream_order_delta"]
    edge_raw = edges_df[edge_cols].to_numpy().astype(np.float32)
    edge_scaled = edge_scaler.transform(edge_raw)
    edge_attr = torch.tensor(edge_scaled, dtype=torch.float32)

    with torch.no_grad():
        gat_emb = gat_model(x_nodes, edge_index, edge_attr)

    nodes_df["gauge_id_str"] = nodes_df["gauge_id"].astype(str).str.zfill(5)
    gid_to_idx = {gid: idx for idx, gid in enumerate(nodes_df["gauge_id_str"])}

    return gat_emb, gid_to_idx


def run_continuous_inference():
    start_time = time.time()
    set_seed(42)

    print("=" * 80)
    print("CONTINUOUS DAILY INFERENCE: OPTION B QUANTILE REGRESSION MODEL")
    print("Test Period: 1999-10-01 to 2009-09-30 (10 Water Years | 3,653 days)")
    print("=" * 80)

    repo_root = get_repo_root()
    data_dir = repo_root / "data" / "processed"
    results_dir = repo_root / "results" / "quantile"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Frozen Test Catchments
    test_catchments_file = data_dir / "candidate_unseen_test_catchments.csv"
    assert test_catchments_file.exists(), f"Missing {test_catchments_file}"
    test_df = pd.read_csv(test_catchments_file)
    test_ids = [str(gid).zfill(5) for gid in test_df["gauge_id"]]
    assert len(test_ids) == 37, f"Expected 37 test catchments, got {len(test_ids)}"
    n_mahanadi = (test_df["river_basin"] == "Mahanadi").sum()
    n_narmada = (test_df["river_basin"] == "Narmada").sum()
    print(f"Verified test catchments: {len(test_ids)} total (Mahanadi: {n_mahanadi}, Narmada: {n_narmada}).")

    # 2. Check & Load Checkpoints
    tcn_path = repo_root / "models" / "regional_tcn" / "regional_tcn_best.pt"
    gat_path = repo_root / "models" / "gat" / "river_gat_encoder.pt"
    edge_scaler_path = repo_root / "models" / "gat" / "edge_scaler.joblib"
    node_scaler_path = data_dir / "scalers" / "graph_node_scaler.joblib"
    q_path = repo_root / "models" / "quantile" / "option_b_quantile_head_best.pt"

    for p in [tcn_path, gat_path, edge_scaler_path, node_scaler_path, q_path]:
        assert p.exists(), f"CRITICAL: Required checkpoint/scaler not found at {p}"

    print("\n--- Loading Trained Models ---")
    # Regional TCN
    tcn_ckpt = torch.load(tcn_path, map_location="cpu", weights_only=False)
    tcn_model = RegionalTCN()
    tcn_model.load_state_dict(tcn_ckpt["model_state_dict"])
    tcn_model.eval()
    for param in tcn_model.parameters():
        param.requires_grad = False
    target_mean = float(tcn_ckpt["target_mean"])
    target_std = float(tcn_ckpt["target_std"])
    print(f"Loaded Regional TCN from {tcn_path} (Frozen). Scaling: mean={target_mean:.3f}, std={target_std:.3f}")

    # River GAT
    gat_ckpt = torch.load(gat_path, map_location="cpu", weights_only=False)
    gat_config = GATConfig(spatial_embedding_dim=64)
    gat_model = RiverGAT(gat_config)
    gat_model.load_state_dict(gat_ckpt)
    gat_model.eval()
    for param in gat_model.parameters():
        param.requires_grad = False
    node_scaler = joblib.load(node_scaler_path)
    edge_scaler = joblib.load(edge_scaler_path)
    print(f"Loaded River GAT from {gat_path} (Frozen).")

    # QuantileHead
    q_ckpt = torch.load(q_path, map_location="cpu", weights_only=False)
    q_model = QuantileHead(
        input_dim=192,
        hidden_dim=256,
        num_blocks=4,
        dropout=0.10,
        quantiles=(0.10, 0.50, 0.90),
        positive_output=False,
    )
    q_model.load_state_dict(q_ckpt["model_state_dict"])
    q_model.eval()
    for param in q_model.parameters():
        param.requires_grad = False
    print(f"Loaded Option B QuantileHead from {q_path} (Trained Epoch {q_ckpt.get('epoch', 'N/A')}).")

    # 3. Extract GAT Spatial Embeddings
    print("\n--- Extracting Spatial Embeddings from River Network Graph ---")
    test_gat_emb, test_gid_to_idx = extract_test_gat_embeddings(
        gat_model=gat_model,
        node_scaler=node_scaler,
        edge_scaler=edge_scaler,
        data_dir=data_dir,
    )
    print(f"Extracted GAT embeddings for test graph: {tuple(test_gat_emb.shape)}")

    # 4. Load Static Attributes and Preprocessor
    preprocessor = Preprocessor()
    static_df = load_static_attributes()
    flow_file = repo_root / "data" / "raw" / "CAMELS_IND_All_Catchments" / "streamflow_timeseries" / "streamflow_observed.csv"
    assert flow_file.exists(), f"streamflow_observed.csv missing at {flow_file}"
    flow_df = pd.read_csv(flow_file)

    # 5. Continuous Daily Inference Loop
    seq_len = 64
    batch_size = 256
    start_date_str = "1999-10-01"
    end_date_str = "2009-09-30"
    expected_dates = pd.date_range(start_date_str, end_date_str)
    num_expected_dates = len(expected_dates)
    print(f"\n--- Running Continuous Inference ({start_date_str} to {end_date_str}: {num_expected_dates} days) ---")

    all_records = []

    for i, gid_str in enumerate(test_ids, 1):
        col_key = str(int(gid_str))
        df_forcing = load_catchment_forcing(gid_str)
        dyn_mat = extract_catchment_matrix(df_forcing, preprocessor)
        stat_vec = static_df.loc[gid_str].to_numpy().astype(np.float32)

        # Dates and masks
        dt_series = pd.to_datetime(df_forcing[["year", "month", "day"]])
        test_mask = (dt_series >= start_date_str) & (dt_series <= end_date_str)
        test_indices = np.where(test_mask)[0]

        assert len(test_indices) == num_expected_dates, (
            f"Catchment {gid_str} has {len(test_indices)} dates in test range, expected {num_expected_dates}"
        )

        # Build contiguous sequence batches
        seq_batches = np.array([dyn_mat[idx - seq_len : idx] for idx in test_indices], dtype=np.float32)
        stat_batches = np.tile(stat_vec, (len(seq_batches), 1))

        # DataLoader for memory efficiency
        dataset = TensorDataset(torch.tensor(seq_batches), torch.tensor(stat_batches))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # GAT embedding for this catchment
        gat_idx = test_gid_to_idx[gid_str]
        gat_vec = test_gat_emb[gat_idx : gat_idx + 1]  # [1, 64]

        catchment_q_preds = []

        with torch.no_grad():
            for bx_d, bx_s in loader:
                tcn_emb = tcn_model(bx_d, bx_s, mode="embedding")  # [B, 128]
                gat_tiled = gat_vec.expand(len(tcn_emb), -1)       # [B, 64]
                coupled = torch.cat([tcn_emb, gat_tiled], dim=-1)   # [B, 192]
                norm_q = q_model(coupled)                           # [B, 3]
                catchment_q_preds.append(norm_q.numpy())

        catchment_q_preds = np.concatenate(catchment_q_preds, axis=0)  # [3653, 3]

        # Denormalize to physical units (m3/s) and apply physical clipping
        p10 = np.maximum(0.0, catchment_q_preds[:, 0] * target_std + target_mean)
        p50 = np.maximum(0.0, catchment_q_preds[:, 1] * target_std + target_mean)
        p90 = np.maximum(0.0, catchment_q_preds[:, 2] * target_std + target_mean)

        # Retrieve observed streamflow
        if col_key in flow_df.columns:
            obs_vals = flow_df[col_key].iloc[test_indices].to_numpy(dtype=np.float64)
        else:
            obs_vals = np.full(num_expected_dates, np.nan, dtype=np.float64)

        dates_str = dt_series.iloc[test_indices].dt.strftime("%Y-%m-%d").values

        for d_str, y_o, q10_val, q50_val, q90_val in zip(dates_str, obs_vals, p10, p50, p90):
            all_records.append({
                "date": d_str,
                "gauge_id": gid_str,
                "y_true": float(y_o),
                "q10": float(q10_val),
                "q50": float(q50_val),
                "q90": float(q90_val),
            })

        if i % 5 == 0 or i == len(test_ids):
            print(f"  Processed {i:02d}/{len(test_ids)} catchments ({gid_str}: {num_expected_dates} daily steps).")

    # 6. Assemble DataFrame and Sort
    print("\n--- Assembling and Sorting Continuous Predictions ---")
    pred_df = pd.DataFrame(all_records)
    # Order columns strictly as requested
    pred_df = pred_df[["date", "gauge_id", "y_true", "q10", "q50", "q90"]]
    pred_df.sort_values(by=["gauge_id", "date"], ascending=[True, True], inplace=True)
    pred_df.reset_index(drop=True, inplace=True)

    # 7. Validate Predictions
    print("\n--- Validating Continuous Predictions ---")
    n_unique_catchments = pred_df["gauge_id"].nunique()
    n_unique_dates = pred_df["date"].nunique()
    total_rows = len(pred_df)
    earliest_date = str(pred_df["date"].min())
    latest_date = str(pred_df["date"].max())
    rows_per_catchment = pred_df.groupby("gauge_id").size().to_dict()

    assert n_unique_catchments == 37, f"Expected 37 catchments, got {n_unique_catchments}"
    assert n_unique_dates == num_expected_dates, f"Expected {num_expected_dates} dates, got {n_unique_dates}"
    assert total_rows == 37 * num_expected_dates, f"Expected {37 * num_expected_dates} rows, got {total_rows}"

    # Numerical Checks
    nan_q10 = int(pred_df["q10"].isna().sum())
    nan_q50 = int(pred_df["q50"].isna().sum())
    nan_q90 = int(pred_df["q90"].isna().sum())
    inf_q10 = int(np.isinf(pred_df["q10"]).sum())
    inf_q50 = int(np.isinf(pred_df["q50"]).sum())
    inf_q90 = int(np.isinf(pred_df["q90"]).sum())

    assert nan_q10 == 0 and nan_q50 == 0 and nan_q90 == 0, "NaN values detected in quantiles!"
    assert inf_q10 == 0 and inf_q50 == 0 and inf_q90 == 0, "Inf values detected in quantiles!"

    # Non-negative check
    neg_q10 = int((pred_df["q10"] < 0.0).sum())
    neg_q50 = int((pred_df["q50"] < 0.0).sum())
    neg_q90 = int((pred_df["q90"] < 0.0).sum())
    assert neg_q10 == 0 and neg_q50 == 0 and neg_q90 == 0, "Negative values detected in quantiles!"

    # Quantile Monotonicity (q10 <= q50 <= q90)
    # Using tiny epsilon (1e-5) for floating point tolerance
    eps = 1e-5
    valid_monotonic = (pred_df["q10"] <= pred_df["q50"] + eps) & (pred_df["q50"] <= pred_df["q90"] + eps)
    crossing_violations_count = int((~valid_monotonic).sum())
    crossing_rate = float(crossing_violations_count / total_rows * 100.0)

    # Min/Max ranges
    valid_y = pred_df["y_true"].dropna()
    min_y_true = float(valid_y.min())
    max_y_true = float(valid_y.max())
    nan_y_true = int(pred_df["y_true"].isna().sum())

    min_q10 = float(pred_df["q10"].min())
    max_q10 = float(pred_df["q10"].max())
    min_q50 = float(pred_df["q50"].min())
    max_q50 = float(pred_df["q50"].max())
    min_q90 = float(pred_df["q90"].min())
    max_q90 = float(pred_df["q90"].max())

    print(f"Catchments: {n_unique_catchments} (All 37 present)")
    print(f"Date range: {earliest_date} to {latest_date} ({n_unique_dates} unique daily timestamps)")
    print(f"Total rows: {total_rows:,} ({num_expected_dates} per catchment)")
    print(f"Quantile Crossing Violations: {crossing_violations_count} ({crossing_rate:.4f}%)")
    print(f"NaNs: q10={nan_q10}, q50={nan_q50}, q90={nan_q90}, y_true={nan_y_true}")
    print(f"q10 range: [{min_q10:.3f}, {max_q10:.3f}] m3/s")
    print(f"q50 range: [{min_q50:.3f}, {max_q50:.3f}] m3/s")
    print(f"q90 range: [{min_q90:.3f}, {max_q90:.3f}] m3/s")
    print(f"y_true range: [{min_y_true:.3f}, {max_y_true:.3f}] m3/s (across non-null observations)")

    # 8. Save Compressed Predictions
    out_file = results_dir / "quantile_test_predictions.csv.gz"
    print(f"\nSaving compressed predictions to {out_file}...")
    pred_df.to_csv(out_file, index=False, compression="gzip")
    file_size_kb = out_file.stat().st_size / 1024.0
    print(f"Saved successfully ({file_size_kb:.1f} KB).")

    # 9. Save Verification Report
    report_data = {
        "status": "SUCCESS",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "execution_time_seconds": round(time.time() - start_time, 2),
        "checkpoints_used": {
            "regional_tcn": str(tcn_path),
            "river_gat": str(gat_path),
            "edge_scaler": str(edge_scaler_path),
            "node_scaler": str(node_scaler_path),
            "quantile_head": str(q_path),
        },
        "target_scaling": {"mean": target_mean, "std": target_std},
        "coverage": {
            "num_test_catchments": n_unique_catchments,
            "mahanadi_count": n_mahanadi,
            "narmada_count": n_narmada,
            "earliest_date": earliest_date,
            "latest_date": latest_date,
            "num_unique_dates": n_unique_dates,
            "total_prediction_rows": total_rows,
            "rows_per_catchment": num_expected_dates,
            "sequence_length": seq_len,
        },
        "validation": {
            "nan_counts": {
                "q10": nan_q10,
                "q50": nan_q50,
                "q90": nan_q90,
                "y_true_missing": nan_y_true,
            },
            "inf_counts": {
                "q10": inf_q10,
                "q50": inf_q50,
                "q90": inf_q90,
            },
            "negative_counts": {
                "q10": neg_q10,
                "q50": neg_q50,
                "q90": neg_q90,
            },
            "crossing_violations_count": crossing_violations_count,
            "crossing_rate_pct": crossing_rate,
            "ranges_m3s": {
                "y_true_min": min_y_true,
                "y_true_max": max_y_true,
                "q10_min": min_q10,
                "q10_max": max_q10,
                "q50_min": min_q50,
                "q50_max": max_q50,
                "q90_min": min_q90,
                "q90_max": max_q90,
            },
        },
        "output_file": str(out_file),
    }

    def json_serializer(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    report_path = results_dir / "continuous_inference_report.json"
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2, default=json_serializer)
    print(f"Saved verification report to {report_path}.")

    return report_data


if __name__ == "__main__":
    run_continuous_inference()
