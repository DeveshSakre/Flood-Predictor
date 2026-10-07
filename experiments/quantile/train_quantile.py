"""
Training, Validation, and Calibration for REAL Option B Quantile Regression.

Coupled Spatiotemporal Architecture:
REAL CAMELS-IND Dynamic Meteorology + Static Catchment Descriptors
        ↓
REAL Regional TCN (FROZEN, models/regional_tcn/regional_tcn_best.pt)
        ↓
Temporal embedding [B, 128]

REAL HydroRIVERS Topology
        ↓
REAL River GAT (FROZEN, models/gat/river_gat_encoder.pt)
        ↓
Spatial embedding [B, 64]

Catchment-aligned concatenation
        ↓
[B, 192] coupled representation

QuantileHead (TRAINABLE, input_dim=192)
        ↓
P10 / P50 / P90 (strictly monotonic: P10 <= P50 <= P90)

RESPONSIBILITY: Member 3 (Quantile Regression & Uncertainty Estimation)
BRANCH: Aditya_Quantile_Regression
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import get_repo_root
from src.utils.reproducibility import set_seed
from src.data.loaders import (
    load_split_catchment_ids,
    load_catchment_forcing,
    load_static_attributes,
)
from src.data.preprocessing import Preprocessor
from src.models.regional_tcn import RegionalTCN
from src.models.gat_config import GATConfig
from src.models.river_gat import RiverGAT
from src.uncertainty.quantile_regression import QuantileHead
from src.uncertainty.quantile_loss import MultiQuantilePinballLoss, pinball_loss
from src.uncertainty.calibration import (
    prediction_interval_coverage_probability,
    mean_prediction_interval_width,
    check_crossing_violations,
    winkler_score,
    evaluate_uncertainty_forecasts,
)
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.peak_metrics import peak_flow_error, peak_timing_error, high_flow_nse

FEATURE_COLS = [
    "prcp(mm/day)", "tmax(C)", "tmin(C)", "srad_lw(w/m2)", "srad_sw(w/m2)",
    "wind_u(m/s)", "wind_v(m/s)", "rel_hum(%)", "pet_gleam(mm/day)", "aet_gleam(mm/day)",
    "evap_canopy(mm/day)", "evap_surface(mm/day)", "sm_lvl1(kg/m2)", "sm_lvl2(kg/m2)",
    "sm_lvl3(kg/m2)", "sm_lvl4(kg/m2)", "sin_month", "cos_month", "sin_doy", "cos_doy"
]


def extract_catchment_matrix(df_raw: pd.DataFrame, preprocessor: Preprocessor) -> np.ndarray:
    """Preprocess dynamic physical forcings and compute cyclical calendar encodings."""
    df_norm = preprocessor.transform_dynamic(df_raw)
    dt = pd.to_datetime(df_raw[["year", "month", "day"]])
    df_norm["sin_month"] = np.sin(2 * np.pi * dt.dt.month / 12.0)
    df_norm["cos_month"] = np.cos(2 * np.pi * dt.dt.month / 12.0)
    df_norm["sin_doy"] = np.sin(2 * np.pi * dt.dt.dayofyear / 365.25)
    df_norm["cos_doy"] = np.cos(2 * np.pi * dt.dt.dayofyear / 365.25)
    return df_norm[FEATURE_COLS].to_numpy().astype(np.float32)


def run_anti_leakage_audit(
    train_ids: List[str],
    val_ids: List[str],
    test_ids: List[str],
    demo_ids: List[str],
    static_df: pd.DataFrame,
) -> None:
    """Rigorous pre-flight anti-leakage audit prior to execution."""
    print("\n--- Running Pre-Flight Anti-Leakage Audit ---")
    set_train = set(train_ids)
    set_val = set(val_ids)
    set_test = set(test_ids)
    set_demo = set(demo_ids)

    # 1. Partition disjointness
    assert len(train_ids) == 169, f"Expected 169 train catchments, got {len(train_ids)}"
    assert len(val_ids) == 36, f"Expected 36 val catchments, got {len(val_ids)}"
    assert len(test_ids) == 37, f"Expected 37 test catchments, got {len(test_ids)}"
    assert set_train.isdisjoint(set_val), "CRITICAL: Train and validation partitions overlap!"
    assert set_train.isdisjoint(set_test), "CRITICAL: Train and test partitions overlap!"
    assert set_val.isdisjoint(set_test), "CRITICAL: Validation and test partitions overlap!"
    assert set_train.isdisjoint(set_demo), "CRITICAL: Train and ungauged demo overlap!"
    print(f"Partition disjointness: VERIFIED (169 train, 36 val, 37 test, {len(demo_ids)} demo).")

    # 2. Static feature leakage checks
    forbidden = ["reservoir_index", "flow_availability", "DIS_AV_CMS", "ORD_FLOW", "lstm_pred_streamflow"]
    for f in forbidden:
        assert f not in static_df.columns, f"CRITICAL: Forbidden feature '{f}' present in static features!"

    streamflow_signatures = ["q_mean", "runoff_ratio", "slope_fdc", "bfi", "q_10", "q_50", "q_90"]
    for sig in streamflow_signatures:
        assert sig not in static_df.columns, f"CRITICAL: Streamflow signature '{sig}' found in static features!"

    assert static_df.shape[1] == 129, f"Expected 129 static features, got {static_df.shape[1]}"
    print("Static features: VERIFIED (129 features, zero streamflow leakage).")
    print("Anti-leakage audit: PASSED WITHOUT ERRORS.\n")


def extract_split_gat_embeddings(
    split: str,
    gat_model: RiverGAT,
    node_scaler: Any,
    edge_scaler: Any,
    data_dir: Path,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """
    Extract genuine spatial embeddings [N, 64] from river network graph topology.
    Returns: (gat_embeddings_tensor, dict mapping 5-digit gauge_id -> node index).
    """
    nodes_csv = data_dir / "graph" / f"nodes_{split}.csv"
    edges_csv = data_dir / "graph" / f"edges_{split}.csv"

    nodes_df = pd.read_csv(nodes_csv)
    edges_df = pd.read_csv(edges_csv)

    # Node features [N, 6]
    node_cols = ["UPLAND_SKM", "CATCH_SKM", "DIST_DN_KM", "DIST_UP_KM", "ORD_STRA", "ORD_CLAS"]
    node_raw = nodes_df[node_cols].to_numpy().astype(np.float32)
    node_scaled = node_scaler.transform(node_raw)
    x_nodes = torch.tensor(node_scaled, dtype=torch.float32)

    # Edge index [2, E] and edge features [E, 4]
    src = torch.tensor(edges_df["source_idx"].to_numpy(), dtype=torch.long)
    dst = torch.tensor(edges_df["target_idx"].to_numpy(), dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)

    edge_cols = ["routing_distance_km", "steps", "drainage_area_ratio", "stream_order_delta"]
    edge_raw = edges_df[edge_cols].to_numpy().astype(np.float32)
    edge_scaled = edge_scaler.transform(edge_raw)
    edge_attr = torch.tensor(edge_scaled, dtype=torch.float32)

    with torch.no_grad():
        gat_emb = gat_model(x_nodes, edge_index, edge_attr)

    assert gat_emb.shape == (len(nodes_df), 64), f"Expected ({len(nodes_df)}, 64), got {gat_emb.shape}"
    assert torch.isfinite(gat_emb).all(), "GAT embeddings contain non-finite values!"

    nodes_df["gauge_id_str"] = nodes_df["gauge_id"].astype(str).str.zfill(5)
    gid_to_idx = {gid: idx for idx, gid in enumerate(nodes_df["gauge_id_str"])}

    return gat_emb, gid_to_idx


def extract_real_spatiotemporal_dataset(
    catchment_ids: List[str],
    samples_per_basin: int,
    tcn_model: RegionalTCN,
    gat_emb: torch.Tensor,
    gid_to_gat_idx: Dict[str, int],
    static_df: pd.DataFrame,
    flow_df: pd.DataFrame,
    preprocessor: Preprocessor,
    target_mean: float,
    target_std: float,
    seq_len: int = 64,
    seed: int = 42,
    split_name: str = "train",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """
    Extract REAL coupled representations [B, 192] and real observed streamflow targets:
    - Real TCN temporal embedding: [B, 128]
    - Real GAT spatial embedding: [B, 64]
    - Catchment-aligned concatenation: [B, 192]
    """
    rng = np.random.RandomState(seed)
    X_coupled_list = []
    y_norm_list = []
    y_phys_list = []
    gids_list = []

    print(f"Extracting real {split_name} representations for {len(catchment_ids)} catchments...")

    for gid in catchment_ids:
        gid_str = str(gid).zfill(5)
        col_key = str(int(gid))

        if col_key not in flow_df.columns:
            continue

        df_forcing = load_catchment_forcing(gid)
        dyn_mat = extract_catchment_matrix(df_forcing, preprocessor)
        stat_vec = static_df.loc[gid_str].to_numpy().astype(np.float32)
        obs_series = flow_df[col_key].to_numpy()

        valid_idx = np.where((~np.isnan(obs_series)) & (np.arange(len(obs_series)) >= seq_len))[0]
        if len(valid_idx) == 0:
            continue

        if len(valid_idx) > samples_per_basin:
            chosen = rng.choice(valid_idx, size=samples_per_basin, replace=False)
        else:
            chosen = valid_idx

        # Construct batch of sequences for this catchment
        seq_batches = np.array([dyn_mat[idx - seq_len : idx] for idx in chosen], dtype=np.float32)
        stat_batches = np.tile(stat_vec, (len(seq_batches), 1))

        t_dyn = torch.tensor(seq_batches, dtype=torch.float32)
        t_stat = torch.tensor(stat_batches, dtype=torch.float32)

        # Forward through frozen real Regional TCN
        with torch.no_grad():
            tcn_emb = tcn_model(t_dyn, t_stat, mode="embedding")  # [N_samples, 128]

        # Retrieve aligned real GAT embedding for this catchment
        gat_idx = gid_to_gat_idx[gid_str]
        gat_vector = gat_emb[gat_idx : gat_idx + 1]  # [1, 64]
        gat_rep_tiled = gat_vector.expand(len(tcn_emb), -1)  # [N_samples, 64]

        # Genuine concatenation [N_samples, 192]
        coupled = torch.cat([tcn_emb, gat_rep_tiled], dim=-1)

        obs_chosen = obs_series[chosen]
        norm_chosen = (obs_chosen - target_mean) / target_std

        X_coupled_list.append(coupled)
        y_norm_list.append(torch.tensor(norm_chosen, dtype=torch.float32).unsqueeze(-1))
        y_phys_list.append(torch.tensor(obs_chosen, dtype=torch.float32).unsqueeze(-1))
        gids_list.extend([gid_str] * len(chosen))

    X_coupled = torch.cat(X_coupled_list, dim=0)
    y_norm = torch.cat(y_norm_list, dim=0)
    y_phys = torch.cat(y_phys_list, dim=0)

    print(f"  {split_name.capitalize()} tensor ready: {X_coupled.shape[0]:,} sequences | Shape: {tuple(X_coupled.shape)}")
    return X_coupled, y_norm, y_phys, gids_list


def run_real_option_b_experiment(
    batch_size: int = 64,
    learning_rate: float = 0.001,
    max_epochs: int = 25,
    patience: int = 6,
    samples_per_train_basin: int = 250,
    samples_per_val_basin: int = 100,
    samples_per_test_basin: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute complete REAL Option B training and calibration evaluation."""
    start_time = time.time()
    set_seed(seed)

    print("=" * 80)
    print("STARTING REAL OPTION B EXPERIMENT (Regional TCN + River GAT + QuantileHead)")
    print(f"Branch: Aditya_Quantile_Regression | Seed: {seed} | Batch Size: {batch_size} | LR: {learning_rate}")
    print("=" * 80)

    repo_root = get_repo_root()
    res_dir = repo_root / "results" / "quantile"
    res_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = res_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    model_dir = repo_root / "models" / "quantile"
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Experimental Splits
    splits = load_split_catchment_ids()
    train_ids = splits["train"]
    val_ids = splits["val"]
    test_ids = splits["test"]
    demo_ids = splits.get("ungauged_demo", [])

    # 2. Load Preprocessors and Static Attributes
    preprocessor = Preprocessor()
    static_df = load_static_attributes()
    run_anti_leakage_audit(train_ids, val_ids, test_ids, demo_ids, static_df)

    # 3. Load Observed Streamflow
    flow_file = repo_root / "data" / "raw" / "CAMELS_IND_All_Catchments" / "streamflow_timeseries" / "streamflow_observed.csv"
    assert flow_file.exists(), f"streamflow_observed.csv not found at {flow_file}"
    flow_df = pd.read_csv(flow_file)
    print(f"Observed streamflow loaded: {flow_df.shape[0]} daily timesteps.")

    # 4. Load FROZEN Upstream Checkpoints
    print("\n--- Loading Trained Upstream Models ---")
    tcn_path = repo_root / "models" / "regional_tcn" / "regional_tcn_best.pt"
    assert tcn_path.exists(), f"TCN checkpoint not found at {tcn_path}"
    tcn_ckpt = torch.load(tcn_path, map_location="cpu", weights_only=False)
    tcn_model = RegionalTCN()
    tcn_model.load_state_dict(tcn_ckpt["model_state_dict"])
    tcn_model.eval()
    for p in tcn_model.parameters():
        p.requires_grad = False
    target_mean = float(tcn_ckpt["target_mean"])
    target_std = float(tcn_ckpt["target_std"])
    print(f"Regional TCN loaded from {tcn_path} (Frozen). Target scaling: mean={target_mean:.3f}, std={target_std:.3f}")

    gat_path = repo_root / "models" / "gat" / "river_gat_encoder.pt"
    assert gat_path.exists(), f"GAT checkpoint not found at {gat_path}"
    gat_ckpt = torch.load(gat_path, map_location="cpu", weights_only=False)
    gat_config = GATConfig(spatial_embedding_dim=64)
    gat_model = RiverGAT(gat_config)
    gat_model.load_state_dict(gat_ckpt)
    gat_model.eval()
    for p in gat_model.parameters():
        p.requires_grad = False
    print(f"River GAT loaded from {gat_path} (Frozen).")

    # 5. Extract GAT Embeddings from Real River Network Graphs
    print("\n--- Extracting Real GAT Spatial Embeddings from Graphs ---")
    node_scaler = joblib.load(repo_root / "data" / "processed" / "scalers" / "graph_node_scaler.joblib")
    edge_scaler = joblib.load(repo_root / "models" / "gat" / "edge_scaler.joblib")
    data_processed = repo_root / "data" / "processed"

    train_gat_emb, train_gid_to_idx = extract_split_gat_embeddings("train", gat_model, node_scaler, edge_scaler, data_processed)
    val_gat_emb, val_gid_to_idx = extract_split_gat_embeddings("val", gat_model, node_scaler, edge_scaler, data_processed)
    test_gat_emb, test_gid_to_idx = extract_split_gat_embeddings("test", gat_model, node_scaler, edge_scaler, data_processed)

    print(f"GAT Embeddings extracted: Train={train_gat_emb.shape}, Val={val_gat_emb.shape}, Test={test_gat_emb.shape}")

    # 6. Extract Real Spatiotemporal Representations
    print("\n--- Extracting Coupled Real [B, 192] Representations ---")
    X_train, y_train_norm, y_train_phys, train_gids = extract_real_spatiotemporal_dataset(
        train_ids, samples_per_train_basin, tcn_model, train_gat_emb, train_gid_to_idx,
        static_df, flow_df, preprocessor, target_mean, target_std, seq_len=64, seed=seed, split_name="train"
    )

    X_val, y_val_norm, y_val_phys, val_gids = extract_real_spatiotemporal_dataset(
        val_ids, samples_per_val_basin, tcn_model, val_gat_emb, val_gid_to_idx,
        static_df, flow_df, preprocessor, target_mean, target_std, seq_len=64, seed=seed + 100, split_name="validation"
    )

    # 7. Initialize QuantileHead (TRAINABLE ONLY)
    print("\n--- Initializing Trainable QuantileHead Architecture ---")
    model = QuantileHead(
        input_dim=192,
        hidden_dim=256,
        num_blocks=4,
        dropout=0.10,
        quantiles=(0.10, 0.50, 0.90),
        positive_output=False,  # Operates on normalized target space; strictly monotonic
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"QuantileHead parameters: {total_params:,} (Trainable)")

    criterion = MultiQuantilePinballLoss(quantiles=(0.10, 0.50, 0.90))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)

    train_loader = DataLoader(TensorDataset(X_train, y_train_norm), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val_norm), batch_size=batch_size, shuffle=False)

    best_checkpoint_path = model_dir / "option_b_quantile_head_best.pt"
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history: List[Dict[str, float]] = []

    print("\n--- Training QuantileHead on Real Representations ---")
    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()
        model.train()
        running_train_loss = 0.0

        for bx, by in train_loader:
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_train_loss += loss.item() * len(bx)

        train_loss = running_train_loss / len(X_train)
        scheduler.step()

        # Validation evaluation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                preds = model(bx)
                val_loss = criterion(preds, by)
                running_val_loss += val_loss.item() * len(bx)

        val_loss = running_val_loss / len(X_val)
        epoch_dur = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        is_best = val_loss < best_val_loss
        marker = "(* best *)" if is_best else ""
        print(f"Epoch {epoch:02d}/{max_epochs:02d} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | LR: {current_lr:.6f} | {epoch_dur:.1f}s {marker}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": current_lr,
        })

        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "target_mean": target_mean,
                "target_std": target_std,
                "input_dim": 192,
                "hidden_dim": 256,
                "num_blocks": 4,
                "quantiles": [0.10, 0.50, 0.90],
                "num_params": total_params,
            }, best_checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch} (patience = {patience}).")
                break

    # Save loss history
    history_df = pd.DataFrame(history)
    history_df.to_csv(res_dir / "option_b_loss_history.csv", index=False)

    # 8. Load Best Checkpoint for Partition Evaluation
    print("\n--- Evaluating Best Model on Validation and Held-Out Test Partitions ---")
    ckpt = torch.load(best_checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Extract test dataset strictly now
    X_test, y_test_norm, y_test_phys, test_gids = extract_real_spatiotemporal_dataset(
        test_ids, samples_per_test_basin, tcn_model, test_gat_emb, test_gid_to_idx,
        static_df, flow_df, preprocessor, target_mean, target_std, seq_len=64, seed=seed + 200, split_name="test"
    )

    def evaluate_partition_predictions(
        X_data: torch.Tensor,
        y_phys_data: torch.Tensor,
        partition_gids: List[str],
        set_name: str,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        with torch.no_grad():
            norm_preds = model(X_data).numpy()

        # Denormalize to physical streamflow (m^3/s) with non-negativity constraint
        p10 = np.maximum(0.0, norm_preds[:, 0] * target_std + target_mean)
        p50 = np.maximum(0.0, norm_preds[:, 1] * target_std + target_mean)
        p90 = np.maximum(0.0, norm_preds[:, 2] * target_std + target_mean)
        y_obs = y_phys_data.numpy().ravel()

        df_preds = pd.DataFrame({
            "gauge_id": partition_gids,
            "observed_flow": y_obs,
            "p10": p10,
            "predicted_flow": p50,
            "p90": p90,
        })

        # Basin-level evaluation
        basin_records = []
        for gid, grp in df_preds.groupby("gauge_id"):
            b_obs = grp["observed_flow"].to_numpy()
            b_p10 = grp["p10"].to_numpy()
            b_p50 = grp["predicted_flow"].to_numpy()
            b_p90 = grp["p90"].to_numpy()

            u_metrics = evaluate_uncertainty_forecasts(b_obs, b_p10, b_p50, b_p90)
            h_metrics = compute_all_metrics(b_obs, b_p50)
            peak = peak_flow_error(b_obs, b_p50)
            timing = peak_timing_error(b_obs, b_p50)
            hf_nse = high_flow_nse(b_obs, b_p50)

            rec = {
                "gauge_id": gid,
                **u_metrics,
                **h_metrics,
                "peak_obs_m3s": peak["obs_peak"],
                "peak_sim_m3s": peak["sim_peak"],
                "peak_diff_m3s": peak["diff_m3s"],
                "peak_rel_error_pct": peak["rel_error_pct"],
                "peak_timing_days": timing,
                "high_flow_nse": hf_nse,
            }
            basin_records.append(rec)

        basin_df = pd.DataFrame(basin_records)

        # Global partition metrics
        global_u = evaluate_uncertainty_forecasts(y_obs, p10, p50, p90)
        global_h = compute_all_metrics(y_obs, p50)

        # High-Flow Analysis (above 90th percentile of observed flow)
        q90_thresh = float(np.percentile(y_obs, 90))
        high_mask = y_obs >= q90_thresh
        high_u = evaluate_uncertainty_forecasts(y_obs[high_mask], p10[high_mask], p50[high_mask], p90[high_mask])
        high_h = compute_all_metrics(y_obs[high_mask], p50[high_mask])

        norm_mask = ~high_mask
        norm_u = evaluate_uncertainty_forecasts(y_obs[norm_mask], p10[norm_mask], p50[norm_mask], p90[norm_mask])

        summary = {
            "overall_uncertainty": global_u,
            "overall_hydrologic": global_h,
            "high_flow_threshold_m3s": q90_thresh,
            "high_flow_uncertainty": high_u,
            "high_flow_hydrologic": high_h,
            "normal_flow_uncertainty": norm_u,
            "basin_median_NSE": float(basin_df["NSE"].median()),
            "basin_median_KGE": float(basin_df["KGE"].median()),
            "basin_median_PICP": float(basin_df["picp_80"].median()),
            "basin_median_MPIW": float(basin_df["mpiw"].median()),
        }

        return basin_df, summary

    val_basin_df, val_summary = evaluate_partition_predictions(X_val, y_val_phys, val_gids, "Validation")
    test_basin_df, test_summary = evaluate_partition_predictions(X_test, y_test_phys, test_gids, "Held-Out Test")

    # Save metrics tables
    val_basin_df.to_csv(res_dir / "option_b_validation_metrics.csv", index=False)
    test_basin_df.to_csv(res_dir / "option_b_test_metrics.csv", index=False)

    full_summary = {
        "model": "QuantileHead_Option_B",
        "upstream_models": {
            "regional_tcn": str(tcn_path),
            "river_gat": str(gat_path),
            "tcn_dim": 128,
            "gat_dim": 64,
            "coupled_dim": 192,
        },
        "target_scaling": {"mean": target_mean, "std": target_std},
        "training": {
            "samples": len(X_train),
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
        },
        "validation_36_catchments": val_summary,
        "held_out_test_37_catchments": test_summary,
    }
    with open(res_dir / "option_b_summary_metrics.json", "w") as f:
        json.dump(full_summary, f, indent=2)

    # 9. Diagnostic Visualizations
    print("\n--- Generating Diagnostic Calibration Plots ---")
    # Loss curve
    plt.figure(figsize=(8, 4.5))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Train Pinball Loss", marker="o", color="#1f77b4")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="Val Pinball Loss", marker="s", color="#ff7f0e")
    plt.axvline(best_epoch, color="#2ca02c", linestyle="--", label=f"Best Checkpoint (Epoch {best_epoch})")
    plt.title("Option B QuantileHead Training & Validation Convergence", fontsize=11, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Multi-Quantile Pinball Loss (Normalized)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "option_b_loss_curve.png", dpi=200)
    plt.close()

    # Reliability Diagram
    nominal_levels = [0.10, 0.50, 0.90]
    val_cov = [
        val_summary["overall_uncertainty"]["coverage_p10"],
        val_summary["overall_uncertainty"]["coverage_p50"],
        val_summary["overall_uncertainty"]["coverage_p90"],
    ]
    test_cov = [
        test_summary["overall_uncertainty"]["coverage_p10"],
        test_summary["overall_uncertainty"]["coverage_p50"],
        test_summary["overall_uncertainty"]["coverage_p90"],
    ]
    plt.figure(figsize=(6.5, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (Ideal)", alpha=0.7)
    plt.plot(nominal_levels, val_cov, "s-", color="#ff7f0e", label=f"Validation (PICP={val_summary['overall_uncertainty']['picp_80']:.1%})", linewidth=2, markersize=8)
    plt.plot(nominal_levels, test_cov, "o-", color="#2ca02c", label=f"Held-Out Test (PICP={test_summary['overall_uncertainty']['picp_80']:.1%})", linewidth=2, markersize=8)
    plt.title("Option B Reliability Diagram: Empirical vs. Nominal Quantile Coverage", fontsize=11, fontweight="bold")
    plt.xlabel("Nominal Quantile Level")
    plt.ylabel("Empirical Coverage Fraction")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "option_b_calibration_curve.png", dpi=200)
    plt.close()

    # Prediction Interval Coverage & Width Distribution
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(val_basin_df["picp_80"], bins=10, alpha=0.6, color="#ff7f0e", label="Val PICP", edgecolor="black")
    axes[0].hist(test_basin_df["picp_80"], bins=10, alpha=0.6, color="#2ca02c", label="Test PICP", edgecolor="black")
    axes[0].axvline(0.80, color="red", linestyle="--", label="Target Nominal (80%)")
    axes[0].set_title("80% Prediction Interval Coverage (PICP)", fontweight="bold")
    axes[0].set_xlabel("PICP")
    axes[0].set_ylabel("Catchment Count")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].hist(val_basin_df["mpiw"], bins=10, alpha=0.6, color="#ff7f0e", label="Val MPIW", edgecolor="black")
    axes[1].hist(test_basin_df["mpiw"], bins=10, alpha=0.6, color="#2ca02c", label="Test MPIW", edgecolor="black")
    axes[1].set_title("Mean Prediction Interval Width (MPIW)", fontweight="bold")
    axes[1].set_xlabel("Width (m3/s)")
    axes[1].set_ylabel("Catchment Count")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(plots_dir / "option_b_interval_coverage.png", dpi=200)
    plt.close()

    # Sample Hydrograph
    sample_gid = test_ids[0]
    sample_mask = np.array(test_gids) == sample_gid
    with torch.no_grad():
        s_preds = model(X_test[sample_mask]).numpy()
    s_p10 = np.maximum(0.0, s_preds[:, 0] * target_std + target_mean)
    s_p50 = np.maximum(0.0, s_preds[:, 1] * target_std + target_mean)
    s_p90 = np.maximum(0.0, s_preds[:, 2] * target_std + target_mean)
    s_obs = y_test_phys[sample_mask].numpy().ravel()
    t_steps = np.arange(len(s_obs))

    plt.figure(figsize=(10, 4.5))
    plt.plot(t_steps, s_obs, "k-", label="Observed Streamflow", linewidth=1.5)
    plt.plot(t_steps, s_p50, "b-", label="Predicted Median (P50)", linewidth=1.5)
    plt.fill_between(t_steps, s_p10, s_p90, color="blue", alpha=0.2, label="80% Prediction Interval [P10, P90]")
    plt.title(f"REAL Option B Discharge Forecast: Held-Out Catchment {sample_gid}", fontsize=11, fontweight="bold")
    plt.xlabel("Sample Index")
    plt.ylabel("Streamflow (m3/s)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plots_dir / "option_b_uncertainty_hydrograph.png", dpi=200)
    plt.close()

    total_time = time.time() - start_time
    print(f"\nREAL Option B Experiment successfully completed in {total_time:.1f}s.")
    print(f"Best checkpoint saved to: {best_checkpoint_path}")
    print(f"Validation PICP: {val_summary['overall_uncertainty']['picp_80']:.3f} | Test PICP: {test_summary['overall_uncertainty']['picp_80']:.3f}")
    print(f"Validation Crossing Violations: {val_summary['overall_uncertainty']['crossing_violations_pct']:.2f}% | Test Crossing: {test_summary['overall_uncertainty']['crossing_violations_pct']:.2f}%")

    return full_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run REAL Option B Quantile Regression Experiment")
    parser.add_argument("--epochs", type=int, default=20, help="Maximum epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_real_option_b_experiment(
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_epochs=args.epochs,
        seed=args.seed,
    )
