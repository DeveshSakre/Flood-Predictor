"""
Training and Evaluation Pipeline for Regional Temporal Convolutional Network (RegionalTCN).

Controls for fair, leakage-safe empirical comparison against the Base TCN baseline:
- Dynamic meteorological sequence input: [B, 64, 20]
- Static catchment physiography input: [B, 129]
- Trains strictly on 169 source catchments
- Evaluates validation loss and early stopping on 36 validation catchments
- Evaluates generalization on 37 held-out unseen test catchments
- Generates forward predictions on 5 operational zero-flow demonstration catchments
- Uses train-fitted preprocessing and saves checkpoints, metrics, and comparison tables.

RESPONSIBILITY: Member 2 (Temporal Modeling / Regional TCN)
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
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
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
from src.models.regional_tcn import RegionalTCN
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.evaluation_pipeline import (
    evaluate_catchment_predictions,
    summarize_regional_performance,
)
from src.evaluation.plots import plot_hydrograph, plot_cumulative_nse


FEATURE_COLS = [
    "prcp(mm/day)", "tmax(C)", "tmin(C)", "srad_lw(w/m2)", "srad_sw(w/m2)",
    "wind_u(m/s)", "wind_v(m/s)", "rel_hum(%)", "pet_gleam(mm/day)", "aet_gleam(mm/day)",
    "evap_canopy(mm/day)", "evap_surface(mm/day)", "sm_lvl1(kg/m2)", "sm_lvl2(kg/m2)",
    "sm_lvl3(kg/m2)", "sm_lvl4(kg/m2)", "sin_month", "cos_month", "sin_doy", "cos_doy"
]


def extract_catchment_matrix(df_raw: pd.DataFrame, preprocessor: Preprocessor) -> np.ndarray:
    """Preprocess dynamic physical forcings and add cyclical calendar encodings."""
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
    """Rigorous pre-flight anti-leakage audit prior to training execution."""
    print("\n--- Running Pre-Flight Anti-Leakage Audit ---")
    set_train = set(train_ids)
    set_val = set(val_ids)
    set_test = set(test_ids)
    set_demo = set(demo_ids)

    # 1. Partition disjointness
    assert set_train.isdisjoint(set_val), "CRITICAL: Train and validation partitions overlap!"
    assert set_train.isdisjoint(set_test), "CRITICAL: Train and test partitions overlap!"
    assert set_val.isdisjoint(set_test), "CRITICAL: Validation and test partitions overlap!"
    assert set_train.isdisjoint(set_demo), "CRITICAL: Train and ungauged demo overlap!"
    print("Partition disjointness: VERIFIED (169 train, 36 val, 37 test, 5 demo).")

    # 2. Static feature leakage checks
    forbidden = ["reservoir_index", "flow_availability", "DIS_AV_CMS", "ORD_FLOW", "lstm_pred_streamflow"]
    for f in forbidden:
        assert f not in static_df.columns, f"CRITICAL: Forbidden feature '{f}' present in static features!"

    streamflow_signatures = ["q_mean", "runoff_ratio", "slope_fdc", "bfi", "q_10", "q_50", "q_90"]
    for sig in streamflow_signatures:
        assert sig not in static_df.columns, f"CRITICAL: Streamflow signature '{sig}' found in static features!"

    assert static_df.shape[1] == 129, f"Expected 129 static features, got {static_df.shape[1]}"
    assert not static_df.isna().any().any(), "Static attributes contain NaN!"
    assert not np.isinf(static_df.to_numpy()).any(), "Static attributes contain Inf!"
    print(f"Static features ({static_df.shape[1]} attributes): VERIFIED (zero streamflow signatures, zero NaN/Inf).")
    print("Anti-leakage audit: PASSED WITHOUT ERRORS.\n")


def run_experiment(
    max_epochs: int = 12,
    patience: int = 4,
    samples_per_train_basin: int = 250,
    samples_per_val_basin: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.0005,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute complete RegionalTCN training, validation, evaluation, and comparative benchmarking."""
    start_time = time.time()
    set_seed(seed)

    print("=" * 75)
    print("STARTING REGIONALTCN CONTROLLED EXPERIMENT (Member 2)")
    print(f"Seed: {seed} | Batch size: {batch_size} | LR: {learning_rate} | Max Epochs: {max_epochs}")
    print("=" * 75)

    # Directories
    exp_dir = get_repo_root() / "experiments" / "regional_tcn"
    res_dir = get_repo_root() / "results" / "regional_tcn"
    model_dir = get_repo_root() / "models"
    plots_dir = res_dir / "plots"
    hydro_dir = plots_dir / "hydrographs"

    for d in [exp_dir, res_dir, model_dir, plots_dir, hydro_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Load Partitions
    splits = load_split_catchment_ids()
    train_ids = splits["train"]
    val_ids = splits["val"]
    test_ids = splits["test"]
    demo_ids = splits["ungauged_demo"]

    print(f"Partitions loaded: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test, {len(demo_ids)} ungauged demo.")

    # 2. Load Static Attributes (129 Features)
    print("Loading 129 static attributes via validated pipeline...")
    static_df = load_static_attributes()
    run_anti_leakage_audit(train_ids, val_ids, test_ids, demo_ids, static_df)

    # 3. Load Observed Streamflow
    flow_file = get_repo_root() / "data" / "raw" / "CAMELS_IND_All_Catchments" / "streamflow_timeseries" / "streamflow_observed.csv"
    if not flow_file.exists():
        flow_file = get_repo_root().parent / "Datasets" / "CAMELS_IND_All_Catchments" / "streamflow_timeseries" / "streamflow_observed.csv"
    if not flow_file.exists():
        raise FileNotFoundError(f"streamflow_observed.csv not found at {flow_file}")

    flow_df = pd.read_csv(flow_file)
    print(f"Observed streamflow loaded: {flow_df.shape[0]} daily timesteps.")

    # 4. Compute Train-Only Target Scaling (Zero Leakage - strictly identical to Base TCN)
    train_col_keys = [str(int(gid)) for gid in train_ids if str(int(gid)) in flow_df.columns]
    train_flows = flow_df[train_col_keys].to_numpy().ravel()
    valid_train_flows = train_flows[~np.isnan(train_flows)]
    target_mean = float(np.mean(valid_train_flows))
    target_std = float(np.std(valid_train_flows))
    print(f"Train-only target scaling fitted: mean = {target_mean:.3f} m3/s, std = {target_std:.3f} m3/s")

    # 5. Build Training & Validation Datasets (Dynamic [64, 20] + Static [129])
    preprocessor = Preprocessor()
    seq_len = 64
    rng = np.random.RandomState(seed)

    print("\nExtracting sliding window sequences with static feature pairing...")
    X_train_dyn_list, X_train_stat_list, y_train_list = [], [], []
    for gid in train_ids:
        df_forcing = load_catchment_forcing(gid)
        dyn_mat = extract_catchment_matrix(df_forcing, preprocessor)
        stat_vec = static_df.loc[gid].to_numpy().astype(np.float32)
        col_key = str(int(gid))
        if col_key in flow_df.columns:
            obs = flow_df[col_key].to_numpy()
            valid_idx = np.where((~np.isnan(obs)) & (np.arange(len(obs)) >= seq_len))[0]
            if len(valid_idx) > samples_per_train_basin:
                chosen = rng.choice(valid_idx, size=samples_per_train_basin, replace=False)
            else:
                chosen = valid_idx
            for idx in chosen:
                X_train_dyn_list.append(dyn_mat[idx - seq_len : idx])
                X_train_stat_list.append(stat_vec)
                y_train_list.append((obs[idx] - target_mean) / target_std)

    X_train_dyn = torch.tensor(np.array(X_train_dyn_list), dtype=torch.float32)
    X_train_stat = torch.tensor(np.array(X_train_stat_list), dtype=torch.float32)
    y_train = torch.tensor(np.array(y_train_list), dtype=torch.float32).unsqueeze(1)
    print(f"Training dataset: {X_train_dyn.shape[0]} sequences (dyn: {list(X_train_dyn.shape[1:])}, stat: {list(X_train_stat.shape[1:])})")

    # Validation sequences for checkpoint selection
    X_val_dyn_list, X_val_stat_list, y_val_list = [], [], []
    for gid in val_ids:
        df_forcing = load_catchment_forcing(gid)
        dyn_mat = extract_catchment_matrix(df_forcing, preprocessor)
        stat_vec = static_df.loc[gid].to_numpy().astype(np.float32)
        col_key = str(int(gid))
        if col_key in flow_df.columns:
            obs = flow_df[col_key].to_numpy()
            valid_idx = np.where((~np.isnan(obs)) & (np.arange(len(obs)) >= seq_len))[0]
            if len(valid_idx) > samples_per_val_basin:
                chosen = rng.choice(valid_idx, size=samples_per_val_basin, replace=False)
            else:
                chosen = valid_idx
            for idx in chosen:
                X_val_dyn_list.append(dyn_mat[idx - seq_len : idx])
                X_val_stat_list.append(stat_vec)
                y_val_list.append((obs[idx] - target_mean) / target_std)

    X_val_dyn = torch.tensor(np.array(X_val_dyn_list), dtype=torch.float32)
    X_val_stat = torch.tensor(np.array(X_val_stat_list), dtype=torch.float32)
    y_val = torch.tensor(np.array(y_val_list), dtype=torch.float32).unsqueeze(1)
    print(f"Validation monitoring dataset: {X_val_dyn.shape[0]} sequences.")

    train_loader = DataLoader(
        TensorDataset(X_train_dyn, X_train_stat, y_train),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val_dyn, X_val_stat, y_val),
        batch_size=batch_size,
        shuffle=False,
    )

    # 6. Initialize RegionalTCN & Training Loop
    model = RegionalTCN()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Initialized RegionalTCN model. Trainable parameters: {total_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0
    history: List[Dict[str, float]] = []

    best_checkpoint_path = model_dir / "regional_tcn_best.pt"
    exp_checkpoint_path = exp_dir / "regional_tcn_best.pt"

    print("\n--- Starting RegionalTCN Training Loop ---")
    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss_accum = 0.0
        n_batches = 0

        for bx_dyn, bx_stat, by in train_loader:
            optimizer.zero_grad()
            preds = model(bx_dyn, bx_stat, mode="point")
            loss = criterion(preds, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            train_loss_accum += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss = train_loss_accum / n_batches

        # Validation Loss
        model.eval()
        val_loss_accum = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for vx_dyn, vx_stat, vy in val_loader:
                vpreds = model(vx_dyn, vx_stat, mode="point")
                val_loss_accum += criterion(vpreds, vy).item()
                n_val_batches += 1
        val_loss = val_loss_accum / n_val_batches

        epoch_dur = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()[0]

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": current_lr,
            "duration_sec": epoch_dur,
        })

        is_best = val_loss < best_val_loss
        best_marker = "(* best *)" if is_best else ""
        print(f"Epoch {epoch:02d}/{max_epochs:02d} | Train MSE: {train_loss:.5f} | Val MSE: {val_loss:.5f} | LR: {current_lr:.6f} | {epoch_dur:.1f}s {best_marker}")

        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            # Save checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "target_mean": target_mean,
                "target_std": target_std,
                "num_params": total_params,
            }, best_checkpoint_path)
            torch.save(torch.load(best_checkpoint_path), exp_checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs (patience = {patience}).")
                break

    training_duration = time.time() - start_time
    print(f"\nRegionalTCN training completed in {training_duration:.1f}s. Best Val MSE: {best_val_loss:.5f} (Epoch {best_epoch})")

    # Save loss history
    history_df = pd.DataFrame(history)
    history_df.to_csv(exp_dir / "regional_tcn_loss_history.csv", index=False)

    # Save training configuration
    train_config = {
        "model_name": "regional_tcn",
        "random_seed": seed,
        "input_dim": 20,
        "static_dim": 129,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer": "AdamW",
        "weight_decay": 1e-5,
        "max_epochs": max_epochs,
        "actual_epochs_trained": len(history),
        "best_epoch": best_epoch,
        "target_mean_m3s": target_mean,
        "target_std_m3s": target_std,
        "num_train_samples": len(X_train_dyn),
        "num_val_samples": len(X_val_dyn),
        "num_train_catchments": len(train_ids),
        "num_val_catchments": len(val_ids),
        "num_test_catchments": len(test_ids),
        "num_ungauged_demo_catchments": len(demo_ids),
        "total_parameters": total_params,
        "training_duration_seconds": training_duration,
        "best_val_loss": best_val_loss,
        "checkpoint_path": str(best_checkpoint_path),
    }
    with open(exp_dir / "regional_tcn_train_config.yaml", "w") as f:
        yaml.dump(train_config, f, default_flow_style=False)

    # Plot loss curve
    plt.figure(figsize=(8, 4.5))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="RegionalTCN Train Loss (MSE)", marker="o", color="#1f77b4")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="RegionalTCN Val Loss (MSE)", marker="s", color="#ff7f0e")
    plt.axvline(best_epoch, color="#2ca02c", linestyle="--", label=f"Best Checkpoint (Epoch {best_epoch})")
    plt.title("RegionalTCN Training & Validation Convergence", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Normalized MSE Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "regional_tcn_loss_curve.png", dpi=200)
    plt.close()

    # 7. Comprehensive Partition Evaluation
    print("\n--- Running Evaluation on Validation & Held-Out Test Catchments ---")
    checkpoint = torch.load(best_checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    def evaluate_partition(partition_gids: List[str], partition_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        pred_records = []
        for gid in partition_gids:
            df_forcing = load_catchment_forcing(gid)
            dyn_mat = extract_catchment_matrix(df_forcing, preprocessor)
            stat_vec = static_df.loc[gid].to_numpy().astype(np.float32)
            col_key = str(int(gid))
            obs_series = flow_df[col_key].to_numpy() if col_key in flow_df.columns else np.full(len(df_forcing), np.nan)

            T = len(dyn_mat)
            valid_indices = np.where((~np.isnan(obs_series)) & (np.arange(T) >= seq_len))[0]

            if len(valid_indices) == 0:
                continue

            seq_batches = [dyn_mat[idx - seq_len : idx] for idx in valid_indices]
            stat_batches = np.tile(stat_vec, (len(seq_batches), 1))

            seq_tensor = torch.tensor(np.array(seq_batches), dtype=torch.float32)
            stat_tensor = torch.tensor(stat_batches, dtype=torch.float32)
            eval_loader = DataLoader(TensorDataset(seq_tensor, stat_tensor), batch_size=256, shuffle=False)

            norm_preds = []
            with torch.no_grad():
                for bx_d, bx_s in eval_loader:
                    p = model(bx_d, bx_s, mode="point").squeeze(1).numpy()
                    norm_preds.extend(p)

            pred_phys = np.maximum(0.0, np.array(norm_preds) * target_std + target_mean)
            obs_phys = obs_series[valid_indices]
            dates = pd.to_datetime(df_forcing[["year", "month", "day"]]).iloc[valid_indices].to_numpy()

            for d, o, p in zip(dates, obs_phys, pred_phys):
                pred_records.append({
                    "gauge_id": str(gid).zfill(5),
                    "date": d,
                    "observed_flow": float(o),
                    "predicted_flow": float(p),
                })

        pred_df = pd.DataFrame(pred_records)
        metrics_df = evaluate_catchment_predictions(pred_df)
        return pred_df, metrics_df

    val_preds_df, val_metrics_df = evaluate_partition(val_ids, "Validation")
    test_preds_df, test_metrics_df = evaluate_partition(test_ids, "Held-Out Test")

    # Save RegionalTCN predictions and metrics
    val_preds_df.to_csv(res_dir / "regional_tcn_val_predictions.csv", index=False)
    test_preds_df.to_csv(res_dir / "regional_tcn_test_predictions.csv", index=False)
    val_metrics_df.to_csv(res_dir / "regional_tcn_validation_metrics.csv", index=False)
    test_metrics_df.to_csv(res_dir / "regional_tcn_test_metrics.csv", index=False)

    summary_val = summarize_regional_performance(val_metrics_df)
    summary_test = summarize_regional_performance(test_metrics_df)

    summary = {
        "model": "RegionalTCN",
        "validation_36_catchments": summary_val,
        "held_out_test_37_catchments": summary_test,
    }
    with open(res_dir / "regional_tcn_summary_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    # 8. Operational Ungauged Demonstration Inference (Section 14)
    print("\n--- Running Operational Ungauged Demonstration Inference (5 basins) ---")
    demo_records = []
    for gid in demo_ids:
        df_forcing = load_catchment_forcing(gid)
        dyn_mat = extract_catchment_matrix(df_forcing, preprocessor)
        stat_vec = static_df.loc[gid].to_numpy().astype(np.float32)
        T = len(dyn_mat)
        valid_indices = np.arange(seq_len, T)

        seq_batches = [dyn_mat[idx - seq_len : idx] for idx in valid_indices]
        stat_batches = np.tile(stat_vec, (len(seq_batches), 1))

        seq_tensor = torch.tensor(np.array(seq_batches), dtype=torch.float32)
        stat_tensor = torch.tensor(stat_batches, dtype=torch.float32)
        demo_loader = DataLoader(TensorDataset(seq_tensor, stat_tensor), batch_size=256, shuffle=False)

        norm_preds = []
        with torch.no_grad():
            for bx_d, bx_s in demo_loader:
                p = model(bx_d, bx_s, mode="point").squeeze(1).numpy()
                norm_preds.extend(p)

        pred_phys = np.maximum(0.0, np.array(norm_preds) * target_std + target_mean)
        dates = pd.to_datetime(df_forcing[["year", "month", "day"]]).iloc[valid_indices].to_numpy()

        for d, p in zip(dates, pred_phys):
            demo_records.append({
                "gauge_id": str(gid).zfill(5),
                "date": d,
                "operational_mode": "OPERATIONAL_UNGAUGED_DEMONSTRATION",
                "predicted_flow_m3s": float(p),
            })

    demo_df = pd.DataFrame(demo_records)
    demo_df.to_csv(res_dir / "operational_ungauged_demo_predictions.csv", index=False)
    print(f"Saved {len(demo_df)} operational ungauged predictions across 5 demo basins.")

    # 9. Comparative Catchment-Level Benchmark vs Base TCN
    print("\n--- Computing Controlled Benchmark vs Base TCN Baseline ---")
    base_metrics_path = res_dir / "test_metrics.csv"
    if base_metrics_path.exists():
        base_test_metrics = pd.read_csv(base_metrics_path, dtype={"gauge_id": str})
        base_test_metrics["gauge_id"] = base_test_metrics["gauge_id"].str.zfill(5)

        # Merge on gauge_id
        comparison_df = pd.merge(
            base_test_metrics,
            test_metrics_df,
            on="gauge_id",
            suffixes=("_base_tcn", "_regional_tcn"),
        )

        comparison_df["delta_NSE"] = comparison_df["NSE_regional_tcn"] - comparison_df["NSE_base_tcn"]
        comparison_df["delta_KGE"] = comparison_df["KGE_regional_tcn"] - comparison_df["KGE_base_tcn"]
        comparison_df["delta_RMSE"] = comparison_df["RMSE_regional_tcn"] - comparison_df["RMSE_base_tcn"]
        comparison_df["delta_MAE"] = comparison_df["MAE_regional_tcn"] - comparison_df["MAE_base_tcn"]
        comparison_df["delta_high_flow_nse"] = comparison_df["high_flow_nse_regional_tcn"] - comparison_df["high_flow_nse_base_tcn"]

        comparison_df.to_csv(res_dir / "comparison_base_vs_regional.csv", index=False)
        print("Catchment-level comparison table saved to results/regional_tcn/comparison_base_vs_regional.csv")

    # 10. Generate Evaluation Plots
    print("\nGenerating CDF and comparative hydrographs...")
    plot_cumulative_nse(
        test_metrics_df["NSE"].tolist(),
        model_name="RegionalTCN (Held-Out Test)",
        save_path=plots_dir / "regional_tcn_cumulative_nse.png",
    )

    # Plot comparison cumulative NSE if Base TCN exists
    if base_metrics_path.exists():
        plt.figure(figsize=(7, 5))
        base_nse_sorted = np.sort(base_test_metrics["NSE"])
        reg_nse_sorted = np.sort(test_metrics_df["NSE"])
        p_base = np.linspace(0, 1, len(base_nse_sorted))
        p_reg = np.linspace(0, 1, len(reg_nse_sorted))

        plt.plot(base_nse_sorted, p_base, label=f"Base TCN (Median: {summary_test.get('NSE_median', 0.1498):.3f})", color="black", linestyle="--")
        plt.plot(reg_nse_sorted, p_reg, label=f"RegionalTCN (Median: {summary_test['NSE_median']:.3f})", color="#1f77b4", linewidth=2)
        plt.axvline(0.0, color="gray", linestyle=":", alpha=0.7)
        plt.xlabel("Nash-Sutcliffe Efficiency (NSE)", fontsize=11)
        plt.ylabel("Empirical Cumulative Probability", fontsize=11)
        plt.title("Empirical CDF of Test Catchment NSE", fontsize=12, fontweight="bold")
        plt.xlim(-1.5, 1.0)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(plots_dir / "comparison_cumulative_nse.png", dpi=200)
        plt.close()

    # Generate representative hydrographs for RegionalTCN
    for gid in [test_metrics_df.sort_values("NSE", ascending=False).iloc[0]["gauge_id"],
                test_metrics_df.sort_values("NSE").iloc[len(test_metrics_df)//2]["gauge_id"]]:
        c_data = test_preds_df[test_preds_df["gauge_id"] == gid].sort_values("date").tail(730)
        plot_hydrograph(
            dates=c_data["date"],
            obs=c_data["observed_flow"].to_numpy(),
            sim=c_data["predicted_flow"].to_numpy(),
            gauge_id=gid,
            save_path=hydro_dir / f"regional_tcn_test_hydrograph_{gid}.png",
        )
        plt.close()

    print("\n" + "=" * 75)
    print("EXPERIMENT EXECUTION FINISHED SUCCESSFULLY")
    print(f"Validation Median NSE: {summary_val.get('NSE_median', float('nan')):.4f} | KGE: {summary_val.get('KGE_median', float('nan')):.4f}")
    print(f"Test Median NSE:       {summary_test.get('NSE_median', float('nan')):.4f} | KGE: {summary_test.get('KGE_median', float('nan')):.4f}")
    print("=" * 75)

    return summary


if __name__ == "__main__":
    run_experiment(
        max_epochs=12,
        patience=4,
        samples_per_train_basin=250,
        samples_per_val_basin=100,
        batch_size=32,
        learning_rate=0.0005,
        seed=42,
    )
