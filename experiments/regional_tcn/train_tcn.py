"""
Training and Evaluation Pipeline for Base Temporal Convolutional Network (TCN).

Adheres strictly to the multi-catchment leakage-safe protocol:
- Trains strictly on 169 source catchments
- Evaluates validation loss and early stopping on 36 validation catchments
- Evaluates generalization on 37 held-out unseen test catchments
- Never evaluates 5 operational zero-flow demonstration catchments on NSE/KGE
- Uses train-fitted preprocessing and saves checkpoints, metrics, and plots.

RESPONSIBILITY: Member 2 (Temporal Modeling / TCN)
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
from src.data.loaders import load_split_catchment_ids, load_catchment_forcing
from src.data.preprocessing import Preprocessor
from src.models.tcn import TCNModel
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.evaluation_pipeline import evaluate_catchment_predictions, summarize_regional_performance
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


def run_experiment(
    max_epochs: int = 15,
    patience: int = 4,
    samples_per_train_basin: int = 250,
    samples_per_val_basin: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.0005,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute complete TCN training, validation, and evaluation experiment."""
    start_time = time.time()
    set_seed(seed)

    print("=" * 70)
    print("STARTING TCN BASELINE EXPERIMENT (Member 2)")
    print(f"Seed: {seed} | Batch size: {batch_size} | LR: {learning_rate} | Max Epochs: {max_epochs}")
    print("=" * 70)

    # Output directories
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

    print(f"Loaded partitions: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test catchments.")

    # 2. Load Observed Streamflow
    flow_file = get_repo_root() / "data" / "raw" / "CAMELS_IND_All_Catchments" / "streamflow_timeseries" / "streamflow_observed.csv"
    if not flow_file.exists():
        flow_file = get_repo_root().parent / "Datasets" / "CAMELS_IND_All_Catchments" / "streamflow_timeseries" / "streamflow_observed.csv"
    if not flow_file.exists():
        raise FileNotFoundError(f"streamflow_observed.csv not found at {flow_file}")

    flow_df = pd.read_csv(flow_file)
    print(f"Observed streamflow loaded: {flow_df.shape[0]} daily timesteps.")

    # 3. Compute Train-Only Target Scaling (Zero Leakage)
    train_col_keys = [str(int(gid)) for gid in train_ids if str(int(gid)) in flow_df.columns]
    train_flows = flow_df[train_col_keys].to_numpy().ravel()
    valid_train_flows = train_flows[~np.isnan(train_flows)]
    target_mean = float(np.mean(valid_train_flows))
    target_std = float(np.std(valid_train_flows))
    print(f"Train-only target scaling fitted: mean = {target_mean:.3f} m3/s, std = {target_std:.3f} m3/s")

    # 4. Build Training & Validation Datasets
    preprocessor = Preprocessor()
    seq_len = 64
    rng = np.random.RandomState(seed)

    print("\nExtracting sliding window sequences...")
    X_train_list, y_train_list = [], []
    for gid in train_ids:
        df_forcing = load_catchment_forcing(gid)
        dyn_mat = extract_catchment_matrix(df_forcing, preprocessor)
        col_key = str(int(gid))
        if col_key in flow_df.columns:
            obs = flow_df[col_key].to_numpy()
            valid_idx = np.where((~np.isnan(obs)) & (np.arange(len(obs)) >= seq_len))[0]
            if len(valid_idx) > samples_per_train_basin:
                chosen = rng.choice(valid_idx, size=samples_per_train_basin, replace=False)
            else:
                chosen = valid_idx
            for idx in chosen:
                X_train_list.append(dyn_mat[idx - seq_len : idx])
                y_train_list.append((obs[idx] - target_mean) / target_std)

    X_train = torch.tensor(np.array(X_train_list), dtype=torch.float32)
    y_train = torch.tensor(np.array(y_train_list), dtype=torch.float32).unsqueeze(1)
    print(f"Training dataset: {X_train.shape[0]} sequences of shape {list(X_train.shape[1:])}")

    # Validation sequences for checkpoint selection
    X_val_list, y_val_list = [], []
    for gid in val_ids:
        df_forcing = load_catchment_forcing(gid)
        dyn_mat = extract_catchment_matrix(df_forcing, preprocessor)
        col_key = str(int(gid))
        if col_key in flow_df.columns:
            obs = flow_df[col_key].to_numpy()
            valid_idx = np.where((~np.isnan(obs)) & (np.arange(len(obs)) >= seq_len))[0]
            if len(valid_idx) > samples_per_val_basin:
                chosen = rng.choice(valid_idx, size=samples_per_val_basin, replace=False)
            else:
                chosen = valid_idx
            for idx in chosen:
                X_val_list.append(dyn_mat[idx - seq_len : idx])
                y_val_list.append((obs[idx] - target_mean) / target_std)

    X_val = torch.tensor(np.array(X_val_list), dtype=torch.float32)
    y_val = torch.tensor(np.array(y_val_list), dtype=torch.float32).unsqueeze(1)
    print(f"Validation monitoring dataset: {X_val.shape[0]} sequences.")

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    # 5. Initialize Model & Training Loop
    model = TCNModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    best_val_loss = float("inf")
    patience_counter = 0
    history: List[Dict[str, float]] = []
    best_checkpoint_path = model_dir / "tcn_baseline_best.pt"
    exp_checkpoint_path = exp_dir / "tcn_baseline_best.pt"

    print("\n--- Training Loop ---")
    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss_accum = 0.0
        n_batches = 0

        for bx, by in train_loader:
            optimizer.zero_grad()
            preds = model(bx, mode="point")
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
            for vx, vy in val_loader:
                vpreds = model(vx, mode="point")
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
            patience_counter = 0
            # Save checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "target_mean": target_mean,
                "target_std": target_std,
                "config": model.tcn_channels,
            }, best_checkpoint_path)
            torch.save(torch.load(best_checkpoint_path), exp_checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs (patience = {patience}).")
                break

    training_duration = time.time() - start_time
    print(f"\nTraining completed in {training_duration:.1f}s. Best Val MSE: {best_val_loss:.5f}")

    # Save loss history
    history_df = pd.DataFrame(history)
    history_df.to_csv(exp_dir / "loss_history.csv", index=False)

    # Save training configuration
    train_config = {
        "model_name": "tcn_baseline",
        "random_seed": seed,
        "input_dim": 20,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer": "AdamW",
        "weight_decay": 1e-5,
        "max_epochs": max_epochs,
        "target_mean_m3s": target_mean,
        "target_std_m3s": target_std,
        "num_train_samples": len(X_train),
        "num_val_samples": len(X_val),
        "num_train_catchments": len(train_ids),
        "num_val_catchments": len(val_ids),
        "num_test_catchments": len(test_ids),
        "training_duration_seconds": training_duration,
        "best_val_loss": best_val_loss,
    }
    with open(exp_dir / "tcn_train_config.yaml", "w") as f:
        yaml.dump(train_config, f, default_flow_style=False)

    # Plot loss curve
    plt.figure(figsize=(8, 4.5))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss (Normalized MSE)", marker="o")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="Val Loss (Normalized MSE)", marker="s")
    plt.title("TCN Baseline Training & Validation Convergence", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "loss_curve.png", dpi=200)
    plt.close()

    # 6. Comprehensive Catchment Evaluation
    print("\n--- Running Comprehensive Evaluation on Validation & Test Catchments ---")
    checkpoint = torch.load(best_checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    def evaluate_partition(partition_gids: List[str], partition_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        pred_records = []
        for gid in partition_gids:
            df_forcing = load_catchment_forcing(gid)
            dyn_mat = extract_catchment_matrix(df_forcing, preprocessor)
            col_key = str(int(gid))
            obs_series = flow_df[col_key].to_numpy() if col_key in flow_df.columns else np.full(len(df_forcing), np.nan)

            # Subsample evaluation to non-NaN observed dates (stride of 1 for complete continuity or consecutive test blocks)
            # Efficient sliding window evaluation:
            T = len(dyn_mat)
            valid_indices = np.where((~np.isnan(obs_series)) & (np.arange(T) >= seq_len))[0]

            if len(valid_indices) == 0:
                continue

            # Batch inference over all valid days for this catchment
            seq_batches = []
            for idx in valid_indices:
                seq_batches.append(dyn_mat[idx - seq_len : idx])

            seq_tensor = torch.tensor(np.array(seq_batches), dtype=torch.float32)
            eval_loader = DataLoader(TensorDataset(seq_tensor), batch_size=256, shuffle=False)

            norm_preds = []
            with torch.no_grad():
                for (bx,) in eval_loader:
                    p = model(bx, mode="point").squeeze(1).numpy()
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

    # Save predictions and metrics
    val_preds_df.to_csv(res_dir / "val_predictions.csv", index=False)
    test_preds_df.to_csv(res_dir / "test_predictions.csv", index=False)
    val_metrics_df.to_csv(res_dir / "validation_metrics.csv", index=False)
    test_metrics_df.to_csv(res_dir / "test_metrics.csv", index=False)

    summary_val = summarize_regional_performance(val_metrics_df)
    summary_test = summarize_regional_performance(test_metrics_df)

    summary = {
        "validation_36_catchments": summary_val,
        "held_out_test_37_catchments": summary_test,
    }
    with open(res_dir / "summary_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    # 7. Generate Hydrograph Plots
    print("\nGenerating representative hydrographs and CDF plots...")
    plot_cumulative_nse(test_metrics_df["NSE"].tolist(), model_name="TCN Baseline (Test)", save_path=plots_dir / "cumulative_nse.png")

    # Plot representative hydrographs (best, median, challenging for val and test)
    for p_name, p_df, m_df in [("val", val_preds_df, val_metrics_df), ("test", test_preds_df, test_metrics_df)]:
        sorted_m = m_df.sort_values(by="NSE", ascending=False)
        rep_ids = [sorted_m.iloc[0]["gauge_id"], sorted_m.iloc[len(sorted_m)//2]["gauge_id"], sorted_m.iloc[-1]["gauge_id"]]
        for gid in rep_ids:
            c_data = p_df[p_df["gauge_id"] == gid].sort_values("date").tail(730) # last 2 years of observations
            plot_hydrograph(
                dates=c_data["date"],
                obs=c_data["observed_flow"].to_numpy(),
                sim=c_data["predicted_flow"].to_numpy(),
                gauge_id=gid,
                save_path=hydro_dir / f"{p_name}_hydrograph_{gid}.png",
            )
            plt.close()

    print("\n" + "=" * 70)
    print("EXPERIMENT EXECUTION FINISHED SUCCESSFULLY")
    print(f"Validation Median NSE: {summary_val.get('NSE_median', float('nan')):.4f} | KGE: {summary_val.get('KGE_median', float('nan')):.4f}")
    print(f"Test Median NSE:       {summary_test.get('NSE_median', float('nan')):.4f} | KGE: {summary_test.get('KGE_median', float('nan')):.4f}")
    print("=" * 70)

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
