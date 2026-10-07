"""
Generate representative hydrograph comparisons for Member 5 benchmark comparison.
Creates composite side-by-side comparison figures for representative unseen catchments:
- Good: Catchment 08013 (Mahanadi, Area 19,600 km²)
- Average: Catchment 08029 (Mahanadi, Area 8,760 km²)
- Difficult: Catchment 08001 (Mahanadi headwater, Area 2,210 km²) & Catchment 12016 (Narmada, Area 2,292 km²)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.evaluation.plots import plot_hydrograph

def main():
    xgb_pred_path = PROJECT_ROOT / "results" / "baseline" / "xgboost_test_predictions.csv.gz"
    print(f"Loading XGBoost predictions from {xgb_pred_path}...")
    df_xgb = pd.read_csv(xgb_pred_path)
    
    # Generate XGBoost hydrograph for 08013 if not present
    xgb_08013_path = PROJECT_ROOT / "results" / "baseline" / "plots" / "hydrographs" / "test_hydrograph_08013.png"
    df_08013 = df_xgb[df_xgb["gauge_id"] == 8013].sort_values("date")
    if not df_08013.empty:
        dates = pd.to_datetime(df_08013["date"])
        obs = df_08013["y_true"].values
        sim = df_08013["y_pred"].values
        fig = plot_hydrograph(
            dates=dates,
            obs=obs,
            sim=sim,
            gauge_id="08013 (XGBoost)",
            save_path=xgb_08013_path
        )
        plt.close(fig)
        print(f"Saved XGBoost hydrograph for 08013 to {xgb_08013_path}")
        
    out_dir = PROJECT_ROOT / "results" / "comparison" / "representative_hydrographs"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Composite Good: 08013 (XGBoost vs Regional TCN)
    img_xgb = Image.open(xgb_08013_path)
    img_reg = Image.open(PROJECT_ROOT / "results" / "regional_tcn" / "plots" / "hydrographs" / "regional_tcn_test_hydrograph_08013.png")
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    axes[0].imshow(img_xgb)
    axes[0].set_title("XGBoost Baseline — Catchment 08013 (NSE: +0.282 | KGE: +0.474 | Pearson r: 0.728)", fontsize=13, fontweight="bold", pad=8)
    axes[0].axis("off")
    axes[1].imshow(img_reg)
    axes[1].set_title("Regional TCN — Catchment 08013 (NSE: +0.633 | KGE: +0.760 | Pearson r: 0.887)", fontsize=13, fontweight="bold", pad=8)
    axes[1].axis("off")
    fig.suptitle("REPRESENTATIVE GOOD CATCHMENT: 08013 (Mahanadi Basin | Drainage Area: 19,600 km²)\nRegional TCN provides superior hydrograph shape, peak timing, and high flow fidelity", fontsize=15, fontweight="bold")
    plt.tight_layout()
    comp_good_path = out_dir / "representative_good_08013.png"
    fig.savefig(comp_good_path, dpi=300)
    plt.close(fig)
    print(f"Saved {comp_good_path}")
    
    # 2. Composite Average: 08029 (XGBoost vs Regional TCN)
    xgb_08029_path = PROJECT_ROOT / "results" / "baseline" / "plots" / "hydrographs" / "test_hydrograph_08029.png"
    reg_08029_path = PROJECT_ROOT / "results" / "regional_tcn" / "plots" / "hydrographs" / "regional_tcn_test_hydrograph_08029.png"
    img_xgb_29 = Image.open(xgb_08029_path)
    img_reg_29 = Image.open(reg_08029_path)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    axes[0].imshow(img_xgb_29)
    axes[0].set_title("XGBoost Baseline — Catchment 08029 (NSE: -1.249 | KGE: -2.716 | PBIAS: +363.8% | Pearson r: 0.617)", fontsize=13, fontweight="bold", pad=8)
    axes[0].axis("off")
    axes[1].imshow(img_reg_29)
    axes[1].set_title("Regional TCN — Catchment 08029 (NSE: +0.207 | KGE: +0.428 | PBIAS: +29.1% | Pearson r: 0.692)", fontsize=13, fontweight="bold", pad=8)
    axes[1].axis("off")
    fig.suptitle("REPRESENTATIVE AVERAGE CATCHMENT: 08029 (Mahanadi Basin | Drainage Area: 8,760 km²)\nRegional TCN achieves median performance (+0.207 NSE); XGBoost suffers severe volume overestimation", fontsize=15, fontweight="bold")
    plt.tight_layout()
    comp_avg_path = out_dir / "representative_average_08029.png"
    fig.savefig(comp_avg_path, dpi=300)
    plt.close(fig)
    print(f"Saved {comp_avg_path}")
    
    # 3. Composite Difficult: 12016 (XGBoost vs Base TCN)
    xgb_12016_path = PROJECT_ROOT / "results" / "baseline" / "plots" / "hydrographs" / "test_hydrograph_12016.png"
    base_12016_path = PROJECT_ROOT / "results" / "regional_tcn" / "plots" / "hydrographs" / "test_hydrograph_12016.png"
    img_xgb_16 = Image.open(xgb_12016_path)
    img_base_16 = Image.open(base_12016_path)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    axes[0].imshow(img_xgb_16)
    axes[0].set_title("XGBoost Baseline — Catchment 12016 (NSE: -30.19 | KGE: -47.07 | Pearson r: 0.627)", fontsize=13, fontweight="bold", pad=8)
    axes[0].axis("off")
    axes[1].imshow(img_base_16)
    axes[1].set_title("Base TCN — Catchment 12016 (NSE: +0.150 | KGE: -0.082 | Pearson r: 0.672)", fontsize=13, fontweight="bold", pad=8)
    axes[1].axis("off")
    fig.suptitle("REPRESENTATIVE CHALLENGING CATCHMENT: 12016 (Narmada Basin | Drainage Area: 2,292 km²)\nBase TCN retains positive NSE (+0.150); XGBoost exhibits massive scaling mismatch in small basin", fontsize=15, fontweight="bold")
    plt.tight_layout()
    comp_diff_path = out_dir / "representative_difficult_12016.png"
    fig.savefig(comp_diff_path, dpi=300)
    plt.close(fig)
    print(f"Saved {comp_diff_path}")
    
    # 4. Composite Difficult Headwater: 08001 (XGBoost vs Base TCN)
    xgb_08001_path = PROJECT_ROOT / "results" / "baseline" / "plots" / "hydrographs" / "test_hydrograph_08001.png"
    base_08001_path = PROJECT_ROOT / "results" / "regional_tcn" / "plots" / "hydrographs" / "test_hydrograph_08001.png"
    img_xgb_01 = Image.open(xgb_08001_path)
    img_base_01 = Image.open(base_08001_path)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    axes[0].imshow(img_xgb_01)
    axes[0].set_title("XGBoost Baseline — Catchment 08001 (NSE: -259.20 | KGE: -188.4 | Pearson r: 0.407)", fontsize=13, fontweight="bold", pad=8)
    axes[0].axis("off")
    axes[1].imshow(img_base_01)
    axes[1].set_title("Base TCN — Catchment 08001 (NSE: -4.15 | KGE: -0.803 | Pearson r: 0.528)", fontsize=13, fontweight="bold", pad=8)
    axes[1].axis("off")
    fig.suptitle("SYSTEMATIC FAILURE CASE: 08001 (Mahanadi Headwater | Drainage Area: 2,210 km²)\nExtreme flashiness and attenuation challenge all models; Base TCN limits error orders of magnitude better", fontsize=15, fontweight="bold")
    plt.tight_layout()
    comp_fail_path = out_dir / "representative_difficult_08001.png"
    fig.savefig(comp_fail_path, dpi=300)
    plt.close(fig)
    print(f"Saved {comp_fail_path}")
    
    print("All representative hydrograph composites generated successfully!")

if __name__ == "__main__":
    main()
