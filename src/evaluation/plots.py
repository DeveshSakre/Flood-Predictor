"""
Hydrologic visualization utilities for model hydrographs and uncertainty bands.
"""

from pathlib import Path
from typing import Optional, Union, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_hydrograph(
    dates: pd.Series,
    obs: np.ndarray,
    sim: np.ndarray,
    gauge_id: str,
    q10: Optional[np.ndarray] = None,
    q90: Optional[np.ndarray] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple[int, int] = (12, 5),
) -> plt.Figure:
    """
    Plot observed vs predicted hydrograph with optional uncertainty ribbon (P10-P90).
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(dates, obs, label="Observed Flow (Ground Truth)", color="black", linewidth=1.5, alpha=0.85)
    ax.plot(dates, sim, label="Predicted Flow (Median / P50)", color="#2563eb", linewidth=1.5)

    if q10 is not None and q90 is not None:
        ax.fill_between(
            dates,
            q10,
            q90,
            color="#93c5fd",
            alpha=0.45,
            label="80% Uncertainty Band (P10-P90)",
        )

    ax.set_title(f"Streamflow Hydrograph — Catchment {gauge_id}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Streamflow (m³/s)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)

    return fig


def plot_cumulative_nse(
    nse_scores: List[float],
    model_name: str = "Model",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot cumulative empirical CDF of NSE across unseen catchments.
    Standard hydrologic benchmark visualization.
    """
    scores = np.sort(np.asarray(nse_scores))
    scores = scores[~np.isnan(scores)]
    y_vals = np.linspace(0, 1, len(scores))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(scores, y_vals, label=model_name, linewidth=2, color="#1e40af")
    ax.axvline(0, color="gray", linestyle=":", label="NSE = 0 Baseline")
    ax.set_xlabel("Nash-Sutcliffe Efficiency (NSE)", fontsize=11)
    ax.set_ylabel("Cumulative Frequency", fontsize=11)
    ax.set_title("Empirical CDF of Ungauged Catchment NSE", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlim(-1.0, 1.0)
    ax.legend(loc="lower right")
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)

    return fig
