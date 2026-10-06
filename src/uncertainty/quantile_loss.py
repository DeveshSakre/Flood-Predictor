"""
Asymmetric Pinball (Quantile) Loss functions.

Supports probabilistic evaluation for quantiles tau in (0, 1) (e.g. 0.10, 0.50, 0.90).
"""

from typing import List, Union
import numpy as np


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, tau: float) -> float:
    """
    Compute mean pinball loss for a single quantile tau.

    Loss formula:
        L_tau(y, q) = max(tau * (y - q), (1 - tau) * (q - y))

    Args:
        y_true: Ground truth observations (1D array).
        y_pred: Predicted quantile values (1D array).
        tau: Target quantile level in (0, 1) (e.g., 0.1, 0.5, 0.9).

    Returns:
        Mean pinball loss scalar.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residual = y_true - y_pred
    loss = np.maximum(tau * residual, (tau - 1.0) * residual)
    return float(np.mean(loss))


def pinball_loss_multi(
    y_true: np.ndarray,
    preds_dict: dict[float, np.ndarray],
    weights: Union[list[float], None] = None,
) -> float:
    """
    Compute average pinball loss across multiple quantiles (e.g. tau = 0.10, 0.50, 0.90).

    Args:
        y_true: Ground truth observations.
        preds_dict: Mapping from quantile tau to predicted array.
        weights: Optional weights for each quantile (default: equal weights).

    Returns:
        Composite weighted pinball loss.
    """
    taus = list(preds_dict.keys())
    if weights is None:
        weights = [1.0 / len(taus)] * len(taus)

    total_loss = 0.0
    for tau, w in zip(taus, weights):
        loss_tau = pinball_loss(y_true, preds_dict[tau], tau)
        total_loss += w * loss_tau

    return float(total_loss)
