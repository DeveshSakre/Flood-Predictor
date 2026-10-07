"""
Asymmetric Pinball (Quantile) Loss functions and neural loss modules.

Supports:
- MultiQuantilePinballLoss: Differentiable PyTorch module for training neural quantile heads (P10, P50, P90) with missing observation masking.
- pinball_loss: Standard evaluation metric for a single quantile level tau in (0, 1).
- pinball_loss_multi: Composite evaluation metric across multiple quantile levels.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union
import numpy as np
import torch
import torch.nn as nn


class MultiQuantilePinballLoss(nn.Module):
    """
    Differentiable multi-quantile pinball (tilted) loss module.

    Computes:
        L_tau(y, q) = max(tau * (y - q), (tau - 1) * (y - q))

    Supports:
        - Arbitrary strictly increasing quantiles (default: 0.10, 0.50, 0.90)
        - Weighted composite loss across quantiles
        - Missing observation masking (e.g. unobserved or NaN streamflow timesteps)
        - Both 2D predictions [B, Q] and 3D sequence predictions [B, T, Q]
    """

    def __init__(
        self,
        quantiles: Sequence[float] = (0.10, 0.50, 0.90),
        weights: Optional[Sequence[float]] = None,
    ):
        super().__init__()

        q = torch.tensor(quantiles, dtype=torch.float32)

        if torch.any(q <= 0) or torch.any(q >= 1):
            raise ValueError("All quantiles must lie strictly in (0, 1).")

        if not torch.all(q[:-1] < q[1:]):
            raise ValueError("Quantiles must be strictly increasing.")

        self.register_buffer("quantiles", q)

        if weights is None:
            weights = torch.ones_like(q)
        else:
            weights = torch.tensor(weights, dtype=torch.float32)
            if weights.shape != q.shape:
                raise ValueError("Weights must have the same length as quantiles.")

        weights = weights / weights.sum()
        self.register_buffer("weights", weights)

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for multi-quantile pinball loss.

        Args:
            y_pred: Predicted quantiles, shape [B, Q] or [B, T, Q].
            y_true: Ground truth target, shape [B], [B, 1], [B, T], or [B, T, 1].
            mask: Optional boolean or float mask indicating valid observations, shape [B] or [B, T].

        Returns:
            Scalar loss tensor.
        """
        if y_pred.shape[-1] != len(self.quantiles):
            raise ValueError(
                f"Last prediction dimension ({y_pred.shape[-1]}) must "
                f"equal number of quantiles ({len(self.quantiles)})."
            )

        # Align target tensor with predictions
        if y_true.ndim == y_pred.ndim - 1:
            y_true = y_true.unsqueeze(-1)

        q = self.quantiles.to(device=y_pred.device, dtype=y_pred.dtype)
        weights = self.weights.to(device=y_pred.device, dtype=y_pred.dtype)

        # Pinball loss formula: max(tau * error, (tau - 1) * error) where error = y_true - y_pred
        error = y_true - y_pred
        loss = torch.maximum(q * error, (q - 1.0) * error)

        # Apply quantile weighting and sum across quantile dimension
        loss = (loss * weights).sum(dim=-1)

        # Apply missing-observation mask if supplied
        if mask is not None:
            mask = mask.to(device=y_pred.device, dtype=loss.dtype)
            while mask.ndim < loss.ndim:
                mask = mask.unsqueeze(-1)
            loss = loss * mask
            denominator = mask.sum().clamp_min(1.0)
            return loss.sum() / denominator

        return loss.mean()


def pinball_loss(
    y_true: Union[np.ndarray, Sequence[float]],
    y_pred: Union[np.ndarray, Sequence[float]],
    tau: float,
) -> float:
    """
    Compute mean pinball loss for a single quantile tau in (0, 1).

    Formula:
        L_tau(y, q) = max(tau * (y - q), (tau - 1.0) * (y - q))

    Args:
        y_true: Ground truth observations (1D array or sequence).
        y_pred: Predicted quantile values (1D array or sequence).
        tau: Target quantile level in (0, 1), e.g. 0.10, 0.50, 0.90.

    Returns:
        Mean pinball loss as float.
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)

    # Filter out NaNs/Infs from missing observations
    valid = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    if not np.any(valid):
        return 0.0

    residual = y_true_arr[valid] - y_pred_arr[valid]
    loss = np.maximum(tau * residual, (tau - 1.0) * residual)
    return float(np.mean(loss))


def pinball_loss_multi(
    y_true: Union[np.ndarray, Sequence[float]],
    preds_dict: Dict[float, Union[np.ndarray, Sequence[float]]],
    weights: Optional[Sequence[float]] = None,
) -> float:
    """
    Compute composite weighted pinball loss across multiple quantiles.

    Args:
        y_true: Ground truth observations.
        preds_dict: Mapping from quantile level tau (e.g. 0.1, 0.5, 0.9) to predictions.
        weights: Optional weights for each quantile (default: equal weights).

    Returns:
        Composite weighted pinball loss.
    """
    taus = list(preds_dict.keys())
    if weights is None:
        weights = [1.0 / len(taus)] * len(taus)
    else:
        w_sum = sum(weights)
        weights = [w / w_sum for w in weights]

    total_loss = 0.0
    for tau, w in zip(taus, weights):
        loss_tau = pinball_loss(y_true, preds_dict[tau], tau=float(tau))
        total_loss += w * loss_tau

    return float(total_loss)