"""
Neural Multi-Quantile Regression Head for Streamflow Uncertainty.

Outputs:
    P10 (0.10 quantile)
    P50 (0.50 quantile)
    P90 (0.90 quantile)

Supports:
    [B, D]    -> [B, 3]
    [B, T, D] -> [B, T, 3]

The model enforces:
    P10 <= P50 <= P90

This makes it suitable for:
    Option A:
        CAMELS-IND representation -> QuantileHead

    Option B:
        Regional TCN + GAT representation -> QuantileHead
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualMLPBlock(nn.Module):
    """
    Pre-normalized residual MLP block.

    Provides nonlinear representation learning before
    the quantile prediction layer.
    """

    def __init__(
        self,
        dim: int,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.fc1 = nn.Linear(dim, 2 * dim)
        self.fc2 = nn.Linear(2 * dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.norm(x)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.dropout(x)

        return residual + x


class QuantileHead(nn.Module):
    """
    Multi-quantile neural regression model.

    Quantiles:
        P10 = 10th percentile
        P50 = 50th percentile / median
        P90 = 90th percentile

    Input:
        [batch, features]
        or
        [batch, time, features]

    Output:
        [batch, 3]
        or
        [batch, time, 3]

    Output ordering:
        [..., 0] = P10
        [..., 1] = P50
        [..., 2] = P90
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 4,
        dropout: float = 0.10,
        quantiles: Sequence[float] = (0.10, 0.50, 0.90),
        positive_output: bool = True,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be > 0")

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")

        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        if len(quantiles) != 3:
            raise ValueError(
                "This implementation expects exactly "
                "(0.10, 0.50, 0.90)."
            )

        q = tuple(float(v) for v in quantiles)

        if not (
            0.0 < q[0] < q[1] < q[2] < 1.0
        ):
            raise ValueError(
                "Quantiles must be strictly increasing "
                "and lie between 0 and 1."
            )

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.quantiles = q
        self.positive_output = positive_output

        # ---------------------------------------------------------
        # Input projection
        # ---------------------------------------------------------

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ---------------------------------------------------------
        # Deep nonlinear representation
        # ---------------------------------------------------------

        self.residual_blocks = nn.Sequential(
            *[
                ResidualMLPBlock(
                    dim=hidden_dim,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )

        self.output_norm = nn.LayerNorm(hidden_dim)

        # ---------------------------------------------------------
        # Quantile parameterization
        #
        # raw[0] = base / lower quantile
        # raw[1] = gap between P10 and P50
        # raw[2] = gap between P50 and P90
        # ---------------------------------------------------------

        self.quantile_projection = nn.Linear(
            hidden_dim,
            3,
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Xavier initialization for linear layers.
        """

        for module in self.modules():

            if isinstance(module, nn.Linear):

                nn.init.xavier_uniform_(
                    module.weight
                )

                if module.bias is not None:
                    nn.init.zeros_(
                        module.bias
                    )

    def forward(
        self,
        representations: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        representations:
            [B, D]
            OR
            [B, T, D]

        Returns
        -------
        Tensor:
            [B, 3]
            OR
            [B, T, 3]
        """

        if not torch.is_tensor(representations):
            raise TypeError(
                "representations must be a torch.Tensor"
            )

        # ---------------------------------------------------------
        # Accept both:
        #
        # [B, D]
        # [B, T, D]
        # ---------------------------------------------------------

        if representations.ndim not in (2, 3):
            raise ValueError(
                "Expected input shape "
                "[B, D] or [B, T, D]. "
                f"Got {tuple(representations.shape)}"
            )

        if representations.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input dimension "
                f"{self.input_dim}, but got "
                f"{representations.shape[-1]}"
            )

        # Never silently train on invalid numerical values.
        if not torch.isfinite(
            representations
        ).all():

            raise ValueError(
                "Input contains NaN or infinite values."
            )

        # Save batch/time dimensions.
        leading_shape = representations.shape[:-1]

        # Flatten:
        #
        # [B, D]
        #     ->
        # [B, D]
        #
        # [B, T, D]
        #     ->
        # [B*T, D]
        #
        x = representations.reshape(
            -1,
            self.input_dim,
        )

        # ---------------------------------------------------------
        # Shared feature encoder
        # ---------------------------------------------------------

        x = self.input_projection(x)

        # ---------------------------------------------------------
        # Residual nonlinear blocks
        # ---------------------------------------------------------

        x = self.residual_blocks(x)

        x = self.output_norm(x)

        # ---------------------------------------------------------
        # Raw quantile parameters
        # ---------------------------------------------------------

        raw = self.quantile_projection(x)

        base = raw[:, 0:1]

        gap_1 = F.softplus(
            raw[:, 1:2]
        )

        gap_2 = F.softplus(
            raw[:, 2:3]
        )

        # ---------------------------------------------------------
        # Non-crossing quantiles
        # ---------------------------------------------------------

        if self.positive_output:

            # Streamflow is non-negative in physical units.
            p10 = F.softplus(base)

        else:

            # Useful when predicting in a transformed space
            # such as log1p(streamflow).
            p10 = base

        p50 = p10 + gap_1

        p90 = p50 + gap_2

        output = torch.cat(
            [
                p10,
                p50,
                p90,
            ],
            dim=-1,
        )

        # Restore original dimensions.
        #
        # [B, D] -> [B, 3]
        #
        # [B, T, D] -> [B, T, 3]
        output = output.reshape(
            *leading_shape,
            3,
        )

        return output

    def predict(
        self,
        representations: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Convenience prediction method.
        """

        output = self.forward(
            representations
        )

        return {
            "p10": output[..., 0],
            "p50": output[..., 1],
            "p90": output[..., 2],
        }

    @property
    def quantile_levels(self) -> tuple[float, float, float]:
        """
        Return quantile levels.
        """

        return self.quantiles