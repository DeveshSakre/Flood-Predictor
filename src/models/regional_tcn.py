"""
Regional Temporal Convolutional Network (Regional TCN) Architecture.

Conditions temporal dilated convolutions with static physical catchment descriptors
(topography, geology, soil, land cover, climate) via a static embedding MLP and
multimodal feature fusion. This enables generalized hydrological modeling and
knowledge transfer to ungauged river basins.

RESPONSIBILITY: Member 2 (Temporal Modeling / Regional TCN)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn

from src.utils.config import load_config
from src.models.tcn import TCNModel


class RegionalTCN(nn.Module):
    """
    Catchment-conditioned TCN architecture for ungauged streamflow forecasting.

    Inputs:
        - Dynamic meteorological sequences: shape [Batch, SeqLen, InFeatures_dyn=20]
        - Static catchment attributes: shape [Batch, InFeatures_static=129]

    Outputs:
        - Point prediction: shape [Batch, 1] (mode='point')
        - Catchment-conditioned temporal embedding: shape [Batch, HiddenDim=128] (mode='embedding')

    Architecture:
        1. Temporal Encoder: Causal dilated TCN -> [Batch, dense_hidden_dim=128]
        2. Static Encoder MLP: 2-layer MLP -> [Batch, dense_hidden_dim=128]
        3. Fusion MLP: Concatenation [Batch, 256] -> [Batch, dense_hidden_dim=128]
        4. Prediction Head: MLP [Batch, 128] -> [Batch, 1]
    """

    def __init__(self, config: Optional[Union[Dict[str, Any], str, Path]] = None):
        super().__init__()

        # Resolve configuration dictionary
        if config is None:
            raw_cfg = load_config("configs/tcn_config.yaml")
        elif isinstance(config, (str, Path)):
            raw_cfg = load_config(str(config))
        elif isinstance(config, dict):
            raw_cfg = config
        else:
            raise TypeError(f"Config must be dict, str, Path, or None; got {type(config)}")

        arch = raw_cfg.get("architecture", raw_cfg)

        self.input_dim: int = int(arch.get("input_dim", 20))
        self.static_dim: int = int(arch.get("static_dim", 129))
        self.dense_hidden_dim: int = int(arch.get("dense_hidden_dim", 128))
        self.dropout: float = float(arch.get("dropout", 0.2))

        # 1. Base Causal TCN Temporal Encoder (reuses TCNModel design)
        self.temporal_encoder = TCNModel(config=raw_cfg)

        # 2. Static Attribute Encoder MLP: maps [Batch, 129] -> [Batch, 128]
        self.static_encoder = nn.Sequential(
            nn.Linear(self.static_dim, self.dense_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.dense_hidden_dim, self.dense_hidden_dim),
            nn.ReLU(),
        )

        # 3. Multimodal Fusion MLP: concatenates temporal [128] + static [128] -> [128]
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.dense_hidden_dim * 2, self.dense_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )

        # 4. Prediction Head: maps fused representation [Batch, 128] -> [Batch, 1]
        self.point_head = nn.Sequential(
            nn.Linear(self.dense_hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 1),
        )

    @property
    def receptive_field(self) -> int:
        """Calculate theoretical receptive field in days from underlying TCN."""
        return self.temporal_encoder.receptive_field

    def forward(
        self,
        x_dynamic: torch.Tensor,
        x_static: torch.Tensor,
        mode: str = "point",
    ) -> torch.Tensor:
        """
        Forward pass through RegionalTCN.

        Args:
            x_dynamic: Meteorological sequence tensor of shape [Batch, SeqLen, InFeatures_dyn].
            x_static: Static physiographic attributes tensor of shape [Batch, InFeatures_static].
            mode: Output mode:
                  - 'point': returns next-day streamflow forecast scalar [Batch, 1]
                  - 'embedding': returns catchment-conditioned representation [Batch, dense_hidden_dim]

        Returns:
            Tensor of shape [Batch, 1] or [Batch, dense_hidden_dim].
        """
        if x_dynamic.dim() != 3:
            raise ValueError(
                f"Expected 3D dynamic input [Batch, SeqLen, InFeatures], got shape {x_dynamic.shape}"
            )

        if x_static.dim() != 2:
            raise ValueError(
                f"Expected 2D static input [Batch, InFeatures_static], got shape {x_static.shape}"
            )

        if x_dynamic.size(0) != x_static.size(0):
            raise ValueError(
                f"Batch dimension mismatch: x_dynamic has {x_dynamic.size(0)}, "
                f"x_static has {x_static.size(0)}"
            )

        if x_dynamic.size(2) != self.input_dim:
            raise ValueError(
                f"Expected dynamic feature dim {self.input_dim}, got {x_dynamic.size(2)}"
            )

        if x_static.size(1) != self.static_dim:
            raise ValueError(
                f"Expected static feature dim {self.static_dim}, got {x_static.size(1)}"
            )

        # 1. Temporal sequence encoding: [Batch, dense_hidden_dim=128]
        h_temporal = self.temporal_encoder(x_dynamic, mode="embedding")

        # 2. Static physiography encoding: [Batch, dense_hidden_dim=128]
        h_static = self.static_encoder(x_static)

        # 3. Multimodal fusion: [Batch, 256] -> [Batch, 128]
        h_combined = torch.cat([h_temporal, h_static], dim=-1)
        fused_embedding = self.fusion_mlp(h_combined)

        if mode == "embedding":
            return fused_embedding
        elif mode == "point":
            return self.point_head(fused_embedding)
        else:
            raise ValueError(
                f"Unsupported mode: '{mode}'. Must be either 'point' or 'embedding'."
            )
