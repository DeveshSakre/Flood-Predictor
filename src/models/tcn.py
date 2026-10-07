"""
Temporal Convolutional Network (TCN) Architecture for Hydrologic Modeling.

Implements dilated causal 1D convolutions with residual blocks.
Captures catchment hydrologic memory (e.g. multi-week recession, soil saturation dynamics)
without leaking future temporal information.

RESPONSIBILITY: Member 2 (Temporal Modeling / TCN)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn

from src.utils.config import load_config


def _apply_weight_norm(module: nn.Module) -> nn.Module:
    """
    Apply weight normalization compatible with modern and legacy PyTorch versions.
    Prefers torch.nn.utils.parametrizations.weight_norm over deprecated torch.nn.utils.weight_norm.
    """
    try:
        from torch.nn.utils.parametrizations import weight_norm
        return weight_norm(module)
    except Exception:
        try:
            from torch.nn.utils import weight_norm
            return weight_norm(module)
        except Exception:
            return module


class Chomp1d(nn.Module):
    """
    Trims trailing padding from 1D convolution outputs to guarantee strict temporal causality.
    Removes the last `chomp_size` elements along the time dimension.
    """

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size <= 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    Dilated causal residual block for sequence modeling.

    Structure:
        x -> Conv1d(dilated) -> Chomp1d -> ReLU -> Dropout
          -> Conv1d(dilated) -> Chomp1d -> ReLU -> Dropout -> (+) -> ReLU
        |                                                      ^
        +-----> [Optional Conv1d(1x1) downsample] -------------+

    Args:
        in_channels: Number of input feature channels.
        out_channels: Number of output feature channels.
        kernel_size: Size of the 1D convolution kernel.
        stride: Stride for convolution (default: 1).
        dilation: Dilation factor for expanding receptive field.
        dropout: Dropout probability.
        use_weight_norm: Whether to apply weight normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        dropout: float = 0.2,
        use_weight_norm: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        padding = (kernel_size - 1) * dilation

        # First dilated causal convolution
        conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.conv1 = _apply_weight_norm(conv1) if use_weight_norm else conv1
        self.chomp1 = Chomp1d(padding)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Second dilated causal convolution
        conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.conv2 = _apply_weight_norm(conv2) if use_weight_norm else conv2
        self.chomp2 = Chomp1d(padding)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.act1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.act2,
            self.dropout2,
        )

        # Residual connection: 1x1 conv if channel dimensions differ, else identity
        if in_channels != out_channels:
            res_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
            self.downsample = _apply_weight_norm(res_conv) if use_weight_norm else res_conv
        else:
            self.downsample = None

        self.final_act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through causal temporal block.

        Args:
            x: Input tensor of shape [Batch, in_channels, SeqLen].

        Returns:
            Output tensor of shape [Batch, out_channels, SeqLen].
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.final_act(out + res)


class TCNModel(nn.Module):
    """
    Standard Temporal Convolutional Network (TCN) for hydrologic sequence modeling.

    Maps dynamic meteorological/soil forcings [Batch, SeqLen, InFeatures] to:
      - Point prediction: [Batch, 1] (mode='point')
      - Catchment temporal embedding: [Batch, HiddenDim] (mode='embedding')

    Configuration is loaded by default from `configs/tcn_config.yaml`.
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
        self.tcn_channels: List[int] = list(arch.get("tcn_channels", [64, 64, 128, 128]))
        self.kernel_size: int = int(arch.get("kernel_size", 3))
        self.dropout: float = float(arch.get("dropout", 0.2))
        self.dense_hidden_dim: int = int(arch.get("dense_hidden_dim", 128))
        self.use_weight_norm: bool = bool(arch.get("use_weight_norm", True))

        # Dilation schedule
        config_dilations = arch.get("dilations", None)
        num_levels = len(self.tcn_channels)
        if config_dilations is not None and len(config_dilations) > 0:
            if len(config_dilations) >= num_levels:
                self.dilations = [int(config_dilations[i]) for i in range(num_levels)]
            else:
                self.dilations = [int(d) for d in config_dilations] + [
                    2**i for i in range(len(config_dilations), num_levels)
                ]
        else:
            self.dilations = [2**i for i in range(num_levels)]

        # Construct causal dilated residual block hierarchy
        layers: List[nn.Module] = []
        for i in range(num_levels):
            in_ch = self.input_dim if i == 0 else self.tcn_channels[i - 1]
            out_ch = self.tcn_channels[i]
            dilation = self.dilations[i]

            layers.append(
                TemporalBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=self.kernel_size,
                    stride=1,
                    dilation=dilation,
                    dropout=self.dropout,
                    use_weight_norm=self.use_weight_norm,
                )
            )

        self.network = nn.Sequential(*layers)

        # Embedding projection head: projects final representation to dense_hidden_dim
        self.embedding_proj = nn.Linear(self.tcn_channels[-1], self.dense_hidden_dim)
        self.embedding_act = nn.ReLU()

        # Point regression head: predicts next-day streamflow scalar [Batch, 1]
        self.point_head = nn.Linear(self.dense_hidden_dim, 1)

    @property
    def receptive_field(self) -> int:
        """
        Calculate total theoretical receptive field in days/steps:
        RF = 1 + sum_l (2 * (kernel_size - 1) * dilation_l)
        """
        rf = 1
        for d in self.dilations:
            rf += 2 * (self.kernel_size - 1) * d
        return rf

    def forward(self, x_dynamic: torch.Tensor, mode: str = "point") -> torch.Tensor:
        """
        Forward pass through causal TCN.

        Args:
            x_dynamic: Input dynamic tensor of shape [Batch, SeqLen, InFeatures=20].
            mode: Output mode, either:
                  - 'point': returns next-day streamflow forecast [Batch, 1]
                  - 'embedding': returns latent temporal representation [Batch, dense_hidden_dim]

        Returns:
            Tensor of shape [Batch, 1] or [Batch, dense_hidden_dim].
        """
        if x_dynamic.dim() != 3:
            raise ValueError(
                f"Expected 3D input [Batch, SeqLen, InFeatures], got shape {x_dynamic.shape}"
            )

        if x_dynamic.size(2) != self.input_dim:
            raise ValueError(
                f"Expected feature dim {self.input_dim}, got {x_dynamic.size(2)} in shape {x_dynamic.shape}"
            )

        # Transpose from [Batch, SeqLen, InFeatures] to [Batch, InFeatures, SeqLen] for 1D convolution
        x = x_dynamic.transpose(1, 2)

        # Pass through causal dilated convolutions -> [Batch, OutChannels, SeqLen]
        h_seq = self.network(x)

        # Extract representation at the final timestep T (causally summarizes 1..T)
        h_last = h_seq[:, :, -1]

        # Project to dense embedding space -> [Batch, dense_hidden_dim]
        embedding = self.embedding_act(self.embedding_proj(h_last))

        if mode == "embedding":
            return embedding
        elif mode == "point":
            return self.point_head(embedding)
        else:
            raise ValueError(
                f"Unsupported mode: '{mode}'. Must be either 'point' or 'embedding'."
            )
