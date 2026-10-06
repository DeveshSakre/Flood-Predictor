"""
Temporal Convolutional Network (TCN) Architecture for Hydrologic Modeling.

Implements causal, dilated 1D convolutions with residual blocks.
Captures long-term catchment hydrologic memory (e.g. 180-day baseflow and snowmelt recession).

RESPONSIBILITY: Member 2 (TCN / Regional TCN)
STATUS: PLANNED / UNDER DEVELOPMENT (Step 5)
"""

from typing import Dict, Any, Optional, List


class TemporalBlock:
    """
    Dilated causal convolution block with weight normalization and residual connection.

    TODO (Member 2):
    Implement in PyTorch nn.Module during Step 5.
    """
    pass


class TCNModel:
    """
    Standard Temporal Convolutional Network baseline.
    Maps dynamic input sequences [Batch, SeqLen, InFeatures] to temporal catchment representations.

    RESPONSIBILITY: Member 2 (TCN / Regional TCN)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        # TODO (Member 2): Initialize convolutional layer hierarchy

    def forward(self, x):
        raise NotImplementedError(
            "TCN architecture implementation is scheduled for Step 5 by Member 2."
        )
