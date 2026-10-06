"""
Regional Temporal Convolutional Network (Regional TCN).

Conditions temporal dilated convolutions with static physical catchment descriptors
(topography, geology, soil, land cover, climate) via FiLM (Feature-wise Linear Modulation)
or latent static concatenation. This allows knowledge transfer to ungauged basins.

RESPONSIBILITY: Member 2 (TCN / Regional TCN)
STATUS: PLANNED / UNDER DEVELOPMENT (Step 6)
"""

from typing import Dict, Any, Optional


class RegionalTCN:
    """
    Catchment-conditioned TCN architecture for ungauged streamflow forecasting.

    Inputs:
        - Dynamic meteorological sequences: shape [Batch, SeqLen, InFeatures_dyn]
        - Static catchment attributes: shape [Batch, InFeatures_static]

    Output:
        - Catchment-conditioned temporal embeddings: shape [Batch, HiddenDim]
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        # TODO (Member 2): Build static attribute encoder and conditioning mechanism (Step 6)

    def forward(self, x_dynamic, x_static):
        raise NotImplementedError(
            "Regional TCN conditioning architecture is scheduled for Step 6 by Member 2."
        )
