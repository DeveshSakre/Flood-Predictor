"""
Model definitions and baseline architectures.

Architectures:
- XGBoostBaseline: Tabular gradient-boosted benchmark (Step 4, Member 5)
- TCNModel: Temporal Convolutional Network with dilated causal convolutions (Step 5, Member 2)
- TemporalBlock: Causal dilated residual block (Step 5, Member 2)
- LSTMModel: Recurrent temporal baseline (Step 5, Member 2)
- RegionalTCN: TCN conditioned on static physical catchment embeddings (Step 6, Member 2)
- GATModel: Graph Attention Network on river routing DAG (Step 7, Member 3)
- RegionalTCN_GAT: Full spatiotemporal coupled architecture (Step 8, Members 2 & 3)
"""

from src.models.xgboost_baseline import XGBoostBaseline
from src.models.tcn import TCNModel, TemporalBlock

__all__ = [
    "XGBoostBaseline",
    "TCNModel",
    "TemporalBlock",
]
