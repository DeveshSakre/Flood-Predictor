"""
Model definitions and baseline architectures.

Planned Architectures:
- XGBoostBaseline: Tabular gradient-boosted benchmark (Step 4, Member 5)
- LSTMModel: Recurrent temporal baseline (Step 5, Member 2)
- TCNModel: Temporal Convolutional Network with dilated causal convolutions (Step 5, Member 2)
- RegionalTCN: TCN conditioned on static physical catchment embeddings (Step 6, Member 2)
- GATModel: Graph Attention Network on river routing DAG (Step 7, Member 3)
- RegionalTCN_GAT: Full spatiotemporal coupled architecture (Step 8, Members 2 & 3)
"""

from src.models.xgboost_baseline import XGBoostBaseline

__all__ = [
    "XGBoostBaseline",
]
