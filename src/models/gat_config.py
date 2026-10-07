from dataclasses import dataclass

@dataclass
class GATConfig:
    """
    Configuration for the River-Network GAT Spatial Encoder.
    
    TCN Integration Contract:
    - Node features: [N, node_dim] (default 6)
    - Edge index: [2, E]
    - Edge attributes: [E, edge_dim] (default 4)
    - Output embedding: [N, spatial_embedding_dim]
    """
    node_dim: int = 6
    edge_dim: int = 4
    spatial_embedding_dim: int = 64
    
    hidden_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.2
    
    # Model architecture choices
    add_self_loops: bool = False  # GATv2Conv adds self loops natively by default, but we should make sure we document this.
    fill_value: float = 0.0       # Fill value for edge attributes on self-loops
    
    def __post_init__(self):
        # We ensure output dimension matches spatial_embedding_dim
        pass
