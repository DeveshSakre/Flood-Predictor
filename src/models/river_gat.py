import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class RiverGAT(nn.Module):
    """
    River-Network GAT Spatial Encoder.
    
    This encoder processes the static graphical structure of river basins 
    using an Edge-Aware Graph Attention Network (GATv2).
    
    TCN Integration Contract:
    - Input node_features: [N, 6]
    - Input edge_index: [2, E]
    - Input edge_attr: [E, 4]
    - Output graph_embedding: [N, spatial_embedding_dim]
    
    Notes on Topological Integrity:
    - The river network directed graph semantics are preserved.
    - We use add_self_loops=True (default in GATv2Conv) to ensure nodes
      attend to themselves. We supply a fill_value=0.0 so self-loop edges
      receive a zero vector for edge features.
    """
    def __init__(self, config):
        super(RiverGAT, self).__init__()
        self.config = config
        
        # We project the 6-dim node features to hidden_dim immediately
        self.node_proj = nn.Linear(config.node_dim, config.hidden_dim)
        
        # GATv2 layers with edge dimensions
        self.convs = nn.ModuleList()
        # For the first layer, input is hidden_dim
        in_channels = config.hidden_dim
        for _ in range(config.num_layers - 1):
            self.convs.append(
                GATv2Conv(
                    in_channels=in_channels,
                    out_channels=config.hidden_dim, # per head
                    heads=config.num_heads,
                    dropout=config.dropout,
                    edge_dim=config.edge_dim,
                    add_self_loops=True,
                    fill_value=config.fill_value,
                    concat=True
                )
            )
            in_channels = config.hidden_dim * config.num_heads
            
        # Final layer projects to spatial_embedding_dim
        # We average multi-head attention outputs (concat=False)
        self.final_conv = GATv2Conv(
            in_channels=in_channels,
            out_channels=config.spatial_embedding_dim,
            heads=config.num_heads,
            dropout=config.dropout,
            edge_dim=config.edge_dim,
            add_self_loops=True,
            fill_value=config.fill_value,
            concat=False
        )

    def forward(self, x, edge_index, edge_attr, return_attention_weights=False):
        """
        Forward pass converting nodes & edges to spatial embeddings.
        """
        # Node projection
        x = self.node_proj(x)
        
        att_weights_list = []
        
        for i, conv in enumerate(self.convs):
            if return_attention_weights:
                x, alpha = conv(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
                att_weights_list.append(alpha)
            else:
                x = conv(x, edge_index, edge_attr=edge_attr)
                
            x = F.elu(x)
            x = F.dropout(x, p=self.config.dropout, training=self.training)
            
        # Final conv layer
        if return_attention_weights:
            x, alpha = self.final_conv(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
            att_weights_list.append(alpha)
        else:
            x = self.final_conv(x, edge_index, edge_attr=edge_attr)
            
        # Optional nonlinear formulation on output could be applied, 
        # but sticking to linear output embedding to allow TCN to handle it.
        
        if return_attention_weights:
            return x, att_weights_list
        return x

class BaselineMLP(nn.Module):
    """
    A simple non-graph node encoder baseline to compare against RiverGAT.
    Uses strictly the 6 node features without any topological routing.
    """
    def __init__(self, config):
        super(BaselineMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(config.node_dim, config.hidden_dim),
            nn.ELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim * config.num_heads),
            nn.ELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * config.num_heads, config.spatial_embedding_dim)
        )
        
    def forward(self, x, edge_index=None, edge_attr=None):
        return self.net(x)
