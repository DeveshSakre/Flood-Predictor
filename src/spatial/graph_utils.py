"""
Utilities for loading, inspecting, and manipulating directed river-network graphs.
Supports train, validation, and test hydrological topologies.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd
import networkx as nx

from src.utils.config import get_repo_root


def load_partition_graph(
    split: str,
    processed_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Load nodes, edges, and adjacency matrix for a specific partition ('train', 'val', or 'test').

    Args:
        split: Partition identifier ('train', 'val', 'test').
        processed_dir: Path to directory containing processed artifacts.

    Returns:
        Dictionary containing:
            - 'nodes_df': DataFrame with node attributes
            - 'edges_df': DataFrame with edge attributes and downstream topology
            - 'adj_matrix': Numpy array representing adjacency
    """
    if processed_dir is None:
        graph_dir = get_repo_root() / "data" / "processed" / "graph"
    else:
        graph_dir = Path(processed_dir) / "graph"

    nodes_file = graph_dir / f"nodes_{split}.csv"
    edges_file = graph_dir / f"edges_{split}.csv"
    adj_file = graph_dir / f"adj_matrix_{split}.npz"

    if not nodes_file.exists():
        raise FileNotFoundError(f"Nodes file not found: {nodes_file}")
    if not edges_file.exists():
        raise FileNotFoundError(f"Edges file not found: {edges_file}")
    if not adj_file.exists():
        raise FileNotFoundError(f"Adjacency file not found: {adj_file}")

    nodes_df = pd.read_csv(nodes_file, dtype={"gauge_id": str})
    nodes_df["gauge_id"] = nodes_df["gauge_id"].str.zfill(5)

    edges_df = pd.read_csv(edges_file)
    if len(edges_df) > 0:
        # Support either source_gauge / target_gauge or u / v
        if "source_gauge" in edges_df.columns and "target_gauge" in edges_df.columns:
            edges_df["u"] = edges_df["source_gauge"].astype(str).str.zfill(5)
            edges_df["v"] = edges_df["target_gauge"].astype(str).str.zfill(5)
        elif "u" in edges_df.columns and "v" in edges_df.columns:
            edges_df["u"] = edges_df["u"].astype(str).str.zfill(5)
            edges_df["v"] = edges_df["v"].astype(str).str.zfill(5)
    else:
        edges_df["u"] = pd.Series(dtype=str)
        edges_df["v"] = pd.Series(dtype=str)

    npz_data = np.load(adj_file)
    adj_matrix = npz_data["adj_matrix"]

    return {
        "nodes_df": nodes_df,
        "edges_df": edges_df,
        "adj_matrix": adj_matrix,
    }


def build_networkx_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> nx.DiGraph:
    """
    Construct a NetworkX directed graph from nodes and edges DataFrames.

    Args:
        nodes_df: DataFrame with node attributes, must contain 'gauge_id'.
        edges_df: DataFrame with edge connections, columns 'u' and 'v'.

    Returns:
        Directed graph (nx.DiGraph) representing downstream river flow.
    """
    G = nx.DiGraph()

    # Add nodes with all associated features
    for _, row in nodes_df.iterrows():
        node_id = str(row["gauge_id"]).zfill(5)
        attrs = row.to_dict()
        G.add_node(node_id, **attrs)

    # Add directed edges: u (upstream) -> v (downstream)
    if len(edges_df) > 0 and "u" in edges_df.columns and "v" in edges_df.columns:
        for _, row in edges_df.iterrows():
            u = str(row["u"]).zfill(5)
            v = str(row["v"]).zfill(5)
            edge_attrs = row.to_dict()
            G.add_edge(u, v, **edge_attrs)

    return G


def is_directed_acyclic(G: nx.DiGraph) -> bool:
    """Check if river network graph is a strictly directed acyclic graph (DAG)."""
    return nx.is_directed_acyclic_graph(G)


def get_upstream_subgraph(G: nx.DiGraph, target_node: str) -> nx.DiGraph:
    """
    Extract the upstream catchment network draining into `target_node`.

    Args:
        G: Full river directed graph.
        target_node: Canonical 5-digit gauge identifier.

    Returns:
        Sub-DiGraph containing target node and all upstream ancestors.
    """
    target_node = str(target_node).zfill(5)
    ancestors = nx.ancestors(G, target_node)
    nodes = ancestors.union({target_node})
    return G.subgraph(nodes).copy()
