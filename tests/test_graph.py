"""
Unit tests for hydrologic river network graphs and DAG topology.
Compatible with standard library unittest and pytest.
"""

import unittest
import networkx as nx

from src.spatial.graph_utils import (
    load_partition_graph,
    build_networkx_graph,
    is_directed_acyclic,
)


class TestGraph(unittest.TestCase):
    def test_train_graph_structure(self):
        """Verify train river graph topology and connected edges."""
        data = load_partition_graph("train")
        nodes_df = data["nodes_df"]
        edges_df = data["edges_df"]
        adj = data["adj_matrix"]

        self.assertEqual(len(nodes_df), 169, f"Expected 169 train graph nodes, got {len(nodes_df)}")
        self.assertEqual(len(edges_df), 119, f"Expected 119 train edges, got {len(edges_df)}")
        self.assertEqual(adj.shape, (169, 169))

        G = build_networkx_graph(nodes_df, edges_df)
        self.assertEqual(G.number_of_nodes(), 169)
        self.assertEqual(G.number_of_edges(), 119)
        self.assertTrue(is_directed_acyclic(G), "Train river graph contains cyclic loops!")

    def test_validation_and_test_graphs(self):
        """Verify validation and test graphs have correct topologies and are DAGs."""
        expected = {
            "val": {"nodes": 36, "edges": 33},
            "test": {"nodes": 37, "edges": 34},
        }

        for split, exp in expected.items():
            data = load_partition_graph(split)
            nodes_df = data["nodes_df"]
            edges_df = data["edges_df"]
            adj = data["adj_matrix"]

            self.assertEqual(len(nodes_df), exp["nodes"])
            self.assertEqual(len(edges_df), exp["edges"])
            self.assertEqual(adj.shape, (exp["nodes"], exp["nodes"]))

            G = build_networkx_graph(nodes_df, edges_df)
            self.assertEqual(G.number_of_nodes(), exp["nodes"])
            self.assertEqual(G.number_of_edges(), exp["edges"])
            self.assertTrue(is_directed_acyclic(G), f"{split} river graph contains cyclic loops!")


if __name__ == "__main__":
    unittest.main()
