"""
Unit tests for data loaders, canonical IDs, and partition integrity.
Compatible with standard library unittest and pytest.
"""

import unittest
from pathlib import Path
import pandas as pd

from src.data.loaders import (
    load_split_catchment_ids,
    load_canonical_id_mapping,
    load_feature_catalog,
)
from src.utils.config import get_repo_root


class TestDataLoading(unittest.TestCase):
    def test_split_counts_and_disjointness(self):
        """Verify that catchment splits have exact expected counts and zero overlap."""
        splits = load_split_catchment_ids()

        train_ids = set(splits["train"])
        val_ids = set(splits["val"])
        test_ids = set(splits["test"])
        demo_ids = set(splits["ungauged_demo"])

        # Check exact counts established in Step 2
        self.assertEqual(len(train_ids), 169, f"Expected 169 train catchments, got {len(train_ids)}")
        self.assertEqual(len(val_ids), 36, f"Expected 36 val catchments, got {len(val_ids)}")
        self.assertEqual(len(test_ids), 37, f"Expected 37 test catchments, got {len(test_ids)}")
        self.assertEqual(len(demo_ids), 5, f"Expected 5 ungauged demo catchments, got {len(demo_ids)}")

        # Check pairwise disjointness (anti-leakage)
        self.assertTrue(train_ids.isdisjoint(val_ids), "Train and validation partitions overlap!")
        self.assertTrue(train_ids.isdisjoint(test_ids), "Train and test partitions overlap!")
        self.assertTrue(val_ids.isdisjoint(test_ids), "Validation and test partitions overlap!")
        self.assertTrue(train_ids.isdisjoint(demo_ids), "Train and ungauged demo partitions overlap!")

    def test_canonical_id_mapping(self):
        """Verify canonical ID mapping table."""
        mapping_df = load_canonical_id_mapping()
        self.assertEqual(len(mapping_df), 472, f"Expected 472 catchments in ID mapping, got {len(mapping_df)}")
        self.assertIn("gauge_id", mapping_df.columns)
        self.assertTrue(all(len(gid) == 5 for gid in mapping_df["gauge_id"]))

    def test_feature_catalog(self):
        """Verify feature catalog integrity."""
        catalog = load_feature_catalog()
        self.assertEqual(len(catalog), 159, f"Expected 159 total features, got {len(catalog)}")

        feat_types = catalog["feature_type"].astype(str).str.upper()
        dyn_count = int((feat_types == "DYNAMIC").sum())
        static_count = int((feat_types == "STATIC").sum())
        graph_count = int(feat_types.str.startswith("GRAPH").sum())

        self.assertEqual(dyn_count, 20, f"Expected 20 dynamic features, got {dyn_count}")
        self.assertEqual(static_count, 129, f"Expected 129 static features, got {static_count}")
        self.assertEqual(graph_count, 10, f"Expected 10 graph features, got {graph_count}")


if __name__ == "__main__":
    unittest.main()
