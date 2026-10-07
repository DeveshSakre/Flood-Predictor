"""
Unit tests for data loaders, canonical IDs, and partition integrity.
Compatible with standard library unittest and pytest.
"""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from src.data.loaders import (
    load_split_catchment_ids,
    load_canonical_id_mapping,
    load_feature_catalog,
    load_static_attributes,
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

    def test_load_static_attributes_shape_and_ordering(self):
        """Verify static loader returns exactly 129 features in deterministic order without NaN/Inf."""
        static_df = load_static_attributes()
        self.assertEqual(static_df.shape, (472, 129), f"Expected shape (472, 129), got {static_df.shape}")
        self.assertEqual(static_df.index.name, "gauge_id")
        self.assertTrue(all(len(gid) == 5 for gid in static_df.index), "Non-5-digit gauge_id detected in index.")

        # Verify exact column order against catalog
        catalog = load_feature_catalog()
        expected_cols = catalog[catalog["feature_type"] == "STATIC"]["feature_name"].tolist()
        self.assertEqual(list(static_df.columns), expected_cols, "Feature ordering does not match catalog.")

        # Verify numerical sanity (no NaN or Inf)
        self.assertFalse(static_df.isna().any().any(), "Static attributes contain NaN.")
        self.assertFalse(np.isinf(static_df.to_numpy()).any(), "Static attributes contain Inf.")

    def test_load_static_attributes_partitions(self):
        """Verify train, validation, and test partitions can be loaded and transformed with zero leakage."""
        splits = load_split_catchment_ids()

        # Train partition
        train_static = load_static_attributes(catchment_ids=splits["train"])
        self.assertEqual(train_static.shape, (169, 129))
        self.assertEqual(train_static.index.tolist(), splits["train"])
        # Verify train normalization properties (mean ~ 0, std ~ 1)
        np.testing.assert_allclose(np.mean(train_static.to_numpy(), axis=0), 0.0, atol=1e-7)
        np.testing.assert_allclose(np.std(train_static.to_numpy(), axis=0), 1.0, atol=1e-7)

        # Validation partition
        val_static = load_static_attributes(catchment_ids=splits["val"])
        self.assertEqual(val_static.shape, (36, 129))
        self.assertFalse(val_static.isna().any().any(), "Validation static features contain NaN.")
        self.assertFalse(np.isinf(val_static.to_numpy()).any(), "Validation static features contain Inf.")

        # Test partition
        test_static = load_static_attributes(catchment_ids=splits["test"])
        self.assertEqual(test_static.shape, (37, 129))
        self.assertFalse(test_static.isna().any().any(), "Test static features contain NaN.")
        self.assertFalse(np.isinf(test_static.to_numpy()).any(), "Test static features contain Inf.")

    def test_load_static_attributes_anti_leakage(self):
        """Verify that no streamflow signatures or forbidden leakage variables exist in static features."""
        static_df = load_static_attributes()
        cols = set(static_df.columns)

        forbidden_leakage_vars = [
            "reservoir_index",
            "flow_availability",
            "DIS_AV_CMS",
            "ORD_FLOW",
            "lstm_pred_streamflow",
        ]
        for f in forbidden_leakage_vars:
            self.assertNotIn(f, cols, f"Forbidden leakage feature '{f}' found in static attributes!")

        # Verify no CAMELS-IND hydro signatures
        streamflow_signatures = [
            "q_mean", "runoff_ratio", "slope_fdc", "baseflow_index", "bfi",
            "q_10", "q_50", "q_90", "hfd_mean", "low_q_freq", "high_q_freq"
        ]
        for sig in streamflow_signatures:
            self.assertNotIn(sig, cols, f"Streamflow signature '{sig}' found in static attributes!")


if __name__ == "__main__":
    unittest.main()
