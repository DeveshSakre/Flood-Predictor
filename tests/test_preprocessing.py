"""
Unit tests for preprocessing, scalers, and parameter inversion.
Compatible with standard library unittest and pytest.
"""

import unittest
import numpy as np
import pandas as pd

from src.data.preprocessing import Preprocessor


class TestPreprocessing(unittest.TestCase):
    def test_preprocessor_loading(self):
        """Verify preprocessor loads all train-fitted artifacts correctly."""
        prep = Preprocessor()
        self.assertGreater(len(prep.dynamic_params), 0, "Dynamic parameters failed to load.")
        self.assertIsNotNone(prep.static_scaler, "Static scaler failed to load.")
        self.assertGreater(len(prep.static_imputation_values), 0, "Static imputation values failed to load.")
        self.assertIsNotNone(prep.graph_node_scaler, "Graph node scaler failed to load.")

    def test_dynamic_transformation_and_inversion(self):
        """Verify dynamic transformation and exact invertibility for precipitation."""
        prep = Preprocessor()
        test_col = "prcp(mm/day)"
        self.assertIn(test_col, prep.dynamic_params)

        mean = prep.dynamic_params[test_col]["mean"]
        std = prep.dynamic_params[test_col]["std"]

        df_test = pd.DataFrame({test_col: [mean, mean + std, mean - std]})
        df_norm = prep.transform_dynamic(df_test)

        # Expected normalized values: 0, 1, -1
        np.testing.assert_allclose(df_norm[test_col].to_numpy(), [0.0, 1.0, -1.0], atol=1e-5)

        # Invert back to physical units
        inverted = prep.inverse_transform_feature(df_norm[test_col], test_col)
        np.testing.assert_allclose(inverted, df_test[test_col].to_numpy(), atol=1e-5)

    def test_static_transformation_key_and_ordering(self):
        """Verify static transformation checks 'features' key and preserves exact 129-feature ordering."""
        prep = Preprocessor()
        expected_cols = prep.static_params.get("features", [])
        self.assertEqual(len(expected_cols), 129, f"Expected 129 static features, got {len(expected_cols)}")

        # Create dummy dataframe with shuffled feature columns
        shuffled_cols = list(reversed(expected_cols))
        dummy_data = np.zeros((3, 129))
        dummy_df = pd.DataFrame(dummy_data, columns=shuffled_cols)
        # Test imputation on a numeric feature present in imputation rules
        numeric_col = list(prep.static_imputation_values.keys())[0]
        dummy_df.loc[0, numeric_col] = np.nan

        transformed = prep.transform_static(dummy_df)

        # Verify exact 129 features
        self.assertEqual(transformed.shape[1], 129)
        # Verify deterministic feature ordering matches scaler metadata
        self.assertEqual(list(transformed.columns), expected_cols)
        # Verify no NaN or Inf
        self.assertFalse(transformed.isna().any().any(), "Transformed static features contain NaN.")
        self.assertFalse(np.isinf(transformed.to_numpy()).any(), "Transformed static features contain Inf.")

    def test_categorical_static_encoding(self):
        """Verify categorical static encoding produces all 13 expected indicator features."""
        prep = Preprocessor()
        cat_meta = prep.categorical_meta

        sample_df = pd.DataFrame({
            "dom_land_cover": ["crops", "rangeland"],
            "hsg_major": ["C", "D"],
            "geol_class_1st": ["Metamorphic Rocks", "Basic Volcanic Rocks"],
        })

        encoded = prep.encode_categorical_static(sample_df)

        # Check land cover indicators
        for c in cat_meta["dom_land_cover_categories"]:
            self.assertIn(f"dom_lc_{c}", encoded.columns)
        # Check HSG indicators
        for c in cat_meta["hsg_major_categories"]:
            self.assertIn(f"hsg_{c}", encoded.columns)
        # Check geology indicators
        for c in cat_meta["geol_class_1st_categories"]:
            self.assertIn(f"geol_{c}", encoded.columns)


if __name__ == "__main__":
    unittest.main()
