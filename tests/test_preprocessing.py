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


if __name__ == "__main__":
    unittest.main()
