"""
Unit tests for Regional Temporal Convolutional Network (RegionalTCN).

Verifies:
- Input shape contracts and dimensional validation.
- Output shapes for point prediction [B, 1] and embedding mode [B, 128].
- Deterministic forward pass in eval mode.
- Static-feature sensitivity (different static features yield distinct predictions).
- Integration with load_static_attributes().
"""

import unittest
import torch
import numpy as np

from src.models.regional_tcn import RegionalTCN
from src.data.loaders import load_static_attributes, load_split_catchment_ids


class TestRegionalTCN(unittest.TestCase):
    """Test suite for RegionalTCN architecture."""

    def setUp(self):
        torch.manual_seed(42)
        self.model = RegionalTCN()
        self.model.eval()

    def test_forward_pass_point_mode(self):
        """Verify dynamic [B, 64, 20] + static [B, 129] in point mode yields [B, 1]."""
        batch_size = 8
        seq_len = 64
        x_dyn = torch.randn(batch_size, seq_len, 20)
        x_stat = torch.randn(batch_size, 129)

        with torch.no_grad():
            out = self.model(x_dyn, x_stat, mode="point")

        self.assertEqual(out.shape, (batch_size, 1))
        self.assertFalse(torch.isnan(out).any(), "Point output contains NaN values.")
        self.assertFalse(torch.isinf(out).any(), "Point output contains Inf values.")

    def test_forward_pass_embedding_mode(self):
        """Verify dynamic [B, 64, 20] + static [B, 129] in embedding mode yields [B, 128]."""
        batch_size = 16
        seq_len = 64
        x_dyn = torch.randn(batch_size, seq_len, 20)
        x_stat = torch.randn(batch_size, 129)

        with torch.no_grad():
            emb = self.model(x_dyn, x_stat, mode="embedding")

        self.assertEqual(emb.shape, (batch_size, 128))
        self.assertFalse(torch.isnan(emb).any(), "Embedding output contains NaN values.")
        self.assertFalse(torch.isinf(emb).any(), "Embedding output contains Inf values.")

    def test_various_batch_sizes(self):
        """Verify model handles diverse batch sizes (1, 4, 16, 32, 64)."""
        seq_len = 64
        for b in [1, 4, 16, 32, 64]:
            x_dyn = torch.randn(b, seq_len, 20)
            x_stat = torch.randn(b, 129)

            with torch.no_grad():
                out_point = self.model(x_dyn, x_stat, mode="point")
                out_emb = self.model(x_dyn, x_stat, mode="embedding")

            self.assertEqual(out_point.shape, (b, 1), f"Failed point shape for batch {b}")
            self.assertEqual(out_emb.shape, (b, 128), f"Failed embedding shape for batch {b}")

    def test_deterministic_forward_pass_in_eval_mode(self):
        """Verify identical inputs produce bitwise identical outputs in eval mode."""
        self.model.eval()
        x_dyn = torch.randn(4, 64, 20)
        x_stat = torch.randn(4, 129)

        with torch.no_grad():
            out1 = self.model(x_dyn, x_stat, mode="point")
            out2 = self.model(x_dyn, x_stat, mode="point")
            emb1 = self.model(x_dyn, x_stat, mode="embedding")
            emb2 = self.model(x_dyn, x_stat, mode="embedding")

        self.assertTrue(torch.equal(out1, out2), "Point outputs differ across identical eval passes.")
        self.assertTrue(torch.equal(emb1, emb2), "Embeddings differ across identical eval passes.")

    def test_static_feature_sensitivity(self):
        """
        Verify that holding meteorological forcings identical while changing
        catchment static features produces distinct predictions and embeddings.
        """
        self.model.eval()
        # Identical dynamic meteorological forcing sequence for two different catchments
        x_dyn = torch.randn(1, 64, 20)
        x_dyn_batch = x_dyn.repeat(2, 1, 1) # [2, 64, 20]

        # Two distinctly different static catchment attributes
        x_stat_1 = torch.ones(1, 129) * 2.0
        x_stat_2 = torch.ones(1, 129) * -2.0
        x_stat_batch = torch.cat([x_stat_1, x_stat_2], dim=0) # [2, 129]

        with torch.no_grad():
            point_preds = self.model(x_dyn_batch, x_stat_batch, mode="point")
            embeddings = self.model(x_dyn_batch, x_stat_batch, mode="embedding")

        # Predictions for basin 1 and basin 2 must differ
        pred_diff = torch.abs(point_preds[0] - point_preds[1]).item()
        emb_diff = torch.norm(embeddings[0] - embeddings[1]).item()

        self.assertGreater(pred_diff, 1e-4, "Model output insensitive to static catchment features.")
        self.assertGreater(emb_diff, 1e-3, "Model embedding insensitive to static catchment features.")

    def test_input_validation(self):
        """Verify input validation catches shape, dimension, and mode errors."""
        # 1. Dynamic tensor not 3D
        with self.assertRaises(ValueError):
            self.model(torch.randn(8, 20), torch.randn(8, 129))

        # 2. Static tensor not 2D
        with self.assertRaises(ValueError):
            self.model(torch.randn(8, 64, 20), torch.randn(8, 64, 129))

        # 3. Batch size mismatch
        with self.assertRaises(ValueError):
            self.model(torch.randn(8, 64, 20), torch.randn(4, 129))

        # 4. Wrong dynamic feature dimension (!= 20)
        with self.assertRaises(ValueError):
            self.model(torch.randn(8, 64, 18), torch.randn(8, 129))

        # 5. Wrong static feature dimension (!= 129)
        with self.assertRaises(ValueError):
            self.model(torch.randn(8, 64, 20), torch.randn(8, 100))

        # 6. Invalid mode
        with self.assertRaises(ValueError):
            self.model(torch.randn(8, 64, 20), torch.randn(8, 129), mode="invalid_mode")

    def test_receptive_field_property(self):
        """Verify theoretical receptive field property matches underlying TCN."""
        self.assertEqual(self.model.receptive_field, self.model.temporal_encoder.receptive_field)
        self.assertEqual(self.model.receptive_field, 61)

    def test_integration_with_load_static_attributes(self):
        """Verify forward pass with actual scaled attributes loaded from data pipeline."""
        splits = load_split_catchment_ids()
        sample_gids = splits["train"][:4]

        static_df = load_static_attributes(catchment_ids=sample_gids)
        static_tensor = torch.tensor(static_df.to_numpy(), dtype=torch.float32)

        x_dyn = torch.randn(4, 64, 20)
        with torch.no_grad():
            preds = self.model(x_dyn, static_tensor, mode="point")
            embs = self.model(x_dyn, static_tensor, mode="embedding")

        self.assertEqual(preds.shape, (4, 1))
        self.assertEqual(embs.shape, (4, 128))
        self.assertFalse(torch.isnan(preds).any())
        self.assertFalse(torch.isnan(embs).any())


if __name__ == "__main__":
    unittest.main()
