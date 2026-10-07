"""
Unit tests for Temporal Convolutional Network (TCN) implementation.
Verifies causality, shape contracts, batch scaling, and numerical stability.
Compatible with standard library unittest and pytest.
"""

import unittest
import torch
import numpy as np

from src.models.tcn import TCNModel, TemporalBlock


class TestTCN(unittest.TestCase):
    """Test suite for base TCN model and TemporalBlock."""

    def setUp(self):
        torch.manual_seed(42)
        self.model = TCNModel()
        self.model.eval()

    def test_forward_pass_point_mode(self):
        """Verify forward pass with [batch, 64, 20] in point mode yields [batch, 1]."""
        batch_size = 8
        seq_len = 64
        in_features = 20

        x = torch.randn(batch_size, seq_len, in_features)
        with torch.no_grad():
            out = self.model(x, mode="point")

        self.assertEqual(out.shape, (batch_size, 1))
        self.assertFalse(torch.isnan(out).any(), "Point output contains NaN values.")
        self.assertFalse(torch.isinf(out).any(), "Point output contains Inf values.")

    def test_forward_pass_embedding_mode(self):
        """Verify forward pass in embedding mode yields [batch, hidden_dim]."""
        batch_size = 16
        seq_len = 64
        in_features = 20
        expected_hidden_dim = self.model.dense_hidden_dim

        x = torch.randn(batch_size, seq_len, in_features)
        with torch.no_grad():
            emb = self.model(x, mode="embedding")

        self.assertEqual(emb.shape, (batch_size, expected_hidden_dim))
        self.assertFalse(torch.isnan(emb).any(), "Embedding output contains NaN values.")
        self.assertFalse(torch.isinf(emb).any(), "Embedding output contains Inf values.")

    def test_different_batch_sizes(self):
        """Verify model processes various batch sizes (1, 4, 16, 32, 64)."""
        seq_len = 64
        in_features = 20

        for b in [1, 4, 16, 32, 64]:
            x = torch.randn(b, seq_len, in_features)
            with torch.no_grad():
                out_point = self.model(x, mode="point")
                out_emb = self.model(x, mode="embedding")

            self.assertEqual(out_point.shape, (b, 1), f"Failed for batch size {b} in point mode")
            self.assertEqual(
                out_emb.shape, (b, self.model.dense_hidden_dim), f"Failed for batch size {b} in embedding mode"
            )

    def test_causal_preserves_sequence_length(self):
        """Verify TemporalBlock maintains exact sequence length through dilated convolutions."""
        in_channels = 20
        out_channels = 64
        kernel_size = 3
        seq_len = 64

        for dilation in [1, 2, 4, 8, 16]:
            block = TemporalBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation,
            )
            block.eval()

            x = torch.randn(2, in_channels, seq_len)
            with torch.no_grad():
                out = block(x)

            self.assertEqual(
                out.shape,
                (2, out_channels, seq_len),
                f"TemporalBlock altered sequence length at dilation {dilation}",
            )

    def test_strict_temporal_causality(self):
        """
        Verify strict causality: perturbing inputs at future timesteps t' > t
        must NOT affect the block output at timestep t.
        """
        in_channels = 20
        out_channels = 64
        seq_len = 64
        cutoff_step = 40

        block = TemporalBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            dilation=4,
        )
        block.eval()

        x1 = torch.randn(2, in_channels, seq_len)
        x2 = x1.clone()

        # Perturb only future timesteps strictly after cutoff_step
        x2[:, :, cutoff_step:] = torch.randn(2, in_channels, seq_len - cutoff_step)

        with torch.no_grad():
            y1 = block(x1)
            y2 = block(x2)

        # Output up to cutoff_step must be identical
        diff_past = (y1[:, :, :cutoff_step] - y2[:, :, :cutoff_step]).abs().max().item()
        self.assertAlmostEqual(
            diff_past,
            0.0,
            places=5,
            msg=f"Temporal leakage detected! Diff before cutoff: {diff_past}",
        )

        # Output after cutoff must differ due to the perturbation
        diff_future = (y1[:, :, cutoff_step:] - y2[:, :, cutoff_step:]).abs().max().item()
        self.assertGreater(diff_future, 0.0, "Future perturbation had no effect.")

    def test_no_nan_or_inf(self):
        """Verify forward pass does not produce NaN or Inf under standard random inputs."""
        x = torch.randn(32, 64, 20)
        with torch.no_grad():
            out_point = self.model(x, mode="point")
            out_emb = self.model(x, mode="embedding")

        self.assertTrue(torch.isfinite(out_point).all(), "Non-finite values found in point predictions.")
        self.assertTrue(torch.isfinite(out_emb).all(), "Non-finite values found in embeddings.")

    def test_receptive_field_calculation(self):
        """Verify receptive field computation matches theoretical formula."""
        # For kernel_size=3 and dilations=[1, 2, 4, 8]:
        # RF = 1 + sum(2 * (3-1) * d) = 1 + 4*(1+2+4+8) = 61
        rf = self.model.receptive_field
        self.assertGreater(rf, 0)
        self.assertEqual(rf, 61)

    def test_invalid_input_dimensions(self):
        """Verify invalid input shapes raise informative ValueError."""
        # 2D input instead of 3D
        with self.assertRaises(ValueError):
            self.model(torch.randn(32, 64))

        # Wrong feature dimension (10 instead of 20)
        with self.assertRaises(ValueError):
            self.model(torch.randn(32, 64, 10))

        # Unsupported mode
        with self.assertRaises(ValueError):
            self.model(torch.randn(32, 64, 20), mode="unsupported_mode")


if __name__ == "__main__":
    unittest.main()
