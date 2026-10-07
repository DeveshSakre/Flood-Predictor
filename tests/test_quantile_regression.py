"""
Comprehensive unit tests for Quantile Regression & Uncertainty Quantification (Member 3).

Verifies:
- QuantileHead architecture: shapes, 2D/3D inputs, varied representations (Option A, Option B).
- Non-crossing mathematical guarantees across random, extreme, and edge-case inputs.
- Non-negativity constraints (positive_output=True) and standardized mode (positive_output=False).
- Differentiable MultiQuantilePinballLoss with masking and gradient flow.
- Evaluation metrics: pinball loss, PICP, MPIW, Winkler score, calibration diagnostics.
"""

import unittest
import torch
import numpy as np

from src.uncertainty.quantile_regression import QuantileHead, ResidualMLPBlock
from src.uncertainty.quantile_loss import (
    MultiQuantilePinballLoss,
    pinball_loss,
    pinball_loss_multi,
)
from src.uncertainty.calibration import (
    prediction_interval_coverage_probability,
    mean_prediction_interval_width,
    quantile_coverage,
    quantile_calibration_error,
    check_crossing_violations,
    winkler_score,
    evaluate_uncertainty_forecasts,
)


class TestQuantileHead(unittest.TestCase):
    """Test suite for neural QuantileHead module."""

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)

    def test_forward_2d_shape(self):
        """Verify [B, D] representations produce [B, 3] quantiles."""
        batch_size = 16
        input_dim = 128  # RegionalTCN representation dim
        model = QuantileHead(input_dim=input_dim, hidden_dim=64, num_blocks=2)
        model.eval()

        x = torch.randn(batch_size, input_dim)
        with torch.no_grad():
            out = model(x)

        self.assertEqual(out.shape, (batch_size, 3))
        self.assertTrue(torch.isfinite(out).all())

    def test_forward_3d_shape(self):
        """Verify [B, T, D] temporal sequence representations produce [B, T, 3] quantiles."""
        batch_size = 8
        seq_len = 32
        input_dim = 20  # CAMELS-IND raw dynamic features (Option A)
        model = QuantileHead(input_dim=input_dim, hidden_dim=64, num_blocks=2)
        model.eval()

        x = torch.randn(batch_size, seq_len, input_dim)
        with torch.no_grad():
            out = model(x)

        self.assertEqual(out.shape, (batch_size, seq_len, 3))
        self.assertTrue(torch.isfinite(out).all())

    def test_representation_agnostic_interfaces(self):
        """
        Verify representation-agnostic property:
        - Option A: CAMELS-IND features (e.g. 20 dynamic, 149 dynamic+static)
        - Option B: Regional TCN (128) + GAT (64) = 192 coupled representation
        """
        dims = [20, 129, 149, 128, 64, 192]
        for d in dims:
            head = QuantileHead(input_dim=d, hidden_dim=64, num_blocks=1)
            head.eval()
            rep = torch.randn(4, d)
            with torch.no_grad():
                out = head(rep)
            self.assertEqual(out.shape, (4, 3))

    def test_strict_non_crossing_guarantee(self):
        """Verify mathematical guarantee P10 <= P50 <= P90 under random and extreme inputs."""
        model = QuantileHead(input_dim=64, hidden_dim=128, num_blocks=3, positive_output=True)
        model.eval()

        # Random inputs
        x_rand = torch.randn(100, 64)
        # Extreme inputs
        x_extreme = torch.cat([
            torch.ones(10, 64) * 100.0,
            torch.ones(10, 64) * -100.0,
            torch.zeros(10, 64),
        ], dim=0)

        for x in [x_rand, x_extreme]:
            with torch.no_grad():
                out = model(x)
                p10 = out[:, 0]
                p50 = out[:, 1]
                p90 = out[:, 2]

                # Check monotonicity
                self.assertTrue(torch.all(p10 <= p50).item(), "P10 > P50 violation detected!")
                self.assertTrue(torch.all(p50 <= p90).item(), "P50 > P90 violation detected!")
                # Check non-negativity under positive_output=True
                self.assertTrue(torch.all(p10 >= 0.0).item(), "Negative streamflow detected!")

    def test_predict_dictionary_interface(self):
        """Verify convenience predict method returns named quantiles."""
        model = QuantileHead(input_dim=32, hidden_dim=64, num_blocks=1)
        model.eval()
        x = torch.randn(5, 32)
        with torch.no_grad():
            preds = model.predict(x)

        self.assertIn("p10", preds)
        self.assertIn("p50", preds)
        self.assertIn("p90", preds)
        self.assertEqual(preds["p10"].shape, (5,))

    def test_gradient_flow_backward(self):
        """Verify loss backpropagation produces non-zero gradients for all parameters."""
        model = QuantileHead(input_dim=16, hidden_dim=32, num_blocks=2, positive_output=False)
        model.train()
        criterion = MultiQuantilePinballLoss()

        x = torch.randn(8, 16)
        y = torch.randn(8, 1)

        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Gradient missing for {name}")
                self.assertTrue(torch.isfinite(param.grad).all(), f"Non-finite grad in {name}")


class TestPinballLoss(unittest.TestCase):
    """Test suite for differentiable and numpy pinball loss implementations."""

    def test_differentiable_multi_quantile_loss(self):
        """Verify PyTorch MultiQuantilePinballLoss computation."""
        criterion = MultiQuantilePinballLoss(quantiles=(0.10, 0.50, 0.90))

        y_true = torch.tensor([[100.0]], dtype=torch.float32)
        # Predictions: underpredict by 20 on all quantiles
        y_pred = torch.tensor([[80.0, 80.0, 80.0]], dtype=torch.float32)

        # Residual = +20
        # Loss at 0.10: 0.10 * 20 = 2.0
        # Loss at 0.50: 0.50 * 20 = 10.0
        # Loss at 0.90: 0.90 * 20 = 18.0
        # Mean = (2.0 + 10.0 + 18.0) / 3 = 10.0
        loss = criterion(y_pred, y_true)
        self.assertAlmostEqual(loss.item(), 10.0, places=4)

    def test_masking_functionality(self):
        """Verify observation mask zeroes out missing target timesteps."""
        criterion = MultiQuantilePinballLoss()

        y_true = torch.tensor([10.0, 20.0, 999.0], dtype=torch.float32)
        y_pred = torch.tensor([
            [10.0, 10.0, 10.0],
            [20.0, 20.0, 20.0],
            [0.0, 0.0, 0.0],  # Outlier prediction for missing observation
        ], dtype=torch.float32)
        mask = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32)

        loss_masked = criterion(y_pred, y_true, mask=mask)
        # For first two samples, y_true == y_pred so pinball loss is 0.0
        self.assertAlmostEqual(loss_masked.item(), 0.0, places=4)

    def test_numpy_pinball_functions(self):
        """Verify scalar numpy pinball loss and multi-quantile composite loss."""
        y_true = np.array([50.0, 100.0])
        y_pred = np.array([40.0, 110.0])  # Residuals: +10, -10

        # At tau = 0.5 (MAE / 2):
        # Sample 1: 0.5 * 10 = 5.0
        # Sample 2: (1 - 0.5) * 10 = 5.0
        # Mean = 5.0
        loss_50 = pinball_loss(y_true, y_pred, tau=0.50)
        self.assertAlmostEqual(loss_50, 5.0, places=5)

        preds_dict = {
            0.10: np.array([40.0, 90.0]),
            0.50: np.array([50.0, 100.0]),
            0.90: np.array([60.0, 110.0]),
        }
        loss_composite = pinball_loss_multi(y_true, preds_dict)
        self.assertGreater(loss_composite, 0.0)


class TestUncertaintyCalibration(unittest.TestCase):
    """Test suite for calibration and interval diagnostics."""

    def test_picp_and_mpiw(self):
        """Verify prediction interval coverage and width."""
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        low = np.array([5.0, 15.0, 25.0, 35.0, 55.0])   # 4 covered, 1 missed (50.0 < 55.0)
        high = np.array([15.0, 25.0, 35.0, 45.0, 65.0])

        picp = prediction_interval_coverage_probability(y, low, high)
        self.assertAlmostEqual(picp, 0.80, places=4)

        mpiw = mean_prediction_interval_width(low, high)
        self.assertAlmostEqual(mpiw, 10.0, places=4)

    def test_quantile_coverage_and_cal_error(self):
        """Verify empirical coverage and calibration error."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        # P50 predicts 5.0 -> values <= 5.0 are [1, 2, 3, 4, 5] -> 5/10 = 50% coverage
        p50 = np.full_like(y, 5.0)
        cov = quantile_coverage(y, p50)
        self.assertAlmostEqual(cov, 0.50, places=4)

        err = quantile_calibration_error(y, p50, nominal_quantile=0.50)
        self.assertAlmostEqual(err, 0.0, places=4)

    def test_winkler_score(self):
        """Verify Winkler score for well-calibrated and poorly calibrated intervals."""
        y = np.array([20.0])
        # In-bounds: L=10, U=30 -> width = 20, penalty = 0
        score_in = winkler_score(y, np.array([10.0]), np.array([30.0]), alpha=0.20)
        self.assertAlmostEqual(score_in, 20.0, places=4)

        # Out-of-bounds (underprediction): L=25, U=35 (y=20 < L=25)
        # width = 10, penalty = (2 / 0.2) * (25 - 20) = 10 * 5 = 50
        # Total = 60.0
        score_out = winkler_score(y, np.array([25.0]), np.array([35.0]), alpha=0.20)
        self.assertAlmostEqual(score_out, 60.0, places=4)

    def test_crossing_violations_check(self):
        """Verify check_crossing_violations catches monotonic violations."""
        low = np.array([10.0, 25.0])
        med = np.array([15.0, 20.0])  # Sample 2: low (25) > med (20)
        high = np.array([20.0, 30.0])

        violations = check_crossing_violations(low, med, high)
        self.assertEqual(violations["lower_gt_median_pct"], 50.0)
        self.assertEqual(violations["any_crossing_pct"], 50.0)

    def test_evaluate_uncertainty_forecasts(self):
        """Verify full forecast evaluation report structure."""
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        p10 = np.array([5.0, 15.0, 25.0, 35.0, 45.0])
        p50 = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        p90 = np.array([15.0, 25.0, 35.0, 45.0, 55.0])

        report = evaluate_uncertainty_forecasts(y, p10, p50, p90)
        self.assertIn("picp_80", report)
        self.assertIn("mpiw", report)
        self.assertIn("ace_80", report)
        self.assertIn("winkler_score_80", report)
        self.assertIn("composite_pinball_loss", report)
        self.assertEqual(report["crossing_violations_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
