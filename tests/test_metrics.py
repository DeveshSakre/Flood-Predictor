"""
Unit tests for hydrologic metrics (NSE, KGE) and quantile uncertainty loss.
Compatible with standard library unittest and pytest.
"""

import unittest
import math
import numpy as np

from src.evaluation.metrics import (
    nash_sutcliffe_efficiency,
    kling_gupta_efficiency,
    compute_all_metrics,
)
from src.uncertainty.quantile_loss import pinball_loss
from src.uncertainty.calibration import (
    prediction_interval_coverage_probability,
    mean_prediction_interval_width,
)


class TestMetrics(unittest.TestCase):
    def test_perfect_prediction_metrics(self):
        """Verify that perfect forecasts yield NSE = 1.0 and KGE = 1.0."""
        obs = np.array([10.0, 25.0, 50.0, 100.0, 30.0, 15.0])
        sim = np.array([10.0, 25.0, 50.0, 100.0, 30.0, 15.0])

        nse = nash_sutcliffe_efficiency(obs, sim)
        kge = kling_gupta_efficiency(obs, sim)

        self.assertTrue(math.isclose(nse, 1.0, abs_tol=1e-5))
        self.assertTrue(math.isclose(kge, 1.0, abs_tol=1e-5))

    def test_mean_prediction_nse(self):
        """Verify that predicting the mean observed flow yields NSE = 0.0."""
        obs = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        sim = np.full_like(obs, np.mean(obs))

        nse = nash_sutcliffe_efficiency(obs, sim)
        self.assertTrue(math.isclose(nse, 0.0, abs_tol=1e-5))

    def test_pinball_loss_properties(self):
        """Verify that pinball loss is non-negative and properly penalizes under/over prediction."""
        y_true = np.array([100.0])
        y_pred_under = np.array([80.0])   # Residual = +20
        y_pred_over = np.array([120.0])   # Residual = -20

        # At tau = 0.9, underprediction is penalized by 0.9 * 20 = 18.0
        loss_under = pinball_loss(y_true, y_pred_under, tau=0.9)
        self.assertTrue(math.isclose(loss_under, 18.0, abs_tol=1e-5))

        # At tau = 0.9, overprediction is penalized by (1 - 0.9) * 20 = 2.0
        loss_over = pinball_loss(y_true, y_pred_over, tau=0.9)
        self.assertTrue(math.isclose(loss_over, 2.0, abs_tol=1e-5))

    def test_picp_and_mpiw(self):
        """Verify PICP coverage and interval width calculations."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        lower = np.array([5.0, 15.0, 25.0, 35.0, 60.0])   # 4 covered, 1 out of bounds
        upper = np.array([15.0, 25.0, 35.0, 45.0, 70.0])

        picp = prediction_interval_coverage_probability(y_true, lower, upper)
        self.assertTrue(math.isclose(picp, 0.80, abs_tol=1e-5))

        mpiw = mean_prediction_interval_width(lower, upper)
        self.assertTrue(math.isclose(mpiw, 10.0, abs_tol=1e-5))


if __name__ == "__main__":
    unittest.main()
