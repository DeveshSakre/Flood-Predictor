"""
Quantile regression and uncertainty quantification module.

RESPONSIBILITY: Member 3 (Quantile Regression & Uncertainty Estimation)
"""

from src.uncertainty.quantile_regression import QuantileHead, ResidualMLPBlock
from src.uncertainty.quantile_loss import (
    MultiQuantilePinballLoss,
    pinball_loss,
    pinball_loss_multi,
)
from src.uncertainty.calibration import (
    prediction_interval_coverage_probability,
    mean_prediction_interval_width,
    check_crossing_violations,
    quantile_coverage,
    quantile_calibration_error,
    winkler_score,
    evaluate_uncertainty_forecasts,
)

__all__ = [
    "QuantileHead",
    "ResidualMLPBlock",
    "MultiQuantilePinballLoss",
    "pinball_loss",
    "pinball_loss_multi",
    "prediction_interval_coverage_probability",
    "mean_prediction_interval_width",
    "check_crossing_violations",
    "quantile_coverage",
    "quantile_calibration_error",
    "winkler_score",
    "evaluate_uncertainty_forecasts",
]
