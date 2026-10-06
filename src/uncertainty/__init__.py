"""
Quantile regression and uncertainty quantification module.
"""

from src.uncertainty.quantile_loss import pinball_loss, pinball_loss_multi
from src.uncertainty.calibration import (
    prediction_interval_coverage_probability,
    mean_prediction_interval_width,
    check_crossing_violations,
)

__all__ = [
    "pinball_loss",
    "pinball_loss_multi",
    "prediction_interval_coverage_probability",
    "mean_prediction_interval_width",
    "check_crossing_violations",
]
