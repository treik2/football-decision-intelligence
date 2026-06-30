import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from typing import List


class IsotonicCalibrator:
    """Isotonic regression calibrator for post-hoc probability calibration."""

    def __init__(self):
        self._iso = IsotonicRegression(out_of_bounds="clip")
        self._fitted = False

    def fit(self, probs: List[float], outcomes: List[int]):
        self._iso.fit(probs, outcomes)
        self._fitted = True

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return probs
        return self._iso.predict(probs)

    def brier_score(self, probs: np.ndarray, outcomes: np.ndarray) -> float:
        return float(np.mean((probs - outcomes) ** 2))

    def calibration_error(self, probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> float:
        fraction_pos, mean_pred = calibration_curve(outcomes, probs, n_bins=n_bins, strategy="uniform")
        return float(np.mean(np.abs(fraction_pos - mean_pred)))
