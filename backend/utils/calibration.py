"""Calibration utilities: Platt scaling and isotonic regression."""
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from typing import Tuple, List


def calibration_plot_data(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    """Return data for a reliability diagram."""
    fraction_of_positives, mean_predicted = calibration_curve(y_true, y_prob, n_bins=n_bins)
    return {
        "fraction_of_positives": fraction_of_positives.tolist(),
        "mean_predicted": mean_predicted.tolist(),
    }


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score — lower is better, 0 is perfect."""
    return float(np.mean((y_prob - y_true) ** 2))


def isotonic_calibrate(probs: np.ndarray, labels: np.ndarray, new_probs: np.ndarray) -> np.ndarray:
    """Fit isotonic regression and transform new_probs."""
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(probs, labels)
    return ir.transform(new_probs)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE)."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += mask.sum() * abs(acc - conf)
    return float(ece / len(y_true))
