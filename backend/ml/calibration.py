"""Post-hoc calibration helpers — Platt scaling and isotonic regression."""
from __future__ import annotations
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from typing import List


def platt_calibrate(probs: List[float], outcomes: List[int]) -> List[float]:
    """Fit isotonic regression calibration on historical predictions vs outcomes."""
    iso = IsotonicRegression(out_of_bounds="clip")
    arr = np.array(probs).reshape(-1, 1)
    iso.fit(arr.ravel(), outcomes)
    return iso.predict(arr.ravel()).tolist()


def expected_calibration_error(probs: List[float], outcomes: List[int], n_bins: int = 10) -> float:
    """Compute ECE — lower is better."""
    probs_arr = np.array(probs)
    outcomes_arr = np.array(outcomes)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs_arr >= bins[i]) & (probs_arr < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = outcomes_arr[mask].mean()
        bin_conf = probs_arr[mask].mean()
        ece += mask.sum() * abs(bin_acc - bin_conf)
    return float(ece / len(probs_arr))
