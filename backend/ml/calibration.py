import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from typing import List


class PlattCalibrator:
    def __init__(self):
        self.lr = LogisticRegression()

    def fit(self, raw_probs: List[float], labels: List[int]):
        self.lr.fit(np.array(raw_probs).reshape(-1, 1), labels)

    def calibrate(self, p: float) -> float:
        return float(self.lr.predict_proba([[p]])[0][1])


class IsotonicCalibrator:
    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds="clip")

    def fit(self, raw_probs: List[float], labels: List[int]):
        self.iso.fit(raw_probs, labels)

    def calibrate(self, p: float) -> float:
        return float(self.iso.predict([p])[0])


def brier_score(probs: List[float], outcomes: List[int]) -> float:
    return float(np.mean((np.array(probs) - np.array(outcomes)) ** 2))


def expected_calibration_error(probs: List[float], outcomes: List[int], n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece, n = 0.0, len(probs)
    p, o = np.array(probs), np.array(outcomes)
    for i in range(n_bins):
        mask = (p >= bins[i]) & (p < bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(p[mask].mean() - o[mask].mean())
    return ece
