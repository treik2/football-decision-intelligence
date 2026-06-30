import itertools
from typing import List, Dict, Tuple
import numpy as np

# Market correlation matrix — pairs that are highly correlated
HIGH_CORR_PAIRS = [
    ("over_2_5", "over_3_5"),
    ("over_3_5", "over_4_5"),
    ("home_win", "away_clean_sheet"),
    ("away_win", "home_clean_sheet"),
    ("btts", "over_2_5"),  # moderate, allow but flag
]


def correlation_score(leg_a: str, leg_b: str) -> float:
    """Return estimated correlation coefficient between two market legs."""
    corr_map = {
        ("over_2_5", "over_3_5"): 0.82,
        ("over_3_5", "over_4_5"): 0.79,
        ("home_win", "away_clean_sheet"): 0.71,
        ("away_win", "home_clean_sheet"): 0.71,
        ("btts", "over_2_5"): 0.55,
        ("btts", "over_1_5"): 0.48,
        ("home_win", "over_2_5"): 0.12,
        ("draw", "btts"): 0.30,
    }
    key = tuple(sorted([leg_a, leg_b]))
    return corr_map.get(key, 0.05)


def build_combinations(
    legs: List[Dict],
    max_legs: int = 4,
    max_corr: float = 0.65,
    min_confidence: float = 0.40,
) -> List[List[Dict]]:
    """
    Generate all valid bet-builder combinations from candidate legs.
    Rejects combinations with high inter-leg correlation or low confidence.
    """
    valid_combos = []
    for size in range(2, max_legs + 1):
        for combo in itertools.combinations(legs, size):
            combo = list(combo)
            # Check confidence threshold
            if any(l["confidence"] < min_confidence for l in combo):
                continue
            # Check pairwise correlation
            names = [l["name"] for l in combo]
            bad = False
            for a, b in itertools.combinations(names, 2):
                if correlation_score(a, b) > max_corr:
                    bad = True
                    break
            if bad:
                continue
            valid_combos.append(combo)
    return valid_combos
