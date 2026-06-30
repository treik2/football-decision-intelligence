from typing import List

# Markets that share the same underlying event — combining them amplifies one assumption
CORRELATED_GROUPS = [
    {"home_win", "ah_home", "home_win_and_over"},
    {"away_win", "ah_away"},
    {"over_2_5", "over_3_5", "over_4_5", "btts"},
    {"under_2_5", "btts_no", "under_1_5"},
    {"draw", "under_2_5", "btts_no"},
]

# Hard-blocked combinations (extremely high correlation)
BLOCKED_PAIRS = [
    {"over_2_5", "over_1_5"},  # over_1_5 subsumes over_2_5
    {"home_win", "draw"},       # mutually exclusive but correlated reasoning
    {"over_3_5", "btts"},       # very high co-occurrence
    {"under_2_5", "over_2_5"}, # contradictory
    {"btts", "btts_no"},
]


def correlation_score(markets: List[str]) -> float:
    """
    Returns an estimated correlation score [0, 1] for a set of markets.
    Higher = more correlated = less diversified.
    """
    market_set = set(markets)
    max_overlap = 0.0
    for group in CORRELATED_GROUPS:
        overlap = len(market_set & group) / len(group)
        if overlap > max_overlap:
            max_overlap = overlap
    return max_overlap


def is_correlated(markets: List[str], threshold: float = 0.65) -> bool:
    market_set = set(markets)
    # Block pairs first
    for pair in BLOCKED_PAIRS:
        if pair.issubset(market_set):
            return True
    return correlation_score(markets) >= threshold
