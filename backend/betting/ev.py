def expected_value(model_prob: float, odds: float) -> float:
    """
    EV = p * (odds - 1) - (1 - p)
    Returns edge as a decimal (positive = value bet)
    """
    return model_prob * (odds - 1.0) - (1.0 - model_prob)


def fractional_kelly(model_prob: float, odds: float, fraction: float = 0.25) -> float:
    """
    Full Kelly = (bp - q) / b   where b = odds - 1, p = model prob, q = 1 - p
    Returns fraction * Kelly, clamped to [0, 1].
    """
    b = odds - 1.0
    q = 1.0 - model_prob
    if b <= 0:
        return 0.0
    k = (model_prob * b - q) / b
    return float(max(0.0, min(k * fraction, 1.0)))


def implied_prob(odds: float) -> float:
    return 1.0 / odds


def overround(odds_list: list) -> float:
    return sum(implied_prob(o) for o in odds_list) - 1.0


def fair_odds(model_prob: float) -> float:
    if model_prob <= 0:
        return 999.0
    return round(1.0 / model_prob, 3)
