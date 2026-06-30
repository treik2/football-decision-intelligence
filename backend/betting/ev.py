def implied_prob(odds: float) -> float:
    if odds <= 1.0:
        return 1.0
    return 1.0 / odds


def expected_value(model_prob: float, odds: float) -> float:
    """EV = p*(odds-1) - (1-p). Positive = value over market."""
    return model_prob * (odds - 1) - (1 - model_prob)


def is_value(model_prob: float, odds: float, min_edge: float = 0.02) -> bool:
    return (model_prob - implied_prob(odds)) >= min_edge


def overround(odds_dict: dict) -> float:
    return sum(implied_prob(o) for o in odds_dict.values()) - 1.0


def fair_odds(model_prob: float) -> float:
    if model_prob <= 0:
        return 999.0
    return round(1.0 / model_prob, 3)
