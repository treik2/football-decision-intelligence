def kelly_fraction(model_prob: float, odds: float, fraction: float = 0.25) -> float:
    """
    Fractional Kelly criterion.
    fraction=0.25 = quarter-Kelly (conservative, recommended).
    Hard cap at 10% of bankroll per bet.
    """
    b = odds - 1.0
    p = model_prob
    q = 1.0 - p
    if b <= 0 or p <= 0:
        return 0.0
    k = (p * b - q) / b
    if k <= 0:
        return 0.0
    return min(k * fraction, 0.10)


def stake_amount(model_prob: float, odds: float, bankroll: float, fraction: float = 0.25) -> float:
    return round(bankroll * kelly_fraction(model_prob, odds, fraction), 2)
