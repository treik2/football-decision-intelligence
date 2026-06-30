import numpy as np
from typing import Dict
from backend.ml.goals import GoalsModel


class MonteCarloSimulator:
    """
    Vectorised Monte Carlo match simulator.
    Samples Poisson goals for each team across N simulations and
    derives market-relevant probabilities from the distribution.
    """

    def __init__(self, goals_model: GoalsModel):
        self.goals_model = goals_model

    def run(self, features: Dict, n: int = 100_000, seed: int = 42) -> Dict:
        rng = np.random.default_rng(seed)
        mu_home, mu_away = self.goals_model.expected_goals(features)

        home_goals = rng.poisson(mu_home, size=n)
        away_goals = rng.poisson(mu_away, size=n)
        total_goals = home_goals + away_goals

        outcome = np.sign(home_goals - away_goals)  # 1=home, 0=draw, -1=away

        home_win   = float(np.mean(outcome == 1))
        draw       = float(np.mean(outcome == 0))
        away_win   = float(np.mean(outcome == -1))
        over_0_5   = float(np.mean(total_goals > 0.5))
        over_1_5   = float(np.mean(total_goals > 1.5))
        over_2_5   = float(np.mean(total_goals > 2.5))
        over_3_5   = float(np.mean(total_goals > 3.5))
        over_4_5   = float(np.mean(total_goals > 4.5))
        btts       = float(np.mean((home_goals > 0) & (away_goals > 0)))
        btts_no    = 1.0 - btts
        clean_home = float(np.mean(away_goals == 0))
        clean_away = float(np.mean(home_goals == 0))

        # Correct score probabilities (top 12 by frequency)
        pairs = list(zip(home_goals.tolist(), away_goals.tolist()))
        from collections import Counter
        score_counts = Counter(pairs)
        top_scores = [
            {"score": f"{h}-{a}", "prob": round(cnt / n, 4)}
            for (h, a), cnt in score_counts.most_common(12)
        ]

        # Asian Handicap -0.5 / +0.5
        ah_home_minus_half = float(np.mean(home_goals > away_goals))  # home -0.5 wins
        ah_away_plus_half  = 1.0 - ah_home_minus_half

        # First half: approximate as ~45% of full time goal rate
        mu_h_ht = mu_home * 0.45
        mu_a_ht = mu_away * 0.45
        ht_home = rng.poisson(mu_h_ht, size=n)
        ht_away = rng.poisson(mu_a_ht, size=n)
        ht_over_0_5 = float(np.mean(ht_home + ht_away > 0.5))
        ht_over_1_5 = float(np.mean(ht_home + ht_away > 1.5))
        ht_home_win = float(np.mean(ht_home > ht_away))
        ht_draw     = float(np.mean(ht_home == ht_away))
        ht_away_win = float(np.mean(ht_home < ht_away))

        return {
            "n": n,
            "home_win":  round(home_win, 4),
            "draw":      round(draw, 4),
            "away_win":  round(away_win, 4),
            "over_0_5":  round(over_0_5, 4),
            "over_1_5":  round(over_1_5, 4),
            "over_2_5":  round(over_2_5, 4),
            "over_3_5":  round(over_3_5, 4),
            "over_4_5":  round(over_4_5, 4),
            "btts":      round(btts, 4),
            "btts_no":   round(btts_no, 4),
            "clean_sheet_home": round(clean_home, 4),
            "clean_sheet_away": round(clean_away, 4),
            "ah_home_minus_half": round(ah_home_minus_half, 4),
            "ah_away_plus_half":  round(ah_away_plus_half, 4),
            "ht_home_win": round(ht_home_win, 4),
            "ht_draw":     round(ht_draw, 4),
            "ht_away_win": round(ht_away_win, 4),
            "ht_over_0_5": round(ht_over_0_5, 4),
            "ht_over_1_5": round(ht_over_1_5, 4),
            "home_goals_mean": round(float(np.mean(home_goals)), 3),
            "away_goals_mean": round(float(np.mean(away_goals)), 3),
            "top_correct_scores": top_scores,
        }
