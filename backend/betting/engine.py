import uuid
import numpy as np
from typing import Dict, List
from backend.betting.ev import expected_value, is_value, implied_prob, fair_odds
from backend.betting.kelly import kelly_fraction, stake_amount
from backend.simulation.bet_builder import build_combinations


class BettingEngine:
    def build_suggestions(
        self,
        match_id: str,
        win_probs: Dict[str, float],
        sim_result: Dict,
        odds: Dict[str, float],
        bankroll: float = 1000.0,
        min_ev: float = 0.02,
        kelly_fraction_val: float = 0.25,
    ) -> List[Dict]:
        all_probs = {
            "home_win": win_probs.get("home", 0),
            "draw": win_probs.get("draw", 0),
            "away_win": win_probs.get("away", 0),
            "over_1_5": sim_result.get("over_1_5", 0),
            "over_2_5": sim_result.get("over_2_5", 0),
            "over_3_5": sim_result.get("over_3_5", 0),
            "btts": sim_result.get("btts", 0),
            "btts_no": sim_result.get("btts_no", 0),
            "home_clean_sheet": sim_result.get("home_clean_sheet", 0),
            "away_clean_sheet": sim_result.get("away_clean_sheet", 0),
            "over_8_5_corners": sim_result.get("over_8_5_corners", 0),
            "over_9_5_corners": sim_result.get("over_9_5_corners", 0),
            "over_3_5_cards": sim_result.get("over_3_5_cards", 0),
        }

        candidate_legs = []
        for market, model_p in all_probs.items():
            odd = odds.get(market)
            if odd is None or odd <= 1.0:
                continue
            if not is_value(model_p, odd, min_edge=min_ev):
                continue
            ev = expected_value(model_p, odd)
            imp = implied_prob(odd)
            conf = min(0.95, max(0.20, model_p * 0.9 + 0.05))
            candidate_legs.append({
                "name": market,
                "probability": round(model_p, 4),
                "implied_prob": round(imp, 4),
                "edge": round(model_p - imp, 4),
                "ev": round(ev, 4),
                "odds": odd,
                "fair_odds": fair_odds(model_p),
                "confidence": round(conf, 3),
                "kelly_fraction": round(kelly_fraction(model_p, odd, kelly_fraction_val), 4),
                "stake": stake_amount(model_p, odd, bankroll, kelly_fraction_val),
            })

        suggestions = []

        # Single-leg bets
        for leg in sorted(candidate_legs, key=lambda x: -x["ev"]):
            suggestions.append({
                "bet_id": str(uuid.uuid4()),
                "match_id": match_id,
                "type": "single",
                "legs": [leg],
                "combined_prob": leg["probability"],
                "combined_ev": leg["ev"],
                "combined_odds": leg["odds"],
                "kelly_fraction": leg["kelly_fraction"],
                "stake_advice": leg["stake"],
                "notes": f"Edge: {leg['edge']*100:.1f}% over market",
            })

        # Multi-leg accumulators
        combos = build_combinations(candidate_legs, max_legs=4, max_corr=0.65, min_confidence=0.40)
        for combo in combos:
            combo_prob = float(np.prod([l["probability"] for l in combo]))
            combo_odds = float(np.prod([l["odds"] for l in combo]))
            ev = expected_value(combo_prob, combo_odds)
            if ev < min_ev:
                continue
            kf = kelly_fraction(combo_prob, combo_odds, kelly_fraction_val)
            suggestions.append({
                "bet_id": str(uuid.uuid4()),
                "match_id": match_id,
                "type": "acca",
                "legs": combo,
                "combined_prob": round(combo_prob, 5),
                "combined_ev": round(ev, 4),
                "combined_odds": round(combo_odds, 3),
                "kelly_fraction": round(kf, 4),
                "stake_advice": round(bankroll * kf, 2),
                "notes": f"{len(combo)}-leg accumulator",
            })

        return sorted(suggestions, key=lambda x: -x["combined_ev"])[:20]
