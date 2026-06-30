from typing import Dict


PROMPT_TEMPLATE = """\
You are a professional football analyst and betting strategist.
Given the data below, produce:
1. A 3-4 sentence plain-language explanation of the key factors driving the model recommendation.
2. A bullet-point list of the top 3 reasons supporting this estimate.
3. A risk note explaining confidence level and any reasons to be cautious.

Do NOT recommend placing any specific bet. Your role is to explain the data, not to advise gambling.

--- MATCH ---
Home: {home_team}
Away: {away_team}

--- FEATURES ---
Elo difference (home − away): {elo_diff:+.1f}
xG difference (home − away): {xg_diff:+.2f}
Home expected goals: {home_xg:.2f}
Away expected goals: {away_xg:.2f}
Home form (last 5, 0-1): {home_form:.2f}
Away form (last 5, 0-1): {away_form:.2f}
Home rest days: {home_rest}
Away rest days: {away_rest}
Home injury score: {home_inj:.2f}
Away injury score: {away_inj:.2f}
Weather: {temp}°C, rain {rain}mm

--- MODEL PROBABILITIES ---
Home Win: {p_home:.1%}
Draw:     {p_draw:.1%}
Away Win: {p_away:.1%}
Over 2.5: {over25:.1%}
BTTS:     {btts:.1%}

--- MARKET ODDS PROVIDED ---
{market_odds_str}

--- MARKET IMPLIED PROBABILITIES ---
{market_impl_str}

Explain concisely and factually.
"""


def build_prompt(
    home_team: str,
    away_team: str,
    features: Dict,
    win_probs: Dict,
    sim_result: Dict,
    market_odds: Dict[str, float],
) -> str:
    market_odds_str = "\n".join(f"  {k}: {v}" for k, v in market_odds.items())
    market_impl_str = "\n".join(
        f"  {k}: {1/v:.1%}" for k, v in market_odds.items() if v > 0
    )
    return PROMPT_TEMPLATE.format(
        home_team=home_team,
        away_team=away_team,
        elo_diff=features.get("elo_diff", 0.0),
        xg_diff=features.get("xg_diff", 0.0),
        home_xg=features.get("home_xg", 1.3),
        away_xg=features.get("away_xg", 1.0),
        home_form=features.get("home_form", 0.5),
        away_form=features.get("away_form", 0.5),
        home_rest=int(features.get("home_rest_days", 7)),
        away_rest=int(features.get("away_rest_days", 7)),
        home_inj=features.get("home_injury_score", 0.0),
        away_inj=features.get("away_injury_score", 0.0),
        temp=features.get("temperature", 15),
        rain=features.get("rain_mm", 0),
        p_home=win_probs.get("home", 0.0),
        p_draw=win_probs.get("draw", 0.0),
        p_away=win_probs.get("away", 0.0),
        over25=sim_result.get("over_2_5", 0.0),
        btts=sim_result.get("btts", 0.0),
        market_odds_str=market_odds_str or "  (none provided)",
        market_impl_str=market_impl_str or "  (none provided)",
    )
