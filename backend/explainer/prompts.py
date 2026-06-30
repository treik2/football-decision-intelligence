EXPLANATION_PROMPT = """
You are an expert football analyst and quantitative betting researcher.
Given the following match data and model signals, produce a concise analytical explanation.

Your response must include:
1. A 3-4 sentence plain-language summary of why the model sees value (or not).
2. Top 3 reasons supporting the model estimate, as bullet points.
3. A calibration note: how confident is the model and should the user trust recent market movement?
4. A one-sentence risk warning.

Do NOT recommend a specific stake or claim certainty. The user makes the final decision.

--- MATCH ---
Home: {home}
Away: {away}
League: {league}
Kickoff: {kickoff}

--- MODEL SIGNALS ---
Model win probabilities:
  Home win: {home_win_pct}%
  Draw:     {draw_pct}%
  Away win: {away_win_pct}%

Market implied probabilities:
  Home win: {mkt_home_pct}%
  Draw:     {mkt_draw_pct}%
  Away win: {mkt_away_pct}%

Expected goals:
  Home xG: {home_xg}
  Away xG: {away_xg}
  xG diff: {xg_diff}

Elo rating difference (home minus away): {elo_diff}
Home form (last 5, pts): {home_form}
Away form (last 5, pts): {away_form}
Home rest days: {home_rest}
Away rest days: {away_rest}
Home injury score (0=full squad, 1=heavily depleted): {home_injury}
Away injury score: {away_injury}
Weather: {weather}

Simulation results (100k runs):
  Over 2.5 goals: {over25_pct}%
  BTTS:           {btts_pct}%
  Over 1.5 goals: {over15_pct}%

Market movement: {market_move}

--- VALUE LEGS IDENTIFIED ---
{value_legs}

Analyse the above and respond clearly.
"""


def build_prompt(
    home: str,
    away: str,
    league: str,
    kickoff: str,
    features: dict,
    prediction,
    market_odds: dict,
    value_legs: list,
    market_move: str = "none",
) -> str:
    from backend.betting.ev import implied_prob

    mkt_home = implied_prob(market_odds.get("home_win", 2.0)) * 100
    mkt_draw = implied_prob(market_odds.get("draw", 3.5)) * 100
    mkt_away = implied_prob(market_odds.get("away_win", 4.0)) * 100

    legs_text = "\n".join(
        f"  - {l['name']}: model {l['probability']*100:.1f}% vs market {l['implied_prob']*100:.1f}% | EV={l['ev']:+.3f} | edge={l['edge']*100:.1f}%"
        for l in value_legs
    ) or "  None identified above threshold."

    return EXPLANATION_PROMPT.format(
        home=home,
        away=away,
        league=league,
        kickoff=kickoff,
        home_win_pct=round(prediction.home_win_prob * 100, 1),
        draw_pct=round(prediction.draw_prob * 100, 1),
        away_win_pct=round(prediction.away_win_prob * 100, 1),
        mkt_home_pct=round(mkt_home, 1),
        mkt_draw_pct=round(mkt_draw, 1),
        mkt_away_pct=round(mkt_away, 1),
        home_xg=round(prediction.home_xg, 2),
        away_xg=round(prediction.away_xg, 2),
        xg_diff=round(prediction.home_xg - prediction.away_xg, 2),
        elo_diff=features.get("elo_diff", 0),
        home_form=features.get("home_form", "N/A"),
        away_form=features.get("away_form", "N/A"),
        home_rest=features.get("home_rest_days", "N/A"),
        away_rest=features.get("away_rest_days", "N/A"),
        home_injury=round(features.get("home_injury_score", 0), 2),
        away_injury=round(features.get("away_injury_score", 0), 2),
        weather=features.get("weather_desc", "unknown"),
        over25_pct=round(prediction.over25_prob * 100, 1),
        btts_pct=round(prediction.btts_prob * 100, 1),
        over15_pct=round(prediction.over15_prob * 100, 1),
        market_move=market_move,
        value_legs=legs_text,
    )
