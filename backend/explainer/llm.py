"""LLM-powered explanation layer.
Builds a structured prompt from model signals, calls OpenAI (if key set),
and returns a formatted explanation dict. Falls back to template text.
"""
from __future__ import annotations
import os
from typing import Dict

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

PROMPT_TEMPLATE = """
You are an expert football analyst providing reasoning — NOT predictions.
Given the model signals below, explain in 3-4 plain sentences why the model
favours the indicated outcome. Then list the top 3 supporting reasons.
Finally, add a one-sentence calibration note about model confidence.

Match: {home} vs {away}
League: {league}

Model signals:
- Elo difference (home minus away): {elo_diff:+.0f}
- xG difference (home minus away): {xg_diff:+.2f}
- Home expected goals: {home_xg:.2f}
- Away expected goals: {away_xg:.2f}
- Home recent form (last 5): {home_form}
- Away recent form (last 5): {away_form}
- Home rest days: {home_rest}
- Away rest days: {away_rest}
- Home key injuries: {home_injuries}
- Away key injuries: {away_injuries}
- Weather: {weather}
- Market odds movement: {market_move}
- Model win probabilities: Home {home_win:.1%} | Draw {draw:.1%} | Away {away_win:.1%}
- Market implied: Home {imp_home:.1%} | Draw {imp_draw:.1%} | Away {imp_away:.1%}

Respond with:
EXPLANATION: <3-4 sentences>
REASONS:
1. <reason>
2. <reason>
3. <reason>
CALIBRATION: <1 sentence>
"""


def build_explanation(
    features: Dict,
    model_probs: Dict[str, float],
    market_probs: Dict[str, float],
) -> Dict[str, str]:
    prompt = PROMPT_TEMPLATE.format(
        home=features.get("home_team", "Home"),
        away=features.get("away_team", "Away"),
        league=features.get("league", "Unknown"),
        elo_diff=features.get("elo_diff", 0),
        xg_diff=features.get("xg_diff", 0),
        home_xg=features.get("home_xg", 1.3),
        away_xg=features.get("away_xg", 1.0),
        home_form=features.get("home_form", "N/A"),
        away_form=features.get("away_form", "N/A"),
        home_rest=features.get("home_rest_days", "N/A"),
        away_rest=features.get("away_rest_days", "N/A"),
        home_injuries=features.get("home_injuries", "none reported"),
        away_injuries=features.get("away_injuries", "none reported"),
        weather=features.get("weather_summary", "dry"),
        market_move=features.get("market_move", "stable"),
        home_win=model_probs.get("home", 0),
        draw=model_probs.get("draw", 0),
        away_win=model_probs.get("away", 0),
        imp_home=market_probs.get("home", 0),
        imp_draw=market_probs.get("draw", 0),
        imp_away=market_probs.get("away", 0),
    )

    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.3,
            )
            raw = resp.choices[0].message.content
            return _parse_llm_response(raw)
        except Exception:
            pass

    # fallback template
    home = features.get("home_team", "Home")
    away = features.get("away_team", "Away")
    elo_diff = features.get("elo_diff", 0)
    xg_diff = features.get("xg_diff", 0)
    return {
        "text": (
            f"{home} hold a {elo_diff:+.0f} Elo advantage over {away} "
            f"with an xG difference of {xg_diff:+.2f}. "
            f"The model assigns {model_probs.get('home', 0):.1%} probability to a home win. "
            "Always verify against latest news and lineups before placing any bet."
        ),
        "reasons": [
            f"Elo advantage: {elo_diff:+.0f} points",
            f"xG advantage: {xg_diff:+.2f}",
            f"Home win probability exceeds market implied by "
            f"{model_probs.get('home',0) - market_probs.get('home',0):+.1%}",
        ],
        "calibration": "Fallback estimates — load trained model for calibrated output.",
    }


def _parse_llm_response(raw: str) -> Dict[str, str]:
    text, reasons, calib = "", [], ""
    lines = raw.strip().splitlines()
    section = None
    for line in lines:
        if line.startswith("EXPLANATION:"):
            text = line.replace("EXPLANATION:", "").strip()
            section = "explanation"
        elif line.startswith("REASONS:"):
            section = "reasons"
        elif line.startswith("CALIBRATION:"):
            calib = line.replace("CALIBRATION:", "").strip()
            section = None
        elif section == "explanation" and line.strip():
            text += " " + line.strip()
        elif section == "reasons" and line.strip():
            reasons.append(line.lstrip("123. ").strip())
    return {"text": text.strip(), "reasons": reasons[:3], "calibration": calib}
