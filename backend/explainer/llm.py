import openai
from backend.config import settings
from backend.explainer.prompts import build_prompt
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class ExplainerClient:
    def __init__(self):
        openai.api_key = settings.OPENAI_API_KEY
        self.model = settings.OPENAI_MODEL  # e.g. "gpt-4o-mini"

    def explain(
        self,
        home: str,
        away: str,
        features: dict,
        prediction,
        market_odds: dict = None,
        value_legs: list = None,
        market_move: str = "none",
        league: str = "",
        kickoff: str = "",
    ) -> str:
        market_odds = market_odds or {}
        value_legs = value_legs or []

        prompt = build_prompt(
            home=home,
            away=away,
            league=league,
            kickoff=kickoff,
            features=features,
            prediction=prediction,
            market_odds=market_odds,
            value_legs=value_legs,
            market_move=market_move,
        )

        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a football analytics expert. Be concise, factual, and never guarantee outcomes."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=600,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM explanation failed: {e}")
            return (
                f"Model signals: Home {prediction.home_win_prob*100:.1f}% / "
                f"Draw {prediction.draw_prob*100:.1f}% / "
                f"Away {prediction.away_win_prob*100:.1f}%. "
                f"xG: {prediction.home_xg:.2f} vs {prediction.away_xg:.2f}. "
                f"(LLM explanation unavailable: {e})"
            )
