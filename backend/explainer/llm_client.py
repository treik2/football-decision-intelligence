import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Thin wrapper around OpenAI chat completions.
    Falls back to a stub message when OPENAI_API_KEY is not configured.
    """

    def __init__(self):
        self._key = os.getenv("OPENAI_API_KEY", "")
        self._model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self._client = None
        if self._key:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(api_key=self._key)
            except ImportError:
                logger.warning("openai package not installed — LLM explanations disabled")

    async def complete(self, prompt: str, max_tokens: int = 500) -> str:
        if self._client is None:
            return (
                "LLM explanation unavailable: set OPENAI_API_KEY in your environment "
                "and install the openai package."
            )
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": "You are a professional football analyst."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.error(f"LLM call failed: {exc}")
            return f"Explanation unavailable due to API error: {exc}"
