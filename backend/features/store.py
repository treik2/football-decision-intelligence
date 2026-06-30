"""Feature store client. Wraps Redis for online serving and PostgreSQL for offline.

In production, replace Redis direct calls with Feast online feature retrieval.
See: https://docs.feast.dev/reference/feature-servers/python-feature-server
"""
import json
import redis.asyncio as aioredis
from typing import Dict, Optional
from config import settings
from utils.logger import log


class FeatureStoreClient:
    def __init__(self):
        self._redis: Optional[aioredis.Redis] = None

    async def connect(self):
        self._redis = await aioredis.from_url(settings.redis_url, decode_responses=True)
        log.info("FeatureStore connected to Redis")

    async def close(self):
        if self._redis:
            await self._redis.close()

    async def materialize(self, match_id: str, features: Dict) -> None:
        """Push feature vector to online store (Redis), TTL 48h."""
        key = f"features:{match_id}"
        await self._redis.setex(key, 172800, json.dumps(features))
        log.debug(f"Materialized features for {match_id}")

    async def get(self, match_id: str) -> Dict:
        key = f"features:{match_id}"
        raw = await self._redis.get(key)
        if raw is None:
            raise KeyError(f"No features found for match_id={match_id}")
        return json.loads(raw)

    async def delete(self, match_id: str) -> None:
        await self._redis.delete(f"features:{match_id}")


# singleton
feature_store = FeatureStoreClient()
