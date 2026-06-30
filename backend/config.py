from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # App
    app_name: str = "Football Decision Intelligence"
    debug: bool = False
    log_level: str = "INFO"
    secret_key: str = "change_me"

    # Database
    database_url: str = "postgresql+asyncpg://fdi:fdi_secret@postgres:5432/fdi"
    database_url_sync: str = "postgresql://fdi:fdi_secret@postgres:5432/fdi"

    # Redis
    redis_url: str = "redis://redis:6379/0"
    celery_broker_url: str = "redis://redis:6379/1"
    celery_result_backend: str = "redis://redis:6379/2"

    # External APIs
    odds_api_key: str = ""
    openweather_api_key: str = ""
    openai_api_key: str = ""

    # Monte Carlo
    mc_simulations: int = 100_000
    mc_seed: int = 42

    # Kelly / Bet sizing
    kelly_fraction: float = 0.25
    max_stake_pct: float = 0.05
    min_ev_threshold: float = 0.01
    min_confidence: float = 0.40
    max_leg_correlation: float = 0.60

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
