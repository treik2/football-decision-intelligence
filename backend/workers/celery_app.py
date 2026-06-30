from celery import Celery
from backend.config import settings

celery_app = Celery(
    "football_di",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "backend.workers.tasks.odds_ingest",
        "backend.workers.tasks.feature_materialize",
        "backend.workers.tasks.training",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    beat_schedule={
        "ingest-odds-every-5-minutes": {
            "task": "backend.workers.tasks.odds_ingest.ingest_live_odds",
            "schedule": 300.0,
        },
        "materialize-features-every-hour": {
            "task": "backend.workers.tasks.feature_materialize.materialize_all",
            "schedule": 3600.0,
        },
    },
)
