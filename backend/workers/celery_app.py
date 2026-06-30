from celery import Celery
from backend.config import settings

celery_app = Celery(
    "football_di",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=[
        "backend.workers.tasks_odds",
        "backend.workers.tasks_features",
        "backend.workers.tasks_predictions",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    beat_schedule={
        "ingest-odds-every-5-minutes": {
            "task": "backend.workers.tasks_odds.ingest_live_odds",
            "schedule": 300.0,  # every 5 minutes
        },
        "materialize-features-every-hour": {
            "task": "backend.workers.tasks_features.materialize_all_features",
            "schedule": 3600.0,
        },
        "run-predictions-every-hour": {
            "task": "backend.workers.tasks_predictions.predict_upcoming_matches",
            "schedule": 3600.0,
        },
    },
)
