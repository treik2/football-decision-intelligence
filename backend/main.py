from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api import matches, predictions, bets, explain, odds
from backend.db.base import engine, Base
from backend.config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Football Decision Intelligence API",
    description="Decision intelligence platform for football betting analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(matches.router, prefix="/api/v1/matches", tags=["matches"])
app.include_router(predictions.router, prefix="/api/v1/predictions", tags=["predictions"])
app.include_router(bets.router, prefix="/api/v1/bets", tags=["bets"])
app.include_router(explain.router, prefix="/api/v1/explain", tags=["explain"])
app.include_router(odds.router, prefix="/api/v1/odds", tags=["odds"])


@app.on_event("startup")
async def startup():
    logger.info("Starting Football Decision Intelligence API")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}
