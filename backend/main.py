from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api import matches, predictions, bets, explain, odds
from backend.db.base import engine, Base
from backend.config import settings

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Football Decision Intelligence API",
    description="Calibrated football prediction and value bet identification platform",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(matches.router, prefix="/api/v1/matches", tags=["Matches"])
app.include_router(predictions.router, prefix="/api/v1/predictions", tags=["Predictions"])
app.include_router(bets.router, prefix="/api/v1/bets", tags=["Bets"])
app.include_router(explain.router, prefix="/api/v1/explain", tags=["Explain"])
app.include_router(odds.router, prefix="/api/v1/odds", tags=["Odds"])


@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}
