from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api import matches, predictions, bets, explain, odds
from backend.db.base import Base, engine

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Football Decision Intelligence API",
    description="Calibrated probability estimates, EV-based bet suggestions, and explainable predictions for football markets.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(matches.router, prefix="/matches", tags=["Matches"])
app.include_router(predictions.router, prefix="/predictions", tags=["Predictions"])
app.include_router(bets.router, prefix="/bets", tags=["Bets"])
app.include_router(explain.router, prefix="/explain", tags=["Explain"])
app.include_router(odds.router, prefix="/odds", tags=["Odds"])


@app.get("/health")
def health():
    return {"status": "ok"}
