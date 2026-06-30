# ⚽ Football Decision Intelligence Platform

A production-grade **decision intelligence platform** for football betting analytics. This is NOT an automated betting bot — the AI estimates probabilities, quantifies uncertainty, explains predictions, and surfaces positive-EV bets. **You make the final decision.**

## Architecture

```
Odds Providers (Pinnacle, Bet365, OddsAPI)
        │
   Feature Engineering Layer
   (Rolling xG, Form, Elo, Rest, Injuries, Weather, Market)
        │
   Feature Store (Feast stub / Redis)
        │
   ML Pipeline: LightGBM + Poisson + Monte Carlo
        │
   Bet Builder (EV + Fractional Kelly + Correlation Filter)
        │
   FastAPI Prediction Service
        │
   Next.js Dashboard
```

## Stack

**Backend:** Python, FastAPI, Celery, Redis, PostgreSQL, TimescaleDB, DuckDB  
**ML:** LightGBM, CatBoost, XGBoost, Bayesian Poisson, Monte Carlo (NumPy)  
**Feature Store:** Feast (stub included, pluggable)  
**Frontend:** Next.js 14, Tailwind CSS, shadcn/ui, TanStack Query, Recharts  
**Infra:** Docker, Docker Compose, PostgreSQL, Redis  

## Quick Start

```bash
git clone https://github.com/treik2/football-decision-intelligence
cd football-decision-intelligence
cp .env.example .env
docker-compose up --build
```

- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Frontend: http://localhost:3000

## Repository Layout

```
backend/
  api/          FastAPI routers
  models/       ML model wrappers
  simulation/   Monte Carlo simulator
  betting/      Bet builder, EV, Kelly
  features/     Feature store client and engineering
  explainer/    LLM explanation prompt builder
  workers/      Celery tasks
  db/           Database connections and models
  utils/        Odds helpers, calibration
frontend/       Next.js dashboard
ml/
  training/     LightGBM + CatBoost training scripts
  poisson/      Bayesian Poisson goal model
  calibration/  Platt / isotonic calibration
  backtest/     Historical backtest runner
infra/
  docker/
scripts/        Data ingestion, seed scripts
```

## Key Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /v1/matches/ingest | Ingest match + features |
| GET  | /v1/matches/{id}/predict | Full model prediction |
| POST | /v1/matches/{id}/simulate | Monte Carlo simulation |
| POST | /v1/bets/suggest | Build & rank bet suggestions |
| GET  | /v1/bets/value | List all value bets |
| GET  | /v1/odds/live | Live odds feed |
| POST | /v1/explain/{id} | LLM explanation prompt |
| GET  | /v1/backtest/summary | Backtest results |

## Disclaimer

This platform is for research and analytical purposes only. No model can reliably produce "safe" bets. Betting markets are highly efficient. Use calibrated models, fractional Kelly sizing, and strict bankroll management. Always verify legal/regulatory compliance in your jurisdiction.
