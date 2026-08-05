"""FastAPI application entry point."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import settings
from .database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "User-Personalized Intelligent Machine Learning Platform with "
        "Hybrid Ensemble Learning. Train, evaluate, compare and deploy "
        "Decision Tree, KNN, Random Forest and hybrid ensemble models."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok", "app": settings.APP_NAME, "version": settings.APP_VERSION}


from .routes import (  # noqa: E402
    assistant,
    auth,
    commands,
    compare,
    dashboard,
    datasets,
    evaluate,
    experiments,
    explain,
    export,
    llm,
    pipeline,
    predict,
    preprocess,
    recommendation,
    train,
)

for r in (auth, datasets, preprocess, train, evaluate, predict, compare, recommendation, dashboard, export, llm, assistant, explain, experiments, commands, pipeline):
    app.include_router(r.router)

# Serve the built React frontend (production single-server mode) if present.
_frontend_dist = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
if _frontend_dist.exists():
    app.mount("/", StaticFiles(directory=_frontend_dist, html=True), name="frontend")
