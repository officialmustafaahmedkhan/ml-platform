"""Personalized dashboard aggregation routes."""
from __future__ import annotations

import json

import pandas as pd
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Dataset, Experiment, ModelArtifact, User
from ..schemas import DashboardResponse, UserOut
from ..services import evaluation as ev
from ..services import recommendation as rec
from ..services import preprocessing as pp
from ..utils.serialization import to_jsonable
from ..utils.security import get_current_user

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("", response_model=DashboardResponse)
def dashboard(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    datasets = db.query(Dataset).filter(Dataset.user_id == user.id).order_by(Dataset.created_at.desc()).all()
    models = db.query(ModelArtifact).filter(ModelArtifact.user_id == user.id).order_by(ModelArtifact.created_at.desc()).all()
    experiments = db.query(Experiment).filter(Experiment.user_id == user.id).order_by(Experiment.created_at.desc()).all()

    accuracies = []
    for m in models:
        stored = json.loads(m.metrics) if m.metrics else {}
        acc = (stored.get("metrics") or {}).get("accuracy")
        if acc is not None:
            accuracies.append(acc)
    avg_accuracy = round(sum(accuracies) / len(accuracies), 4) if accuracies else None

    type_dist: dict = {}
    for m in models:
        type_dist[m.model_type] = type_dist.get(m.model_type, 0) + 1

    trend = []
    recent_models = models[:8]
    for i, m in enumerate(reversed(recent_models)):
        stored = json.loads(m.metrics) if m.metrics else {}
        acc = (stored.get("metrics") or {}).get("accuracy")
        if acc is not None:
            trend.append({"index": i + 1, "model": m.model_type, "accuracy": acc})

    timeline = [
        {
            "action": e.action,
            "created_at": e.created_at.isoformat(),
            "details": json.loads(e.details) if e.details else {},
            "dataset_id": e.dataset_id,
            "model_id": e.model_id,
        }
        for e in experiments[:15]
    ]

    suggestions = []
    if datasets:
        latest = datasets[0]
        df = pd.read_csv(latest.filepath)
        profile = pp.profile_dataset(df)
        for s in rec.recommend_preprocessing(profile):
            suggestions.append(s)
    if models:
        latest_model = models[0]
        stored = json.loads(latest_model.metrics) if latest_model.metrics else {}
        metrics = stored.get("metrics") or {}
        for s in rec.suggest_improvements({}, metrics, latest_model.model_type):
            suggestions.append(s)

    return DashboardResponse(
        user=UserOut.model_validate(user),
        stats={
            "datasets": len(datasets),
            "models": len(models),
            "experiments": len(experiments),
            "avg_accuracy": avg_accuracy,
            "last_activity": experiments[0].created_at.isoformat() if experiments else None,
        },
        recent_datasets=[{"id": d.id, "name": d.name, "rows": d.rows, "columns": len(json.loads(d.columns or "[]")),
                          "created_at": d.created_at.isoformat()} for d in datasets[:5]],
        recent_models=[{"id": m.id, "name": m.name, "model_type": m.model_type,
                        "accuracy": (json.loads(m.metrics or "{}").get("metrics") or {}).get("accuracy"),
                        "created_at": m.created_at.isoformat()} for m in models[:5]],
        accuracy_trend=trend,
        activity_timeline=timeline,
        model_type_distribution=type_dist,
        suggestions=to_jsonable(suggestions[:6]),
    )
