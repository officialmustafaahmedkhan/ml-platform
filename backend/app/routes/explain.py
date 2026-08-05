"""Explainability routes: local (LIME-style) + global (permutation) importance."""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import ModelArtifact, User
from ..services import explain, storage
from ..services.pipeline import load_dataset_df
from ..utils.security import get_current_user

router = APIRouter(prefix="/api/explain", tags=["explainability"])


def _load(model_id: int, user: User, db: Session):
    m = db.query(ModelArtifact).filter(ModelArtifact.id == model_id, ModelArtifact.user_id == user.id).first()
    if m is None:
        raise HTTPException(status_code=404, detail="Model not found")
    model = storage.load_model_artifact(m.filepath)
    pipeline_meta = json.loads(m.pipeline) if m.pipeline else {}
    pipeline = storage.load_pipeline(pipeline_meta.get("path"))
    feature_names = json.loads(m.feature_names) if m.feature_names else []
    class_names = json.loads(m.class_names) if m.class_names else []
    return m, model, pipeline, feature_names, class_names


@router.post("/local")
def local(model_id: int = 0, payload: dict = None,
          user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not model_id and payload:
        model_id = int(payload.get("model_id") or 0)
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id is required")
    input_values = payload.get("input") or {}
    if not isinstance(input_values, dict) or not input_values:
        raise HTTPException(status_code=400, detail="input map is required")

    m, model, pipeline, feature_names, class_names = _load(model_id, user, db)

    df = None
    if m.dataset_id:
        try:
            _, df = load_dataset_df(db, m.dataset_id, user.id)
        except Exception:  # noqa: BLE001
            df = None

    return explain.local_explanation(
        model, pipeline, input_values, feature_names, class_names, df=df,
    )


@router.get("/global/{model_id}")
def global_importance(model_id: int, user: User = Depends(get_current_user),
                      db: Session = Depends(get_db)):
    m, model, pipeline, feature_names, class_names = _load(model_id, user, db)
    if not m.dataset_id:
        raise HTTPException(status_code=400, detail="Model has no source dataset for global analysis")

    _, df = load_dataset_df(db, m.dataset_id, user.id)
    if df is None or df.empty:
        raise HTTPException(status_code=400, detail="Source dataset is empty")

    results = explain.permutation_importance(model, pipeline, df)
    return {"method": "permutation_importance", "baseline_note": "mean accuracy drop when each feature is shuffled", "importance": results}
