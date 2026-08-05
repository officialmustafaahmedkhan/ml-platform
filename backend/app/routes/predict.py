"""Live prediction routes: single row + batch CSV."""
from __future__ import annotations

import io
import json

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import ModelArtifact, User
from ..schemas import BatchPredictResponse, PredictResponse
from ..services import storage
from ..utils.serialization import dumps, to_jsonable
from ..utils.security import get_current_user
from .helpers import log_experiment

router = APIRouter(prefix="/api/predict", tags=["prediction"])


def _load(model_id: int, user: User, db: Session) -> tuple[ModelArtifact, object, dict, list, list]:
    m = db.query(ModelArtifact).filter(ModelArtifact.id == model_id, ModelArtifact.user_id == user.id).first()
    if m is None:
        raise HTTPException(status_code=404, detail="Model not found")
    model = storage.load_model_artifact(m.filepath)
    pipeline_meta = json.loads(m.pipeline) if m.pipeline else {}
    pipeline = storage.load_pipeline(pipeline_meta.get("path"))
    feature_names = json.loads(m.feature_names) if m.feature_names else []
    class_names = json.loads(m.class_names) if m.class_names else []
    return m, model, pipeline, feature_names, class_names


def _explanation(m: ModelArtifact, feature_names: list[str], pipeline: dict) -> list[dict] | None:
    stored = json.loads(m.metrics) if m.metrics else {}
    importance = stored.get("feature_importance")
    if not importance or not importance.get("features"):
        return None
    return [
        {"feature": f, "importance": round(v, 4)}
        for f, v in zip(importance["features"][:5], importance["importance"][:5])
    ]


def _row_to_df(input_values: dict, pipeline: dict) -> pd.DataFrame:
    feature_cols = list(pipeline["feature_columns"])
    row = {}
    for col in feature_cols:
        val = input_values.get(col)
        if val is None or (isinstance(val, float) and pd.isna(val)) or val == "":
            row[col] = None
        else:
            row[col] = str(val)
    return pd.DataFrame([row], columns=feature_cols)


@router.post("", response_model=PredictResponse)
def predict_single(payload: dict, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    model_id = payload.get("model_id")
    input_values = payload.get("input") or {}
    if not model_id or not isinstance(input_values, dict):
        raise HTTPException(status_code=400, detail="model_id and input map are required")

    m, model, pipeline, feature_names, class_names = _load(int(model_id), user, db)
    try:
        df = _row_to_df(input_values, pipeline)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    from ..services import preprocessing as pp

    X = pp.apply_preprocessor(df, pipeline)
    pred_code = int(model.predict(X)[0])
    label = class_names[pred_code] if pred_code < len(class_names) else str(pred_code)

    probabilities = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        probabilities = {class_names[i]: float(probs[i]) for i in range(len(class_names))}

    log_experiment(db, user, "predict", dataset_id=m.dataset_id, model_id=m.id,
                   details={"input": to_jsonable(input_values), "prediction": label})
    db.commit()

    return PredictResponse(
        model_id=m.id,
        prediction=label,
        probabilities=probabilities,
        explanation=_explanation(m, feature_names, pipeline),
    )


@router.post("/batch", response_model=BatchPredictResponse)
def predict_batch(file: UploadFile = File(...), model_id: int = 0,
                  user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id query parameter required")

    m, model, pipeline, feature_names, class_names = _load(model_id, user, db)
    try:
        df = pd.read_csv(io.BytesIO(file.file.read()))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}")

    missing = [c for c in pipeline["feature_columns"] if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"CSV is missing feature columns: {missing}")

    from ..services import preprocessing as pp

    X = pp.apply_preprocessor(df, pipeline)
    preds = model.predict(X)
    labels = [class_names[int(p)] if int(p) < len(class_names) else str(int(p)) for p in preds]

    out_df = df.copy()
    out_df.insert(0, "prediction", labels)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        top_probs = [max(p) for p in probs]
        out_df.insert(1, "confidence", top_probs)

    path = storage.save_csv_report(out_df)
    results = out_df.head(100).to_dict(orient="records")

    log_experiment(db, user, "batch_predict", dataset_id=m.dataset_id, model_id=m.id,
                   details={"rows": int(len(out_df)), "output": str(path.name)})
    db.commit()

    return BatchPredictResponse(
        model_id=m.id,
        total=int(len(out_df)),
        results=results,
        output_filename=path.name,
    )
