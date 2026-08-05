"""Model evaluation routes (metrics + charts).

Besides the persisted metrics, the endpoint regenerates a battery of
prediction-oriented charts (ROC, precision-recall, learning curve, class
balance, predicted-vs-actual, confidence histogram, correlation heatmap) by
re-running the stored preprocessing pipeline over the dataset.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sklearn.model_selection import train_test_split
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Dataset, ModelArtifact, User
from ..schemas import EvaluateResponse
from ..services import evaluation as ev
from ..services import preprocessing as pp
from ..services import storage
from ..utils.serialization import to_jsonable
from ..utils.security import get_current_user

router = APIRouter(prefix="/api/evaluate", tags=["evaluation"])


def _performance_charts(db: Session, m: ModelArtifact, model, class_names: list[str], stored: dict) -> dict:
    """Regenerate advanced charts from the persisted pipeline + dataset."""
    ds = db.query(Dataset).filter(Dataset.id == m.dataset_id).first()
    if ds is None:
        return {}
    df = storage.read_csv(ds.filepath)

    pipeline_meta = json.loads(m.pipeline) if m.pipeline else {}
    pipeline = storage.load_pipeline(pipeline_meta.get("path"))
    if not pipeline:
        return {}

    target = pipeline.get("target_column")
    encoder = pipeline.get("target_encoder")
    if not target or target not in df.columns or encoder is None:
        return {}

    X = pp.apply_preprocessor(df[df[target].notna()].copy(), pipeline)
    y = encoder.transform(df[df[target].notna()][target].astype(str))
    if y.shape[0] != X.shape[0]:
        return {}

    test_size = stored.get("test_size", 0.2)
    random_state = stored.get("random_state", 42)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    charts: dict = {}

    def _try(name: str, fn) -> None:
        try:
            uri = fn()
            if uri:
                charts[name] = uri
        except Exception:  # noqa: BLE001 - best-effort charts
            charts[name] = None

    _try("class_balance", lambda: ev.chart_class_balance(y_train, class_names))

    y_pred = model.predict(X_test)
    _try("predicted_vs_actual", lambda: ev.chart_predicted_vs_actual(y_test, y_pred, class_names))

    has_proba = hasattr(model, "predict_proba")
    if has_proba and len(class_names) >= 2:
        try:
            y_proba = model.predict_proba(X_test)
        except Exception:  # noqa: BLE001
            y_proba = None
        if y_proba is not None and y_proba.shape[1] == len(class_names):
            _try("roc_curve", lambda: ev.chart_roc_curve(y_test, y_proba, class_names))
            _try("precision_recall", lambda: ev.chart_precision_recall(y_test, y_proba, class_names))
            _try("probability_histogram", lambda: ev.chart_probability_histogram(y_test, y_proba, class_names))

    cv = max(2, min(5, X_train.shape[0] // 5))
    if X_train.shape[0] >= 12:
        _try("learning_curve", lambda: ev.chart_learning_curve(model, X_train, y_train, cv=cv))

    numeric_cols = [c for c in df.columns if c != target and pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) >= 2:
        _try("correlation_heatmap", lambda: ev.chart_correlation_heatmap(df[numeric_cols]))

    return charts


@router.get("/{model_id}", response_model=EvaluateResponse)
def evaluate_model(model_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    m = db.query(ModelArtifact).filter(ModelArtifact.id == model_id, ModelArtifact.user_id == user.id).first()
    if m is None:
        raise HTTPException(status_code=404, detail="Model not found")

    stored = json.loads(m.metrics) if m.metrics else {}
    metrics = stored.get("metrics", {})
    cm = stored.get("confusion_matrix", [])
    importance = stored.get("feature_importance")
    class_names = json.loads(m.class_names) if m.class_names else []
    feature_names = json.loads(m.feature_names) if m.feature_names else []

    charts: dict = {}
    if cm:
        charts["confusion_matrix"] = ev.chart_confusion_matrix(cm, class_names)
    if importance and importance.get("features"):
        charts["feature_importance"] = ev.chart_feature_importance(importance)

    model = None
    try:
        model = storage.load_model_artifact(m.filepath)
    except Exception:  # noqa: BLE001
        model = None

    if model is not None:
        try:
            charts["tree"] = ev.chart_tree(model, feature_names, class_names)
        except Exception:  # noqa: BLE001 - only DecisionTree supports a plot
            charts["tree"] = None

        try:
            charts.update(_performance_charts(db, m, model, class_names, stored))
        except Exception:  # noqa: BLE001 - charts are best-effort
            pass

    return EvaluateResponse(
        model_id=m.id,
        model_type=m.model_type,
        metrics=to_jsonable(metrics),
        confusion_matrix=cm,
        class_names=class_names,
        feature_importance=importance,
        charts=charts,
    )
