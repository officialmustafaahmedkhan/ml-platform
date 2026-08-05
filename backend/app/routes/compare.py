"""Model comparison engine: trains every model on the same split and ranks them."""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from sklearn.model_selection import train_test_split
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import ModelArtifact, User
from ..schemas import ComparisonRequest, ComparisonResponse
from ..services import evaluation as ev
from ..services import models as msvc
from ..services import storage
from ..services.pipeline import prepare
from ..utils.serialization import dumps, to_jsonable
from ..utils.security import get_current_user
from .helpers import log_experiment

router = APIRouter(prefix="/api/compare", tags=["comparison"])


@router.post("", response_model=ComparisonResponse)
def compare_models(payload: ComparisonRequest,
                   user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        ds, df, target, X, y, pipeline, report, config, profile = prepare(
            db, payload.dataset_id, user.id,
            preprocess=payload.preprocess, target_column=payload.target_column,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=payload.test_size, random_state=payload.random_state, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=payload.test_size, random_state=payload.random_state
        )

    model_types = payload.model_types or (["dt", "knn", "rf"] + (["voting"] if payload.include_hybrid else []))
    invalid = [mt for mt in model_types if mt not in msvc.MODEL_TYPES]
    if invalid:
        raise HTTPException(status_code=400, detail=f"Unknown model type(s): {invalid}")
    model_types = list(dict.fromkeys(model_types))  # de-duplicate, preserve order
    if not model_types:
        raise HTTPException(status_code=400, detail="Select at least one model to compare")

    rows = []
    for mt in model_types:
        model = msvc.build_model(mt, None, payload.random_state)
        model.fit(X_train, y_train)
        result = ev.evaluate_model(model, X_test, y_test, pipeline["class_names"])
        metrics = result["metrics"]

        importance = ev.feature_importance(model, pipeline["feature_names"])
        model_path = storage.save_model_artifact(model)
        pipeline_path = storage.save_pipeline(pipeline)

        artifact = ModelArtifact(
            user_id=user.id,
            dataset_id=payload.dataset_id,
            name=f"Compare: {msvc.MODEL_REGISTRY[mt]['label']} on {ds.name}",
            model_type=mt,
            params=dumps({"params": {}, "test_size": payload.test_size}),
            pipeline=dumps({"path": str(pipeline_path), "config": config, "target_column": target}),
            filepath=str(model_path),
            metrics=dumps({"metrics": metrics, "confusion_matrix": result["confusion_matrix"],
                           "feature_importance": importance, "test_size": payload.test_size,
                           "random_state": payload.random_state}),
            feature_names=dumps(pipeline["feature_names"]),
            class_names=dumps(pipeline["class_names"]),
        )
        db.add(artifact)
        db.commit()  # short transaction: release the SQLite write lock promptly

        rows.append({
            "model": msvc.MODEL_REGISTRY[mt]["label"],
            "model_type": mt,
            "model_id": artifact.id,
            "accuracy": metrics["accuracy"],
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
            "f1_macro": metrics["f1_macro"],
            "precision_weighted": metrics["precision_weighted"],
            "recall_weighted": metrics["recall_weighted"],
            "f1_weighted": metrics["f1_weighted"],
        })

    log_experiment(db, user, "compare", dataset_id=payload.dataset_id,
                   details={"models": [r["model_type"] for r in rows],
                            "best": max(rows, key=lambda r: r["accuracy"])["model_type"]})
    db.commit()

    best = max(rows, key=lambda r: r["accuracy"])
    return ComparisonResponse(
        table=to_jsonable(rows),
        charts={
            "accuracy_comparison": ev.chart_accuracy_comparison(rows),
            "metric_radar": ev.chart_metric_radar(rows),
        },
        best_model=to_jsonable(best),
        dataset_id=payload.dataset_id,
    )
