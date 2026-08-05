"""Shared training logic used by both the REST route and NL command executor."""
from __future__ import annotations

from sklearn.model_selection import train_test_split
from sqlalchemy.orm import Session

from ..models import ModelArtifact, User
from ..schemas import TrainRequest, TrainResponse
from ..utils.serialization import dumps
from ..services import evaluation as ev
from ..services import models as msvc
from ..services import storage
from ..services.pipeline import prepare
from ..routes.helpers import log_experiment

MODEL_LABELS = {k: v["label"] for k, v in msvc.MODEL_REGISTRY.items()}


def train_artifact(db: Session, user: User, payload: TrainRequest) -> TrainResponse:
    """Train a model, persist the artifact + experiment entry, return the summary.

    Raises ``ValueError`` for bad input, which callers translate to 400.
    """
    if payload.model_type not in msvc.MODEL_TYPES:
        raise ValueError(f"Unknown model type: {payload.model_type}")

    ds, df, target, X, y, pipeline, report, config, profile = prepare(
        db, payload.dataset_id, user.id,
        preprocess=payload.preprocess,
        target_column=payload.target_column,
        model_hint=payload.model_type,
    )

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=payload.test_size, random_state=payload.random_state, stratify=y
        )
    except ValueError:
        # Stratification impossible (e.g. singleton class) -> plain split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=payload.test_size, random_state=payload.random_state
        )

    model, clean_params, cv_mean = msvc.train_model(
        X_train, y_train, payload.model_type, payload.params,
        random_state=payload.random_state, tune=payload.tune, cv_folds=payload.cv_folds,
    )

    eval_result = ev.evaluate_model(model, X_test, y_test, pipeline["class_names"])
    importance = ev.feature_importance(model, pipeline["feature_names"])

    model_path = storage.save_model_artifact(model)
    pipeline_path = storage.save_pipeline(pipeline)

    artifact = ModelArtifact(
        user_id=user.id,
        dataset_id=payload.dataset_id,
        name=f"{MODEL_LABELS[payload.model_type]} on {ds.name}",
        model_type=payload.model_type,
        params=dumps({"params": clean_params, "tune": payload.tune,
                      "cv_mean_accuracy": cv_mean, "test_size": payload.test_size}),
        pipeline=dumps({"path": str(pipeline_path), "config": config, "target_column": target}),
        filepath=str(model_path),
        metrics=dumps({"metrics": eval_result["metrics"], "confusion_matrix": eval_result["confusion_matrix"],
                       "feature_importance": importance, "test_size": payload.test_size,
                       "random_state": payload.random_state}),
        feature_names=dumps(pipeline["feature_names"]),
        class_names=dumps(pipeline["class_names"]),
    )
    db.add(artifact)
    db.flush()
    log_experiment(db, user, "train", dataset_id=payload.dataset_id, model_id=artifact.id,
                   details={"model_type": payload.model_type, "accuracy": eval_result["metrics"]["accuracy"]})
    db.commit()
    db.refresh(artifact)

    return TrainResponse(
        model_id=artifact.id,
        model_type=payload.model_type,
        name=artifact.name,
        params=clean_params,
        metrics=eval_result["metrics"],
        feature_names=pipeline["feature_names"],
        class_names=pipeline["class_names"],
        dataset_id=payload.dataset_id,
    )
