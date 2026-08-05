"""Shared route helpers (DTO conversions, experiment logging)."""
from __future__ import annotations

import json

from sqlalchemy.orm import Session

from ..models import Dataset, Experiment, ModelArtifact, User
from ..schemas import DatasetOut, ModelOut
from ..utils.serialization import dumps


def dataset_out(ds: Dataset) -> DatasetOut:
    return DatasetOut(
        id=ds.id,
        name=ds.name,
        filename=ds.filename,
        rows=ds.rows,
        columns=json.loads(ds.columns) if ds.columns else [],
        preview=json.loads(ds.preview) if ds.preview else [],
        profile=json.loads(ds.profile) if ds.profile else {},
        versions=json.loads(ds.versions) if ds.versions else [],
        created_at=ds.created_at,
    )


def model_out(m: ModelArtifact) -> ModelOut:
    return ModelOut(
        id=m.id,
        name=m.name,
        model_type=m.model_type,
        params=json.loads(m.params) if m.params else {},
        pipeline=json.loads(m.pipeline) if m.pipeline else {},
        metrics=json.loads(m.metrics) if m.metrics else {},
        feature_names=json.loads(m.feature_names) if m.feature_names else [],
        class_names=json.loads(m.class_names) if m.class_names else [],
        dataset_id=m.dataset_id,
        created_at=m.created_at,
    )


def log_experiment(
    db: Session,
    user: User,
    action: str,
    dataset_id: int | None = None,
    model_id: int | None = None,
    details: dict | None = None,
) -> Experiment:
    exp = Experiment(
        user_id=user.id,
        dataset_id=dataset_id,
        model_id=model_id,
        action=action,
        details=dumps(details or {}),
    )
    db.add(exp)
    return exp
