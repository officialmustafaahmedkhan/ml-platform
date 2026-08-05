"""Experiment workspace routes: filterable activity log + per-experiment notes."""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Dataset, Experiment, ModelArtifact, User
from ..schemas import ExperimentOut, ExperimentUpdate
from ..utils.security import get_current_user

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


def _out(exp: Experiment, db: Session) -> ExperimentOut:
    ds = db.get(Dataset, exp.dataset_id) if exp.dataset_id else None
    m = db.get(ModelArtifact, exp.model_id) if exp.model_id else None
    return ExperimentOut(
        id=exp.id,
        action=exp.action,
        details=json.loads(exp.details) if exp.details else {},
        dataset_id=exp.dataset_id,
        model_id=exp.model_id,
        notes=exp.notes or "",
        created_at=exp.created_at,
        dataset_name=ds.name if ds else None,
        model_name=m.name if m else None,
    )


@router.get("", response_model=list[ExperimentOut])
def list_experiments(
    action: str | None = None,
    dataset_id: int | None = None,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    q = db.query(Experiment).filter(Experiment.user_id == user.id)
    if action:
        q = q.filter(Experiment.action == action)
    if dataset_id:
        q = q.filter(Experiment.dataset_id == dataset_id)
    rows = q.order_by(Experiment.created_at.desc()).all()
    return [_out(e, db) for e in rows]


@router.patch("/{experiment_id}", response_model=ExperimentOut)
def update_experiment_notes(
    experiment_id: int,
    payload: ExperimentUpdate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    exp = (
        db.query(Experiment)
        .filter(Experiment.id == experiment_id, Experiment.user_id == user.id)
        .first()
    )
    if exp is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    exp.notes = payload.notes
    db.commit()
    db.refresh(exp)
    return _out(exp, db)
