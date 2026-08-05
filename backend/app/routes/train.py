"""Training + model artifact management routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import ModelArtifact, User
from ..schemas import ModelOut, TrainRequest, TrainResponse
from ..services import models as msvc
from ..services import storage
from ..services.training import train_artifact
from ..utils.security import get_current_user
from .helpers import model_out

router = APIRouter(prefix="/api/train", tags=["training"])


@router.post("", response_model=TrainResponse)
def train_model(payload: TrainRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        return train_artifact(db, user, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/models", response_model=list[ModelOut])
def list_models(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    artifacts = db.query(ModelArtifact).filter(ModelArtifact.user_id == user.id) \
        .order_by(ModelArtifact.created_at.desc()).all()
    return [model_out(m) for m in artifacts]


@router.get("/models/{model_id}", response_model=ModelOut)
def get_model(model_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    m = db.query(ModelArtifact).filter(ModelArtifact.id == model_id, ModelArtifact.user_id == user.id).first()
    if m is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return model_out(m)


@router.delete("/models/{model_id}", status_code=204)
def delete_model(model_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    m = db.query(ModelArtifact).filter(ModelArtifact.id == model_id, ModelArtifact.user_id == user.id).first()
    if m is None:
        raise HTTPException(status_code=404, detail="Model not found")
    storage.delete_file(m.filepath)
    db.delete(m)
    db.commit()


@router.get("/registry")
def model_registry(user: User = Depends(get_current_user)):
    return msvc.get_model_registry()
