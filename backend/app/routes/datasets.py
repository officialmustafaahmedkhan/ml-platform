"""Dataset management routes (upload / list / detail / delete)."""
from __future__ import annotations

import io
import json

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Dataset, Experiment, User
from ..schemas import DatasetOut
from ..services import eda as eda_svc
from ..services import preprocessing as pp
from ..services import storage
from ..utils.serialization import dumps
from ..utils.security import get_current_user

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


def _to_dataset_out(ds: Dataset) -> DatasetOut:
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


@router.post("/upload", response_model=DatasetOut, status_code=201)
def upload_dataset(
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    raw = file.file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV is empty")

    path = storage.save_upload(file.filename, raw)
    profile = pp.profile_dataset(df)

    name = file.filename.rsplit("/", 1)[-1]
    ds = Dataset(
        user_id=user.id,
        name=name,
        filename=name,
        filepath=str(path),
        rows=df.shape[0],
        columns=dumps([str(c) for c in df.columns]),
        preview=dumps(df.head(10).fillna("").astype(str).to_dict(orient="records")),
        profile=dumps(profile),
        versions=dumps([{"version": 1, "path": str(path), "rows": int(df.shape[0])}]),
    )
    db.add(ds)
    db.flush()

    db.add(Experiment(
        user_id=user.id, dataset_id=ds.id, action="upload",
        details=dumps({"filename": name, "rows": int(df.shape[0]), "columns": int(df.shape[1])}),
    ))
    db.commit()
    db.refresh(ds)
    return _to_dataset_out(ds)


@router.get("", response_model=list[DatasetOut])
def list_datasets(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    datasets = db.query(Dataset).filter(Dataset.user_id == user.id).order_by(Dataset.created_at.desc()).all()
    return [_to_dataset_out(ds) for ds in datasets]


@router.get("/{dataset_id}", response_model=DatasetOut)
def get_dataset(dataset_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    ds = db.query(Dataset).filter(Dataset.id == dataset_id, Dataset.user_id == user.id).first()
    if ds is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return _to_dataset_out(ds)


@router.get("/{dataset_id}/head")
def dataset_head(dataset_id: int, n: int = 20, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    ds = db.query(Dataset).filter(Dataset.id == dataset_id, Dataset.user_id == user.id).first()
    if ds is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    df = pd.read_csv(ds.filepath)
    return {"columns": list(df.columns), "preview": df.head(n).fillna("").astype(str).to_dict(orient="records")}


@router.get("/{dataset_id}/eda")
def dataset_eda(dataset_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    ds = db.query(Dataset).filter(Dataset.id == dataset_id, Dataset.user_id == user.id).first()
    if ds is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    try:
        df = pd.read_csv(ds.filepath)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read dataset: {exc}")
    profile = json.loads(ds.profile) if ds.profile else {}
    return eda_svc.eda_dataset(df, profile.get("target_column"))


@router.delete("/{dataset_id}", status_code=204)
def delete_dataset(dataset_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    ds = db.query(Dataset).filter(Dataset.id == dataset_id, Dataset.user_id == user.id).first()
    if ds is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    db.delete(ds)
    db.commit()
