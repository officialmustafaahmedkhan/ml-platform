"""Shared pipeline orchestration used by preprocess/train/compare/predict routes."""
from __future__ import annotations

from typing import Optional

import pandas as pd
from sqlalchemy.orm import Session

from ..models import Dataset
from ..utils.serialization import loads
from . import preprocessing as pp
from . import storage


def load_dataset_df(db: Session, dataset_id: int, user_id: int) -> tuple[Dataset, pd.DataFrame]:
    ds = db.query(Dataset).filter(Dataset.id == dataset_id, Dataset.user_id == user_id).first()
    if ds is None:
        raise ValueError("Dataset not found or not owned by user")
    df = storage.read_csv(ds.filepath)
    return ds, df


def resolve_target(df: pd.DataFrame, requested: Optional[str]) -> str:
    if requested and requested in df.columns:
        return requested
    return pp.default_target(df) or df.columns[-1]


def build_config(
    df: pd.DataFrame,
    target: str,
    preprocess: dict,
    model_hint: Optional[str] = None,
) -> dict:
    """Merge auto heuristics + explicit user overrides into one config dict.

    Any strategy explicitly present in ``preprocess`` (e.g. ``missing_numeric``,
    ``encoding``, ``scaling``, ``smote``) wins over the auto heuristic.
    """
    return pp.auto_config(df, target, model_hint, overrides=preprocess)


def prepare(
    db: Session,
    dataset_id: int,
    user_id: int,
    preprocess: dict,
    target_column: Optional[str] = None,
    model_hint: Optional[str] = None,
):
    """Load a dataset and produce (df, target, X, y, pipeline, report, config, profile)."""
    ds, df = load_dataset_df(db, dataset_id, user_id)
    target = resolve_target(df, target_column or preprocess.get("target_column"))
    config = build_config(df, target, preprocess, model_hint)
    profile = pp.profile_dataset(df, target)
    X, y, pipeline, report = pp.fit_preprocessor(df, target, config)
    return ds, df, target, X, y, pipeline, report, config, profile
