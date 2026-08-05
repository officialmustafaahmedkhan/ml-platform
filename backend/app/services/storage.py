"""Filesystem storage helpers for uploads, pickled models and reports."""
from __future__ import annotations

import pickle
import time
import uuid
from pathlib import Path

import joblib
import pandas as pd

from ..config import MODEL_DIR, REPORT_DIR, UPLOAD_DIR


def _unique_name(prefix: str, ext: str) -> str:
    return f"{prefix}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}{ext}"


# --------------------------------------------------------------------------- #
# Uploads
# --------------------------------------------------------------------------- #
def save_upload(filename: str, content: bytes) -> Path:
    """Persist an uploaded CSV and return its path."""
    safe = Path(filename).name
    path = UPLOAD_DIR / _unique_name("upload", Path(safe).suffix or ".csv")
    path.write_bytes(content)
    return path


# --------------------------------------------------------------------------- #
# Models / pipelines
# --------------------------------------------------------------------------- #
def save_model_artifact(model, filename: str | None = None) -> Path:
    path = MODEL_DIR / (filename or _unique_name("model", ".pkl"))
    joblib.dump(model, path)
    return path


def load_model_artifact(path: str | Path):
    return joblib.load(path)


def save_pipeline(pipeline: dict, filename: str | None = None) -> Path:
    path = MODEL_DIR / (filename or _unique_name("pipeline", ".pkl"))
    with open(path, "wb") as fh:
        pickle.dump(pipeline, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_pipeline(path: str | Path) -> dict:
    with open(path, "rb") as fh:
        return pickle.load(fh)


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
def save_csv_report(df: pd.DataFrame, filename: str | None = None) -> Path:
    path = REPORT_DIR / (filename or _unique_name("report", ".csv"))
    df.to_csv(path, index=False)
    return path


def delete_file(path: str | Path) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass
