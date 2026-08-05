"""Storage abstraction for uploads, pickled models/pipelines and reports.

The provider is selected with ``STORAGE_PROVIDER``:

* ``local`` (default) - files on the local filesystem under ``backend/data/``.
* ``vercel_blob`` - Vercel Blob object storage (serverless-friendly; read/write
  is done via the official ``vercel-blob`` SDK using ``BLOB_READ_WRITE_TOKEN``).

Keys persisted in the database are **relative** (e.g. ``uploads/x.csv``,
``models/x.pkl``, ``reports/x.csv``) so rows are portable across providers.
For backwards compatibility the local provider also resolves legacy absolute
paths stored by earlier versions of the app.
"""
from __future__ import annotations

import io
import os
import pickle
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path

import joblib
import pandas as pd

from ..config import DATA_DIR, MODEL_DIR, REPORT_DIR, UPLOAD_DIR, settings


def _unique_name(prefix: str, ext: str) -> str:
    return f"{prefix}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}{ext}"


def _provider() -> str:
    return (settings.STORAGE_PROVIDER or "local").strip().lower()


def _is_blob() -> bool:
    return _provider() == "vercel_blob"


# --------------------------------------------------------------------------- #
# Provider plumbing
# --------------------------------------------------------------------------- #
def _local_resolve(key: str | Path) -> Path:
    """Resolve a relative key (or legacy absolute path) on the local provider."""
    path = Path(key)
    if path.is_absolute():
        return path
    return DATA_DIR / path


def _blob_url(key: str) -> str:
    """Public URL for a blob object derived from ``BLOB_STORE_ID``."""
    store_id = os.getenv("BLOB_STORE_ID", "").strip() or os.getenv("BLOB_STORE", "").strip()
    sub = store_id.removeprefix("store_").lower()
    if not sub:
        raise RuntimeError("vercel_blob provider requires BLOB_STORE_ID")
    return f"https://{sub}.public.blob.vercel-storage.com/{key.lstrip('/')}"


def _blob_put(key: str, data: bytes) -> None:
    try:
        from vercel_blob import put
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "vercel_blob provider requires the 'vercel-blob' package"
        ) from exc
    put(key, data)


def _blob_get(key: str) -> bytes:
    if not os.getenv("BLOB_STORE_ID"):
        raise RuntimeError("vercel_blob provider requires BLOB_STORE_ID")
    with urllib.request.urlopen(_blob_url(key), timeout=30) as resp:
        return resp.read()


def _blob_delete(key: str) -> None:
    try:
        from vercel_blob import delete
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "vercel_blob provider requires the 'vercel-blob' package"
        ) from exc
    delete(_blob_url(key))


def _write(key: str, data: bytes) -> str:
    if _is_blob():
        _blob_put(key, data)
    else:
        path = _local_resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return key


def _read_bytes(key: str | Path) -> bytes:
    if _is_blob():
        return _blob_get(str(key))
    return _local_resolve(key).read_bytes()


# --------------------------------------------------------------------------- #
# Uploads
# --------------------------------------------------------------------------- #
def save_upload(filename: str, content: bytes) -> str:
    """Persist an uploaded CSV (or labelled version) and return its storage key."""
    safe = Path(filename).name
    key = f"uploads/{_unique_name('upload', Path(safe).suffix or '.csv')}"
    return _write(key, content)


# --------------------------------------------------------------------------- #
# Models / pipelines
# --------------------------------------------------------------------------- #
def save_model_artifact(model, filename: str | None = None) -> str:
    key = f"models/{filename or _unique_name('model', '.pkl')}"
    buf = io.BytesIO()
    joblib.dump(model, buf)
    return _write(key, buf.getvalue())


def load_model_artifact(key: str | Path):
    if _is_blob():
        return joblib.load(io.BytesIO(_blob_get(str(key))))
    return joblib.load(_local_resolve(key))


def save_pipeline(pipeline: dict, filename: str | None = None) -> str:
    key = f"models/{filename or _unique_name('pipeline', '.pkl')}"
    buf = io.BytesIO()
    pickle.dump(pipeline, buf, protocol=pickle.HIGHEST_PROTOCOL)
    return _write(key, buf.getvalue())


def load_pipeline(key: str | Path) -> dict:
    if _is_blob():
        return pickle.load(io.BytesIO(_blob_get(str(key))))
    with open(_local_resolve(key), "rb") as fh:
        return pickle.load(fh)


# --------------------------------------------------------------------------- #
# Reports / data frames
# --------------------------------------------------------------------------- #
def save_csv_report(df: pd.DataFrame, filename: str | None = None) -> str:
    key = f"reports/{filename or _unique_name('report', '.csv')}"
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return _write(key, buf.getvalue().encode("utf-8"))


def read_csv(key: str | Path) -> pd.DataFrame:
    """Read a dataset/report CSV back from whatever provider is active."""
    if _is_blob():
        return pd.read_csv(io.BytesIO(_blob_get(str(key))))
    return pd.read_csv(_local_resolve(key))


def get_bytes(key: str | Path) -> bytes:
    return _read_bytes(key)


def to_local_path(key: str | Path) -> Path:
    """Return a local file path that can be served via FileResponse.

    For the local provider this resolves the key under ``backend/data``; for
    the blob provider the object is downloaded into a temp file first.
    """
    if _is_blob():
        tmp = Path(tempfile.gettempdir()) / ("ml_blob_" + str(Path(str(key)).name))
        tmp.write_bytes(_blob_get(str(key)))
        return tmp
    return _local_resolve(key)


def delete_file(key: str | Path) -> None:
    try:
        if _is_blob():
            _blob_delete(str(key))
        else:
            _local_resolve(key).unlink(missing_ok=True)
    except OSError:
        pass
