"""Vercel serverless entry point for the FastAPI application.

Vercel's Python runtime imports the ASGI application from ``api/index.py``.
The ``backend/`` directory (containing the ``app`` package) is added to
``sys.path`` because this repo is a monorepo (backend/ + frontend/).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

# Object storage defaults to Vercel Blob when running on Vercel. Override via
# the STORAGE_PROVIDER env var (e.g. "local" for local dev).
os.environ.setdefault("STORAGE_PROVIDER", "vercel_blob")

from app.main import app  # noqa: E402  (Vercel ASGI entrypoint)

# TEMP diagnostic: surface blob write error details (remove before final).
import traceback  # noqa: E402

from app.services import storage  # noqa: E402
from starlette.routing import Route  # noqa: E402


# Register before app.main's StaticFiles mount ("/") so it is reachable.
async def _blobtest(request):  # noqa: E402
    info = {"provider": storage._provider()}
    try:
        import vercel_blob
        info["sdk_version"] = getattr(vercel_blob, "__version__", "?")
    except Exception as exc:
        info["sdk_import_error"] = str(exc)
    info["token_in_env"] = bool(os.environ.get("BLOB_READ_WRITE_TOKEN"))
    info["store_id_in_env"] = os.environ.get("BLOB_STORE_ID")
    try:
        key = storage._blob_put("uploads/__test_probe.txt", b"hello from probe")
        info["put_ok"] = True
        info["key"] = key
    except Exception as exc:
        info["put_ok"] = False
        info["put_error"] = str(exc)
        info["traceback"] = traceback.format_exc()[-2000:]
    return info


app.routes.insert(0, Route("/api/__blobtest__", _blobtest, methods=["GET"]))

