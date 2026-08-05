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

# TEMP DIAGNOSTIC: dump what the ASGI app actually receives for a POST body.
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route


async def _diag_echo(request: Request) -> JSONResponse:
    body = await request.body()
    headers = dict(request.headers)
    try:
        form = await request.form()
        form_keys = list(form.keys())
    except Exception as exc:  # noqa: BLE001
        form_keys = f"form error: {exc!r}"
    return JSONResponse({
        "method": request.method,
        "content_type": headers.get("content-type"),
        "content_length_header": headers.get("content-length"),
        "raw_body_len": len(body),
        "raw_body_head": body[:200].decode("utf-8", "replace"),
        "form_keys": form_keys,
    })


app.routes.insert(0, Route("/api/_diag", _diag_echo, methods=["POST"]))



