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

