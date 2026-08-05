"""Pytest fixtures: isolated temp SQLite DB + authenticated TestClient.

NOTE (environment): Windows Smart App Control can reputation-block pandas'
compiled DLLs (transient — cleared once the files gain reputation). The suite
runs anywhere pandas imports normally (Linux/macOS/CI/other Windows).

SMTP is forced off (dev mode) so the OTP flow returns the code in the response
and the suite works without a real mail server.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

# Must be set BEFORE any `app.*` import so config picks up the temp DB.
_tmp_db = Path(tempfile.mkdtemp(prefix="ml_platform_")) / "test.db"
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp_db}"
os.environ["SMTP_HOST"] = ""  # dev mode -> OTP returned in responses
os.environ["SMTP_PORT"] = "0"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


def verify_account(client, email: str) -> dict:
    """Register-time helper: fetch the dev OTP and verify the account."""
    res = client.post("/api/auth/send-otp", json={"email": email})
    assert res.status_code == 200, res.text
    dev_otp = res.json().get("dev_otp")
    assert dev_otp, "Expected dev-mode OTP in the response"
    res = client.post("/api/auth/verify-otp", json={"email": email, "otp": dev_otp})
    assert res.status_code == 200, res.text
    return res.json()


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def auth_headers(client):
    res = client.post(
        "/api/auth/register",
        json={"name": "Tester", "email": "tester@example.com", "password": "secret123"},
    )
    assert res.status_code == 201, res.text
    data = verify_account(client, res.json()["email"])
    return {"Authorization": f"Bearer {data['access_token']}"}


@pytest.fixture(scope="session")
def dataset_id(client, auth_headers):
    sample = Path(__file__).resolve().parent.parent.parent / "sample_data" / "play_tennis.csv"
    with open(sample, "rb") as fh:
        res = client.post(
            "/api/datasets/upload",
            headers=auth_headers,
            files={"file": ("play_tennis.csv", fh, "text/csv")},
        )
    assert res.status_code == 201, res.text
    return res.json()["id"]


@pytest.fixture(scope="session")
def make_batch_csv(client, auth_headers, dataset_id):
    """Return a callable producing a valid batch-CSV for the sample dataset."""

    def _make() -> bytes:
        head = client.get(f"/api/datasets/{dataset_id}/head", headers=auth_headers)
        cols = head.json()["columns"]
        row = {c: head.json()["preview"][0][c] for c in cols}
        import csv
        import io

        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=cols)
        writer.writeheader()
        writer.writerow(row)
        writer.writerow(row)
        return buf.getvalue().encode()

    return _make
