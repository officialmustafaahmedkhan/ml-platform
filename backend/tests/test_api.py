"""End-to-end API smoke tests covering the full user journey.

Run with:  pytest backend/tests -v   (from the project root)
"""
from __future__ import annotations

import pytest

MODEL_TYPES = ["dt", "knn", "rf", "voting", "stacking"]


def test_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_register_login_me(client, auth_headers):
    res = client.get("/api/auth/me", headers=auth_headers)
    assert res.status_code == 200
    body = res.json()
    assert body["email"] == "tester@example.com"
    assert body["email_verified"] is True

    res = client.post(
        "/api/auth/login", json={"email": "tester@example.com", "password": "secret123"}
    )
    assert res.status_code == 200
    assert res.json()["access_token"]

    res = client.post(
        "/api/auth/register",
        json={"name": "Tester", "email": "tester@example.com", "password": "secret123"},
    )
    assert res.status_code == 409  # duplicate email


def test_otp_verification_flow(client):
    """Register returns no token; login is blocked until the OTP verifies."""
    res = client.post(
        "/api/auth/register",
        json={"name": "Otp User", "email": "otp@example.com", "password": "secret123"},
    )
    assert res.status_code == 201, res.text
    reg = res.json()
    assert reg["needs_verification"] is True
    assert reg["dev_otp"]  # dev mode returns the code
    assert "access_token" not in reg

    # Unverified login -> 403 with a structured detail + fresh OTP
    res = client.post(
        "/api/auth/login", json={"email": "otp@example.com", "password": "secret123"}
    )
    assert res.status_code == 403
    detail = res.json()["detail"]
    assert detail["code"] == "email_not_verified"
    fresh_otp = detail["dev_otp"]

    # Wrong OTP -> 400
    res = client.post(
        "/api/auth/verify-otp", json={"email": "otp@example.com", "otp": "000000"}
    )
    assert res.status_code == 400

    # Correct OTP -> token issued
    res = client.post(
        "/api/auth/verify-otp", json={"email": "otp@example.com", "otp": fresh_otp}
    )
    assert res.status_code == 200, res.text
    assert res.json()["access_token"]
    assert res.json()["user"]["email_verified"] is True

    # Verify is idempotent afterwards
    res = client.post(
        "/api/auth/verify-otp", json={"email": "otp@example.com", "otp": "000000"}
    )
    assert res.status_code == 200


def test_upload_list_get(client, auth_headers, dataset_id):
    res = client.get("/api/datasets", headers=auth_headers)
    assert res.status_code == 200
    ids = [d["id"] for d in res.json()]
    assert dataset_id in ids

    res = client.get(f"/api/datasets/{dataset_id}", headers=auth_headers)
    assert res.status_code == 200
    body = res.json()
    assert body["rows"] == 14
    assert "Play" in body["columns"]


def test_profile_and_auto_config(client, auth_headers, dataset_id):
    res = client.get(f"/api/preprocess/profile/{dataset_id}", headers=auth_headers)
    assert res.status_code == 200
    assert res.json()["profile"]["rows"] == 14

    res = client.post(
        f"/api/preprocess/auto-config/{dataset_id}",
        params={"target": "Play"},
        headers=auth_headers,
    )
    assert res.status_code == 200
    assert res.json()["config"]["target_column"] == "Play"


def test_preprocess_auto(client, auth_headers, dataset_id):
    res = client.post(
        "/api/preprocess",
        json={"dataset_id": dataset_id, "mode": "auto", "target_column": "Play"},
        headers=auth_headers,
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["report"]["feature_count"] >= 1
    assert body["recommendation"]["predicted_best_model"]["model_type"]


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_train_evaluate_predict(client, auth_headers, dataset_id, make_batch_csv, model_type):
    params = {}
    if model_type == "knn":
        params = {"n_neighbors": 3}
    if model_type == "rf":
        params = {"n_estimators": 50, "max_depth": 4}
    if model_type == "dt":
        params = {"max_depth": 4}
    if model_type in ("voting", "stacking"):
        params = {"n_estimators": 50, "max_depth": 4, "n_neighbors": 3}

    res = client.post(
        "/api/train",
        json={
            "dataset_id": dataset_id,
            "model_type": model_type,
            "target_column": "Play",
            "params": params,
            "preprocess": {"mode": "auto"},
            "test_size": 0.2,
        },
        headers=auth_headers,
    )
    assert res.status_code == 200, res.text
    body = res.json()
    model_id = body["model_id"]
    assert body["metrics"]["accuracy"] >= 0.0

    # Evaluate
    res = client.get(f"/api/evaluate/{model_id}", headers=auth_headers)
    assert res.status_code == 200
    ev = res.json()
    assert ev["metrics"]["accuracy"] == body["metrics"]["accuracy"]
    assert ev["charts"]["confusion_matrix"].startswith("data:image/png")
    # New prediction charts
    for chart_key in ("roc_curve", "precision_recall", "class_balance", "predicted_vs_actual", "probability_histogram"):
        assert ev["charts"].get(chart_key, "").startswith("data:image/png")

    # Single prediction
    res = client.post(
        "/api/predict",
        json={
            "model_id": model_id,
            "input": {"Outlook": "Sunny", "Temperature": "Mild", "Humidity": "High", "Wind": "Weak"},
        },
        headers=auth_headers,
    )
    assert res.status_code == 200, res.text
    pred = res.json()
    assert pred["prediction"] in ("Yes", "No")
    assert pred["probabilities"]

    # Batch prediction
    content = make_batch_csv()
    res = client.post(
        "/api/predict/batch",
        params={"model_id": model_id},
        headers=auth_headers,
        files={"file": ("batch.csv", content, "text/csv")},
    )
    assert res.status_code == 200, res.text
    batch = res.json()
    assert batch["total"] == 2
    assert "prediction" in batch["results"][0]

    # Export model pickle
    res = client.get(f"/api/export/model/{model_id}", headers=auth_headers)
    assert res.status_code == 200
    assert res.content[:2] in (b"\x80\x05", b"\x80\x04")  # pickle magic
    res = client.get(f"/api/export/report/{model_id}", headers=auth_headers)
    assert res.status_code == 200
    assert "accuracy" in res.text


def test_compare(client, auth_headers, dataset_id):
    res = client.post(
        "/api/compare",
        json={
            "dataset_id": dataset_id,
            "target_column": "Play",
            "preprocess": {"mode": "auto"},
            "include_hybrid": True,
        },
        headers=auth_headers,
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert len(body["table"]) >= 4
    assert body["best_model"]["accuracy"] == max(r["accuracy"] for r in body["table"])
    assert body["charts"]["accuracy_comparison"].startswith("data:image/png")
    assert body["charts"]["metric_radar"].startswith("data:image/png")

    # Explicit algorithm selection (hybrid off -> base models only)
    res = client.post(
        "/api/compare",
        json={
            "dataset_id": dataset_id,
            "target_column": "Play",
            "preprocess": {"mode": "auto"},
            "model_types": ["dt", "knn"],
        },
        headers=auth_headers,
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert {r["model_type"] for r in body["table"]} == {"dt", "knn"}

    # Invalid model type -> 400
    res = client.post(
        "/api/compare",
        json={"dataset_id": dataset_id, "target_column": "Play", "model_types": ["svm"]},
        headers=auth_headers,
    )
    assert res.status_code == 400


def test_recommendation(client, auth_headers, dataset_id):
    res = client.get(f"/api/recommend/{dataset_id}", params={"target": "Play"}, headers=auth_headers)
    assert res.status_code == 200
    body = res.json()
    assert body["model_recommendations"]
    assert body["preprocessing_recommendations"]
    assert body["predicted_best_model"]["model_type"]


def test_dashboard(client, auth_headers):
    res = client.get("/api/dashboard", headers=auth_headers)
    assert res.status_code == 200
    body = res.json()
    assert body["stats"]["datasets"] >= 1
    assert body["stats"]["models"] >= len(MODEL_TYPES)
    assert body["activity_timeline"]
    assert body["accuracy_trend"]


def test_models_list_and_delete(client, auth_headers):
    res = client.get("/api/train/models", headers=auth_headers)
    assert res.status_code == 200
    models = res.json()
    assert models
    res = client.delete(f"/api/train/models/{models[0]['id']}", headers=auth_headers)
    assert res.status_code == 204


def test_auth_required(client):
    res = client.get("/api/dashboard")
    assert res.status_code == 401


def test_preprocess_row_drop_keeps_xy_aligned():
    """Regression: X and y must stay the same length when rows are dropped."""
    import numpy as np
    import pandas as pd

    from app.services import preprocessing as pp

    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, np.nan],
            "b": ["x", "y", "x", "y", "x", "y", "x", "y"],
            "label": ["A", "B", "A", "B", "A", "B", "A", "B"],
        }
    )
    config = {"missing_numeric": "drop", "missing_categorical": "mode",
              "encoding": "label", "scaling": "none"}
    X, y, pipeline, report = pp.fit_preprocessor(df, "label", config)
    assert X.shape[0] == y.shape[0] == 6
    assert report["dropped_rows"] == 2
