"""Live smoke test against a running uvicorn server (not TestClient).

Drives the full user journey over real HTTP: register/login, upload,
preprocess, train all 5 model types, evaluate, predict single+batch,
compare, recommend, dashboard, export.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:8000"
SAMPLE = Path(__file__).resolve().parent.parent / "sample_data" / "play_tennis.csv"

passed = 0
failed = []


def check(name, cond, detail=""):
    global passed
    if cond:
        passed += 1
        print(f"[ok] {name}")
    else:
        failed.append(name)
        print(f"[FAIL] {name} {detail}")


def main():
    c = httpx.Client(base_url=BASE, timeout=60)

    r = c.get("/api/health")
    check("health", r.status_code == 200, r.text[:200])

    import time
    email = f"live_{int(time.time())}@example.com"
    r = c.post("/api/auth/register", json={"name": "Live", "email": email, "password": "secret123"})
    check("register", r.status_code == 201, r.text[:200])
    reg = r.json()
    check("register needs verification", reg.get("needs_verification") is True, str(reg)[:200])
    token = None
    if reg.get("dev_otp"):
        r = c.post("/api/auth/verify-otp", json={"email": email, "otp": reg["dev_otp"]})
        check("verify-otp", r.status_code == 200, r.text[:200])
        token = r.json().get("access_token")
    if not token:
        r = c.post("/api/auth/login", json={"email": email, "password": "secret123"})
        token = r.json()["access_token"]
    h = {"Authorization": f"Bearer {token}"}
    check("auth token", bool(token))

    r = c.get("/api/dashboard", headers=h)
    check("dashboard empty", r.status_code == 200, r.text[:200])

    with open(SAMPLE, "rb") as fh:
        r = c.post("/api/datasets/upload", headers=h,
                   files={"file": ("play_tennis.csv", fh, "text/csv")})
    check("upload", r.status_code == 201, r.text[:300])
    ds = r.json()
    dataset_id = ds["id"]
    check("dataset profile", "rows" in ds and ds["rows"] == 14, str(ds)[:300])

    r = c.get(f"/api/datasets/{dataset_id}/head", headers=h)
    check("dataset head", r.status_code == 200, r.text[:200])

    r = c.post("/api/preprocess", headers=h,
               json={"dataset_id": dataset_id, "mode": "auto", "target_column": "Play"})
    check("preprocess auto", r.status_code == 200, r.text[:300])

    r = c.get(f"/api/recommend/{dataset_id}", headers=h)
    check("recommendations", r.status_code == 200, r.text[:300])

    all_ids = []
    for mtype, params in [("dt", {"max_depth": 4}),
                          ("knn", {"n_neighbors": 3}),
                          ("rf", {"n_estimators": 40, "max_depth": 4}),
                          ("voting", {"n_estimators": 40, "max_depth": 4, "n_neighbors": 3}),
                          ("stacking", {"n_estimators": 40, "max_depth": 4, "n_neighbors": 3})]:
        r = c.post("/api/train", headers=h,
                   json={"dataset_id": dataset_id, "model_type": mtype,
                         "target_column": "Play", "params": params,
                         "preprocess": {"mode": "auto"}, "test_size": 0.2})
        ok = r.status_code == 200 and r.json().get("model_id")
        check(f"train {mtype}", ok, r.text[:300])
        if ok:
            all_ids.append(r.json()["model_id"])

    mid = all_ids[0]
    r = c.get(f"/api/evaluate/{mid}", headers=h)
    ev = r.json()
    check("evaluate", r.status_code == 200 and ev["charts"]["confusion_matrix"].startswith("data:image/png"), r.text[:200])
    chart_keys = [k for k in ("roc_curve", "precision_recall", "learning_curve", "class_balance", "predicted_vs_actual", "probability_histogram") if ev["charts"].get(k)]
    check("evaluate prediction charts", len(chart_keys) >= 4, str(chart_keys)[:200])

    r = c.post("/api/predict", headers=h,
               json={"model_id": mid,
                     "input": {"Outlook": "Sunny", "Temperature": "Mild", "Humidity": "High", "Wind": "Weak"}})
    pred = r.json()
    check("predict single", r.status_code == 200 and pred["prediction"] in ("Yes", "No"), r.text[:300])

    csv_text = "Outlook,Temperature,Humidity,Wind,Play\nSunny,Mild,High,Weak,Yes\nOvercast,Cool,Normal,Strong,Yes\n"
    r = c.post("/api/predict/batch", params={"model_id": mid}, headers=h,
               files={"file": ("batch.csv", csv_text.encode(), "text/csv")})
    check("predict batch", r.status_code == 200 and r.json()["total"] == 2, r.text[:300])

    r = c.post("/api/compare", headers=h, json={"dataset_id": dataset_id, "include_hybrid": True})
    check("compare", r.status_code == 200 and len(r.json()["table"]) == 4, r.text[:300])
    r = c.post("/api/compare", headers=h, json={"dataset_id": dataset_id, "model_types": ["dt", "knn"]})
    check("compare model selection", r.status_code == 200 and {x["model_type"] for x in r.json()["table"]} == {"dt", "knn"}, r.text[:300])

    r = c.get(f"/api/export/model/{mid}", headers=h)
    check("export pickle", r.status_code == 200 and r.content[:2] == b"\x80\x05", "")

    r = c.get(f"/api/export/report/{mid}", headers=h)
    check("export report", r.status_code == 200 and b"accuracy" in r.content, "")

    r = c.get("/api/dashboard", headers=h)
    d = r.json()
    check("dashboard populated", r.status_code == 200 and d.get("stats", {}).get("models") >= 5, r.text[:300])

    r = c.get("/api/train/models", headers=h)
    check("history/models", r.status_code == 200 and len(r.json()) >= 5, r.text[:300])

    print(f"\n{passed} checks passed, {len(failed)} failed")
    if failed:
        print("failed:", failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
