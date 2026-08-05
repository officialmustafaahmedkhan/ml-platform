"""Non-pandas backend smoke test (runs even where pandas DLLs are blocked).

Covers: pydantic schemas, JWT/security, SQLAlchemy model validation,
sklearn model building for all 5 model types, and matplotlib chart output.
Run: python scripts/smoke_no_pandas.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import numpy as np
from app.schemas import (
    AuthResponse, ComparisonResponse, DatasetOut, EvaluateResponse, PredictResponse,
    PreprocessResponse, RegisterRequest, TrainResponse, UserOut,
)
from app.utils.security import (
    create_access_token, decode_token, get_current_user, hash_password, verify_password,
)
from app.services.models import MODEL_REGISTRY, build_model
from app.services.evaluation import (
    chart_accuracy_comparison, chart_confusion_matrix, chart_feature_importance,
    chart_metric_radar, chart_tree, evaluate_model, feature_importance,
)
from app.models import User

# 1. Schemas parse
RegisterRequest(name="Ada", email="ada@example.com", password="secret123")
TrainResponse(
    model_id=1, model_type="rf", name="m", params={}, metrics={"accuracy": 0.9},
    feature_names=["a"], class_names=["x"], dataset_id=1,
)
DatasetOut(
    id=1, name="d", filename="d.csv", rows=10, columns=["a"], preview=[],
    profile={}, versions=[], created_at=__import__("datetime").datetime.now(),
)
UserOut(
    id=1, email="a@b.c", name="Ann", created_at=__import__("datetime").datetime.now(),
)
AuthResponse(
    access_token="t",
    user=UserOut(id=1, email="a@b.c", name="Ann", created_at=__import__("datetime").datetime.now()),
)
print("[ok] pydantic schemas")

# 2. Password hashing + JWT roundtrip
h = hash_password("secret123")
assert verify_password("secret123", h) and not verify_password("wrong", h)
token = create_access_token(7)
assert decode_token(token)["sub"] == "7"
print("[ok] security: hash + JWT")

# 3. SQLAlchemy UserOut validation
from datetime import datetime

u = User(
    id=1, email="a@b.c", name="Ann", password_hash=h, email_verified=False,
    preferences='{"theme":"dark"}', created_at=datetime.now(),
)
from app.schemas import UserOut

out = UserOut.model_validate(u)
assert out.preferences == {"theme": "dark"}
print("[ok] UserOut validation from ORM")

# 4. Model building for every registered type
rng = np.random.default_rng(1)
X = rng.normal(size=(80, 4))
y = (X[:, 0] + X[:, 1] > 0).astype(int)
for mt in MODEL_REGISTRY:
    model = build_model(mt, {}, 42)
    model.fit(X, y)
    assert model.predict(X[:3]).shape == (3,)
print("[ok] build+fit all 5 model types")

# 5. Evaluation + charts
from sklearn.tree import DecisionTreeClassifier

m = DecisionTreeClassifier(max_depth=4, random_state=42)
m.fit(X, y)
res = evaluate_model(m, X, y, ["neg", "pos"])
assert 0 <= res["metrics"]["accuracy"] <= 1
assert len(res["confusion_matrix"]) == 2
imp = feature_importance(m, ["f0", "f1", "f2", "f3"])
assert imp and len(imp["features"]) == 4
for uri in [
    chart_confusion_matrix(res["confusion_matrix"], ["neg", "pos"]),
    chart_feature_importance(imp),
    chart_tree(m, ["f0", "f1", "f2", "f3"], ["neg", "pos"]),
    chart_accuracy_comparison([{"model": "a", "accuracy": 0.8}, {"model": "b", "accuracy": 0.9}]),
    chart_metric_radar([{"model": "a", "accuracy": 0.8, "precision_weighted": 0.7, "recall_weighted": 0.75, "f1_weighted": 0.72}]),
]:
    assert uri.startswith("data:image/png;base64,")
print("[ok] evaluation metrics + 5 chart types")

print("\nAll non-pandas backend checks passed.")
