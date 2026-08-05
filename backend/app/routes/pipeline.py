"""Pipeline visualization routes.

Assemble a staged, human-readable view of each trained model's full pipeline:
data -> preprocessing -> training -> evaluation -> ready-to-predict. All the
information is already persisted on the ``ModelArtifact`` row (pipeline config,
hyperparameters, metrics); this route just restructures it for the UI.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Experiment, ModelArtifact, User
from ..services import models as msvc
from ..utils.serialization import loads
from ..utils.security import get_current_user

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])

_STAGE_ORDER = ["data", "preprocess", "train", "evaluate", "predict"]


def _prettify(key: str) -> str:
    """snake_case -> 'Title Case' for display."""
    return " ".join(p.capitalize() for p in key.replace("_", " ").split())


def _items_from_dict(d: dict, drop: tuple[str, ...] = ()) -> list[dict]:
    items = []
    for k, v in d.items():
        if k in drop or v is None:
            continue
        if isinstance(v, (dict, list)):
            continue
        items.append({"label": _prettify(k), "value": str(v)})
    return items


@router.get("/models")
def list_pipeline_models(user: User = Depends(get_current_user),
                         db: Session = Depends(get_db)):
    artifacts = db.query(ModelArtifact).filter(ModelArtifact.user_id == user.id) \
        .order_by(ModelArtifact.created_at.desc()).all()
    rows = []
    for m in artifacts:
        metrics = loads(m.metrics).get("metrics", {}) if m.metrics else {}
        rows.append({
            "id": m.id,
            "name": m.name,
            "model_type": m.model_type,
            "model_label": msvc.MODEL_REGISTRY.get(m.model_type, {}).get("label", m.model_type),
            "dataset_id": m.dataset_id,
            "dataset_name": m.dataset.name if m.dataset else None,
            "accuracy": round(metrics.get("accuracy", 0.0), 4),
            "f1": round(metrics.get("f1_macro", 0.0), 4),
            "created_at": m.created_at.isoformat() if m.created_at else None,
        })
    return {"models": rows}


@router.get("/{model_id}")
def get_pipeline(model_id: int, user: User = Depends(get_current_user),
                 db: Session = Depends(get_db)):
    m = db.query(ModelArtifact).filter(
        ModelArtifact.id == model_id, ModelArtifact.user_id == user.id
    ).first()
    if m is None:
        raise HTTPException(status_code=404, detail="Model not found")

    pipeline_meta = loads(m.pipeline) if m.pipeline else {}
    config = pipeline_meta.get("config") or {}
    target = pipeline_meta.get("target_column")
    params_meta = loads(m.params) if m.params else {}
    metrics_meta = loads(m.metrics) if m.metrics else {}
    metrics = metrics_meta.get("metrics", {}) or {}
    feature_names = loads(m.feature_names) if m.feature_names else []
    class_names = loads(m.class_names) if m.class_names else []
    importance = metrics_meta.get("feature_importance") or {}

    stages = []

    # 1. Data --------------------------------------------------------------- #
    data_items = [
        {"label": "Dataset", "value": m.dataset.name if m.dataset else "—"},
        {"label": "Dataset ID", "value": str(m.dataset_id)},
        {"label": "Rows", "value": str(m.dataset.rows) if m.dataset else "—"},
        {"label": "Columns", "value": str(len(loads(m.dataset.columns))) if m.dataset and m.dataset.columns else "—"},
        {"label": "Target column", "value": target or "—"},
        {"label": "Classes", "value": str(len(class_names))},
    ]
    stages.append({
        "stage": "data",
        "title": "Data",
        "subtitle": "Input dataset",
        "items": data_items,
        "details": {
            "columns": loads(m.dataset.columns) if m.dataset and m.dataset.columns else [],
            "class_names": class_names,
        },
    })

    # 2. Preprocessing ------------------------------------------------------ #
    pp_items = []
    config_copy = dict(config)
    if "target_column" in config_copy:
        config_copy.pop("target_column")
    pp_items = _items_from_dict(config_copy)
    if not pp_items:
        pp_items = [{"label": "Mode", "value": config.get("mode", "auto")}]
    stages.append({
        "stage": "preprocess",
        "title": "Preprocessing",
        "subtitle": "Strategies applied",
        "items": pp_items,
        "details": {"config": config},
    })

    # 3. Training ----------------------------------------------------------- #
    clean_params = params_meta.get("params") or {}
    cv_mean = params_meta.get("cv_mean_accuracy")
    try:
        cv_display = f"{float(cv_mean):.4f}"
    except (TypeError, ValueError):
        cv_display = "—"
    train_items = [
        {"label": "Model", "value": msvc.MODEL_REGISTRY.get(m.model_type, {}).get("label", m.model_type)},
        {"label": "Test size", "value": str(params_meta.get("test_size", 0.2))},
        {"label": "CV mean accuracy", "value": cv_display},
        {"label": "Tuned", "value": "Yes" if params_meta.get("tune") else "No"},
    ]
    train_items += [{"label": _prettify(k), "value": str(v)} for k, v in clean_params.items()]
    stages.append({
        "stage": "train",
        "title": "Training",
        "subtitle": "Hyperparameters",
        "items": train_items,
        "details": {"params": clean_params},
    })

    # 4. Evaluation --------------------------------------------------------- #
    metric_labels = {
        "accuracy": "Accuracy",
        "precision_macro": "Precision (macro)",
        "recall_macro": "Recall (macro)",
        "f1_macro": "F1 (macro)",
    }
    eval_items = []
    for k, label in metric_labels.items():
        v = metrics.get(k)
        if v is not None:
            try:
                eval_items.append({"label": label, "value": f"{float(v) * 100:.2f}%"})
            except (TypeError, ValueError):
                eval_items.append({"label": label, "value": str(v)})

    top_importance = []
    if isinstance(importance, dict) and isinstance(importance.get("features"), list):
        names = [str(n) for n in importance["features"]]
        vals = []
        for v in importance.get("importance", []):
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                vals.append(0.0)
        top_importance = [{"feature": n, "value": round(v, 4)} for n, v in list(zip(names, vals))[:8]]
    elif isinstance(importance, dict):
        items = []
        for k, v in importance.items():
            if isinstance(v, (list, tuple)):
                try:
                    items.append((str(k), float(max(v, default=0.0))))
                except (TypeError, ValueError):
                    continue
            else:
                try:
                    items.append((str(k), float(v)))
                except (TypeError, ValueError):
                    continue
        items.sort(key=lambda kv: abs(kv[1]), reverse=True)
        top_importance = [{"feature": k, "value": round(v, 4)} for k, v in items[:8]]

    stages.append({
        "stage": "evaluate",
        "title": "Evaluation",
        "subtitle": "Held-out test metrics",
        "items": eval_items,
        "details": {
            "metrics": metrics,
            "confusion_matrix": metrics_meta.get("confusion_matrix"),
            "class_names": class_names,
            "top_importance": top_importance,
        },
    })

    # 5. Ready to predict --------------------------------------------------- #
    predict_items = [
        {"label": "Features", "value": str(len(feature_names))},
        {"label": "Output classes", "value": ", ".join(str(c) for c in class_names) if class_names else "—"},
        {"label": "Artifact", "value": (m.filepath or "—").split("\\")[-1].split("/")[-1]},
        {"label": "Trained at", "value": m.created_at.isoformat() if m.created_at else "—"},
    ]
    stages.append({
        "stage": "predict",
        "title": "Ready to Predict",
        "subtitle": "Model artifact",
        "items": predict_items,
        "details": {
            "feature_names": feature_names,
            "class_names": class_names,
            "filepath": m.filepath,
        },
    })

    # Timeline (experiment log entries tied to this model) ------------------- #
    exps = db.query(Experiment).filter(Experiment.model_id == m.id) \
        .order_by(Experiment.created_at.asc()).all()
    timeline = [
        {
            "action": e.action,
            "created_at": e.created_at.isoformat() if e.created_at else None,
        }
        for e in exps
    ]

    return {
        "model": {
            "id": m.id,
            "name": m.name,
            "model_type": m.model_type,
            "model_label": msvc.MODEL_REGISTRY.get(m.model_type, {}).get("label", m.model_type),
            "created_at": m.created_at.isoformat() if m.created_at else None,
        },
        "stages": stages,
        "timeline": timeline,
    }
