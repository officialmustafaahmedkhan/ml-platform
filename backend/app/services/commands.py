"""Natural-language ML command engine (rule-based, works offline).

Parses short English commands like ``train random forest on dataset 10 with
target Outcome`` into structured actions and executes them against the same
services used by the REST routes. The real LLM is only used for the optional
``label <dataset>`` intent; everything else is deterministic.
"""
from __future__ import annotations

import re

import pandas as pd
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..config import BASE_DIR, settings
from ..models import Dataset, ModelArtifact, User
from ..schemas import LLMLabelRequest, TrainRequest
from ..services import eda as eda_svc
from ..services import explain as explain_svc
from ..services import llm as llm_svc
from ..services import models as msvc
from ..services import preprocessing as pp
from ..services import storage
from ..services.pipeline import load_dataset_df, resolve_target
from ..services.training import train_artifact
from ..utils.serialization import dumps, loads

MODEL_ALIASES: dict[str, str] = {
    "random forest": "rf", "randomforest": "rf", "rf": "rf",
    "decision tree": "dt", "decisiontree": "dt", "tree": "dt", "dt": "dt",
    "k-nearest neighbors": "knn", "k-nearest": "knn", "knn": "knn",
    "voting": "voting", "voting classifier": "voting",
    "stacking": "stacking", "stack": "stacking",
}

_COMMANDS = {
    "help": "Show this help",
    "list datasets": "List your uploaded datasets",
    "list models": "List your trained models",
    "profile <dataset>": "Data quality profile + insights for a dataset",
    "eda on <dataset>": "Interactive exploration stats (correlations, distributions)",
    "train <model> on <dataset> [with target <col>]": "Train a model (e.g. random forest, decision tree, knn, voting, stacking)",
    "explain model <id|name>": "Global feature importance for a trained model",
    "compare <model> and <model>": "Compare metrics of two trained models",
    "label <dataset> with <n> categories": "Have the LLM design and write a target column",
}


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _clean(t: str) -> str:
    return " ".join(t.strip().lower().split())


def _alias_to_type(token: str) -> str | None:
    token = token.strip()
    if token in MODEL_ALIASES:
        return MODEL_ALIASES[token]
    for key in sorted(MODEL_ALIASES, key=len, reverse=True):
        if token.startswith(key):
            return MODEL_ALIASES[key]
    return None


def parse_command(text: str) -> dict:
    t = _clean(text)
    if not t:
        return {"action": "unknown", "text": text}

    if re.match(r"^(help|what can you do|what commands|commands|usage)$", t):
        return {"action": "help", "text": text}

    if re.search(r"\b(list|show)\b.*\bmodels?\b", t):
        return {"action": "list_models", "text": text}
    if re.search(r"\b(list|show)\b.*\bdatasets?\b", t):
        return {"action": "list_datasets", "text": text}

    m = re.search(
        r"label\s+(?:the\s+)?(?:dataset\s+)?#?([\w.\-]+)"
        r"(?:\s+(?:with|using)\s+(?:the\s+)?(?:llm|ai))?"
        r"(?:\s+with)?\s*(\d+)?\s*(?:categories|classes|labels)?"
        r"(?:\s+as\s+([\w.\-]+))?",
        t,
    )
    if m:
        return {
            "action": "llm_label",
            "text": text,
            "dataset": m.group(1),
            "num_categories": int(m.group(2)) if m.group(2) else 3,
            "column": m.group(3),
        }

    m = re.search(
        r"train\s+(?:a|an|the|one)?\s*([a-z0-9\-/ ]+?)\s+(?:on|for|using|with)\s+"
        r"(?:the\s+)?(?:dataset\s+)?#?([\w.\-]+)"
        r"(?:\s+with\s+target\s+([a-zA-Z_][\w.\-]*))?",
        t,
    )
    if m:
        model_text = m.group(1).strip().rstrip("model").strip()
        return {
            "action": "train",
            "text": text,
            "model_type": _alias_to_type(model_text),
            "dataset": m.group(2),
            "target": m.group(3),
        }

    m = re.search(
        r"explain\s+(?:the\s+)?(?:model\s+)?#?([\w.\-]+)", t,
    )
    if m:
        return {"action": "explain", "text": text, "model": m.group(1)}

    m = re.search(
        r"compare\s+(?:model\s+)?#?([\w.\-]+)\s+(?:and|vs\.?|with)\s+(?:model\s+)?#?([\w.\-]+)",
        t,
    )
    if m:
        return {"action": "compare", "text": text, "model_a": m.group(1), "model_b": m.group(2)}

    m = re.search(
        r"(?:eda|analy[sz]e|explore)\s+(?:(?:on|for)\s+)?(?:the\s+)?(?:dataset\s+)?#?([\w.\-]+)", t,
    )
    if m:
        return {"action": "eda", "text": text, "dataset": m.group(1)}

    m = re.search(
        r"(?:profile|describe|summar[yi]ze)\s+(?:the\s+)?(?:dataset\s+)?#?([\w.\-]+)", t,
    )
    if m:
        return {"action": "profile", "text": text, "dataset": m.group(1)}

    return {"action": "unknown", "text": text}


# --------------------------------------------------------------------------- #
# Reference resolution
# --------------------------------------------------------------------------- #
def resolve_dataset(db: Session, user: User, ref: str | None) -> Dataset:
    q = db.query(Dataset).filter(Dataset.user_id == user.id)
    if ref is None:
        return q.order_by(Dataset.created_at.desc()).first()
    m = re.match(r"^#?(\d+)$", str(ref).strip())
    if m:
        return q.filter(Dataset.id == int(m.group(1))).first()
    return q.filter(func.lower(Dataset.name) == str(ref).lower()) \
        .order_by(Dataset.created_at.desc()).first()


def resolve_model(db: Session, user: User, ref: str | None) -> ModelArtifact:
    q = db.query(ModelArtifact).filter(ModelArtifact.user_id == user.id)
    if ref is None:
        return q.order_by(ModelArtifact.created_at.desc()).first()
    m = re.match(r"^#?(\d+)$", str(ref).strip())
    if m:
        return q.filter(ModelArtifact.id == int(m.group(1))).first()
    return q.filter(func.lower(ModelArtifact.name).contains(str(ref).lower())) \
        .order_by(ModelArtifact.created_at.desc()).first()


def _model_metrics(m: ModelArtifact) -> dict:
    return loads(m.metrics).get("metrics", {}) if m.metrics else {}


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #
def execute_command(parsed: dict, db: Session, user: User) -> dict:
    action = parsed.get("action")
    handler = {
        "help": _exec_help,
        "list_datasets": _exec_list_datasets,
        "list_models": _exec_list_models,
        "profile": _exec_profile,
        "eda": _exec_eda,
        "train": _exec_train,
        "explain": _exec_explain,
        "compare": _exec_compare,
        "llm_label": _exec_llm_label,
    }.get(action)

    if handler is None:
        return {
            "action": "unknown",
            "summary": f"Sorry, I couldn't understand: “{parsed.get('text', '')}”. "
                       "Try “help” to see what I can do.",
            "cards": [], "rows": None, "text": None, "detail": None,
        }
    return handler(parsed, db, user)


def _exec_help(parsed: dict, db: Session, user: User) -> dict:
    rows = [{"command": k, "description": v} for k, v in _COMMANDS.items()]
    return {
        "action": "help",
        "summary": "I can run ML tasks from plain English. Try these:",
        "cards": [], "rows": rows, "text": None, "detail": None,
    }


def _exec_list_datasets(parsed: dict, db: Session, user: User) -> dict:
    dss = db.query(Dataset).filter(Dataset.user_id == user.id) \
        .order_by(Dataset.created_at.desc()).all()
    rows = [
        {
            "id": d.id,
            "name": d.name,
            "rows": d.rows,
            "columns": len(loads(d.columns)) if d.columns else 0,
        }
        for d in dss
    ]
    return {
        "action": "list_datasets",
        "summary": f"You have {len(dss)} dataset(s).",
        "cards": [{"label": "Datasets", "value": len(dss)}],
        "rows": rows, "text": None, "detail": None,
    }


def _exec_list_models(parsed: dict, db: Session, user: User) -> dict:
    ms = db.query(ModelArtifact).filter(ModelArtifact.user_id == user.id) \
        .order_by(ModelArtifact.created_at.desc()).all()
    rows = []
    for m in ms:
        metrics = _model_metrics(m)
        rows.append({
            "id": m.id,
            "name": m.name,
            "model_type": m.model_type,
            "accuracy": round(metrics.get("accuracy", 0.0), 4),
            "f1": round(metrics.get("f1_macro", 0.0), 4),
            "dataset_id": m.dataset_id,
        })
    return {
        "action": "list_models",
        "summary": f"You have {len(ms)} trained model(s).",
        "cards": [{"label": "Models", "value": len(ms)}],
        "rows": rows, "text": None, "detail": None,
    }


def _exec_profile(parsed: dict, db: Session, user: User) -> dict:
    ds = resolve_dataset(db, user, parsed.get("dataset"))
    if ds is None:
        raise ValueError("Dataset not found — try “list datasets” first.")
    _, df = load_dataset_df(db, ds.id, user.id)
    target = resolve_target(df, None)
    profile = pp.profile_dataset(df, target)
    return {
        "action": "profile",
        "summary": f"Profile of “{ds.name}” — health score {profile['health_score']}/100.",
        "cards": [
            {"label": "Rows", "value": profile["rows"]},
            {"label": "Columns", "value": profile["columns"]},
            {"label": "Health", "value": f"{profile['health_score']}%"},
            {"label": "Missing", "value": f"{profile['missing_pct']}%"},
            {"label": "Target", "value": target or "—"},
            {"label": "Classes", "value": profile["num_classes"]},
        ],
        "rows": None,
        "text": "\n".join(f"• [{i['level']}] {i['title']}: {i['detail']}" for i in profile["insights"]),
        "detail": profile,
    }


def _exec_eda(parsed: dict, db: Session, user: User) -> dict:
    ds = resolve_dataset(db, user, parsed.get("dataset"))
    if ds is None:
        raise ValueError("Dataset not found — try “list datasets” first.")
    _, df = load_dataset_df(db, ds.id, user.id)
    target = resolve_target(df, None)
    payload = eda_svc.eda_dataset(df, target)
    pairs = payload.get("scatter_pairs", [])[:6]
    rows = [{"x": p["x"], "y": p["y"], "corr": p["corr"]} for p in pairs]
    dist = payload.get("class_counts", {})
    text = "Target distribution: " + (", ".join(f"{k}={v}" for k, v in dist.items()) if dist else "no target set")
    return {
        "action": "eda",
        "summary": f"Exploration of “{ds.name}” — {len(payload['numeric_columns'])} numeric, "
                   f"{len(payload['categorical_columns'])} categorical column(s).",
        "cards": [
            {"label": "Numeric", "value": len(payload["numeric_columns"])},
            {"label": "Categorical", "value": len(payload["categorical_columns"])},
            {"label": "Target", "value": target or "—"},
        ],
        "rows": rows if rows else None,
        "text": text,
        "detail": {"numeric_columns": payload["numeric_columns"],
                   "categorical_columns": payload["categorical_columns"],
                   "class_counts": dist,
                   "column_stats": payload["column_stats"]},
    }


def _exec_train(parsed: dict, db: Session, user: User) -> dict:
    model_type = parsed.get("model_type")
    if model_type is None:
        raise ValueError("Unknown model type. I know: random forest, decision tree, knn, voting, stacking.")
    ds = resolve_dataset(db, user, parsed.get("dataset"))
    if ds is None:
        raise ValueError("Dataset not found — try “list datasets” first.")

    payload = TrainRequest(
        dataset_id=ds.id,
        model_type=model_type,
        target_column=parsed.get("target"),
    )
    resp = train_artifact(db, user, payload)
    metrics = resp.metrics
    top = _top_importances(db, resp.model_id, user.id)
    return {
        "action": "train",
        "summary": f"Trained {resp.name} (model #{resp.model_id}) — accuracy {metrics.get('accuracy', 0):.3f}.",
        "cards": [
            {"label": "Model", "value": resp.model_type},
            {"label": "Accuracy", "value": f"{metrics.get('accuracy', 0) * 100:.1f}%"},
            {"label": "Precision", "value": f"{metrics.get('precision_macro', 0) * 100:.1f}%"},
            {"label": "Recall", "value": f"{metrics.get('recall_macro', 0) * 100:.1f}%"},
            {"label": "F1", "value": f"{metrics.get('f1_macro', 0) * 100:.1f}%"},
        ],
        "rows": top,
        "text": f"Test split: {resp.metrics.get('test_size', 0.2)} · tune: {payload.tune}",
        "detail": {"model_id": resp.model_id, "metrics": metrics,
                   "feature_names": resp.feature_names, "class_names": resp.class_names},
    }


def _top_importances(db: Session, model_id: int, user_id: int) -> list:
    m = db.query(ModelArtifact).filter(ModelArtifact.id == model_id, ModelArtifact.user_id == user_id).first()
    if m is None or not m.metrics:
        return []
    metrics = loads(m.metrics)
    imp = metrics.get("feature_importance") or {}
    # Stored shape is either {feature: value} or parallel lists {features: [...], importance: [...]}
    if isinstance(imp, dict) and isinstance(imp.get("features"), list):
        names = [str(n) for n in imp["features"]]
        vals = []
        for v in imp.get("importance", []):
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                vals.append(0.0)
        pairs = sorted(zip(names, vals), key=lambda kv: abs(kv[1]), reverse=True)[:8]
        return [{"feature": n, "importance": round(v, 4)} for n, v in pairs]
    if isinstance(imp, dict):
        def _mag(v):
            if isinstance(v, (list, tuple)):
                nums = []
                for x in v:
                    try:
                        nums.append(abs(float(x)))
                    except (TypeError, ValueError):
                        continue
                return max(nums) if nums else 0.0
            try:
                return abs(float(v))
            except (TypeError, ValueError):
                return 0.0

        def _val(v):
            if isinstance(v, (list, tuple)):
                nums = []
                for x in v:
                    try:
                        nums.append(float(x))
                    except (TypeError, ValueError):
                        continue
                return round(max(nums, default=0.0), 4)
            try:
                return round(float(v), 4)
            except (TypeError, ValueError):
                return 0.0

        items = sorted(imp.items(), key=lambda kv: _mag(kv[1]), reverse=True)[:8]
        return [{"feature": k, "importance": _val(v)} for k, v in items]
    return []
    return []


def _exec_explain(parsed: dict, db: Session, user: User) -> dict:
    m = resolve_model(db, user, parsed.get("model"))
    if m is None:
        raise ValueError("Model not found — try “list models” first.")
    model = storage.load_model_artifact(m.filepath)
    pipeline_meta = loads(m.pipeline) if m.pipeline else {}
    pipeline = storage.load_pipeline(pipeline_meta.get("path"))
    if not m.dataset_id:
        raise ValueError("Model has no source dataset for global analysis.")
    _, df = load_dataset_df(db, m.dataset_id, user.id)
    if df is None or df.empty:
        raise ValueError("Source dataset is empty.")
    results = explain_svc.permutation_importance(model, pipeline, df)
    if isinstance(results, list):
        rows = [{"feature": r["feature"], "drop": round(float(r["importance"]), 4)}
                for r in results][:8]
        detail = {"method": "permutation_importance", "importance": results}
        note = "mean accuracy drop when each feature is shuffled"
    else:
        rows = [{"feature": r["feature"], "drop": round(float(r["importance"]), 4)}
                for r in results.get("importance", [])][:8]
        detail = results
        note = results.get("baseline_note", "") if isinstance(results, dict) else ""
    return {
        "action": "explain",
        "summary": f"Global importance for “{m.name}” (model #{m.id}) via permutation importance.",
        "cards": [{"label": "Model", "value": m.model_type}],
        "rows": rows,
        "text": note,
        "detail": detail,
    }


def _exec_compare(parsed: dict, db: Session, user: User) -> dict:
    a = resolve_model(db, user, parsed.get("model_a"))
    b = resolve_model(db, user, parsed.get("model_b"))
    if a is None or b is None:
        raise ValueError("Could not find one of the two models — try “list models” first.")
    ma, mb = _model_metrics(a), _model_metrics(b)
    keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    rows = [{"metric": k.replace("_macro", " (macro)").title(), "model_a": ma.get(k), "model_b": mb.get(k)}
            for k in keys]
    return {
        "action": "compare",
        "summary": f"Comparing “{a.name}” (#{a.id}) vs “{b.name}” (#{b.id}).",
        "cards": [],
        "rows": rows,
        "text": None,
        "detail": {"a": {"id": a.id, "name": a.name}, "b": {"id": b.id, "name": b.name}},
    }


def _exec_llm_label(parsed: dict, db: Session, user: User) -> dict:
    if not settings.LLM_ENABLED:
        raise ValueError("LLM is disabled. Set LLM_PROVIDER=openai or ollama in the environment / .env.")
    ds = resolve_dataset(db, user, parsed.get("dataset"))
    if ds is None:
        raise ValueError("Dataset not found — try “list datasets” first.")
    req = LLMLabelRequest(num_categories=parsed.get("num_categories", 3),
                          column_name=parsed.get("column") or "Outcome")
    _, df = load_dataset_df(db, ds.id, user.id)

    column = req.column_name
    if column in df.columns:
        raise ValueError(f"Column '{column}' already exists in this dataset.")

    try:
        categories = llm_svc.propose_outcomes(df, req.num_categories)
        names = [c["name"] for c in categories]
        labels = llm_svc.label_rows(df, names, batch_size=req.batch_size, max_rows=req.max_rows)
    except llm_svc.LLMError as exc:
        raise ValueError(str(exc))

    out = df.copy()
    out[column] = labels[: len(out)]
    for c in (column,):
        out[c] = out[c].astype(object)

    path = BASE_DIR / "data" / "uploads" / f"{ds.name.rsplit('.', 1)[0]}_llm_{column.lower()}.csv"
    out.to_csv(path, index=False)

    counts = out[column].value_counts(dropna=False).astype(int).to_dict()
    counts = {str(k): int(v) for k, v in counts.items()}
    labeled_rows = int(sum(v for k, v in counts.items() if k != "Unknown"))

    profile = pp.profile_dataset(out)
    prev_versions = loads(ds.versions) if ds.versions else []
    prev_versions = prev_versions if isinstance(prev_versions, list) else []
    prev_versions.append({"version": len(prev_versions) + 1, "path": str(path),
                          "rows": int(out.shape[0]), "note": f"LLM-generated '{column}' column"})

    ds.filepath = str(path)
    ds.rows = int(out.shape[0])
    ds.columns = dumps([str(c) for c in out.columns])
    ds.preview = dumps(out.head(10).fillna("").astype(str).to_dict(orient="records"))
    ds.profile = dumps(profile)
    ds.versions = dumps(prev_versions)
    db.add(ds)
    db.commit()
    db.refresh(ds)

    rows = [{"category": k, "count": v} for k, v in counts.items()]
    return {
        "action": "llm_label",
        "summary": f"Labeled {labeled_rows} rows of “{ds.name}” with column “{column}” ({len(names)} categories).",
        "cards": [{"label": "Labeled", "value": labeled_rows}, {"label": "Categories", "value": len(names)}],
        "rows": rows,
        "text": "New column written back to the dataset as a new version.",
        "detail": {"dataset_id": ds.id, "column_name": column, "categories": categories, "counts": counts},
    }
