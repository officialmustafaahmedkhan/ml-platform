"""AI Assistant: chat about the user's datasets and models in plain language.

Reuses the provider-agnostic LLM client. The assistant receives a compact
context snapshot (dataset schema/profile and/or model metrics) and answers
the user's question with that context.
"""
from __future__ import annotations

import json
from typing import Optional

from ..utils.serialization import loads
from . import llm as llm_svc


def build_dataset_context(ds, df, profile: dict | None = None) -> dict:
    """Compact, prompt-friendly summary of a dataset."""
    if profile is None:
        profile = loads(ds.profile) if ds.profile else {}
    profile = profile or {}
    return {
        "name": ds.name,
        "rows": int(ds.rows or df.shape[0]),
        "columns": [str(c) for c in df.columns],
        "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
        "missing": profile.get("missing_pct"),
        "class_counts": profile.get("class_counts"),
        "num_classes": profile.get("num_classes"),
        "target_warnings": profile.get("target_warnings") or [],
        "sample_rows": df.head(8).fillna("").astype(str).to_dict(orient="records"),
    }


def build_model_context(model) -> dict:
    """Compact summary of a trained model artifact."""
    metrics = loads(model.metrics) if isinstance(model.metrics, str) else (model.metrics or {})
    params = loads(model.params) if isinstance(model.params, str) else (model.params or {})
    return {
        "name": model.name,
        "model_type": model.model_type,
        "params": params,
        "metrics": {
            k: v for k, v in (metrics or {}).items()
            if k not in ("confusion_matrix", "classification_report", "charts")
        },
        "feature_names": model.feature_names if isinstance(model.feature_names, list) else (loads(model.feature_names) if model.feature_names else []),
        "class_names": model.class_names if isinstance(model.class_names, list) else (loads(model.class_names) if model.class_names else []),
    }


def ask(message: str, *, dataset_context: Optional[dict] = None,
        model_context: Optional[dict] = None, history: Optional[list] = None) -> dict:
    """Send a chat message to the assistant. Returns the assistant reply + context used."""
    llm = llm_svc.get_llm()
    if llm is None:
        raise llm_svc.LLMError("LLM is disabled (set LLM_PROVIDER=openai or ollama)")

    context = {"dataset": dataset_context, "model": model_context}

    system = (
        "You are the AI assistant for an ML platform. Answer clearly and helpfully. "
        "You may use the provided context about the user's dataset and model. "
        "Be concise but specific; if the context does not contain an answer, say so. "
        "Never invent numbers that are not in the context."
    )

    messages: list[dict] = [{"role": "system", "content": system}]

    if dataset_context is not None:
        messages.append({
            "role": "system",
            "content": f"The user's dataset:\n{json.dumps(dataset_context, default=str)}",
        })
    if model_context is not None:
        messages.append({
            "role": "system",
            "content": f"The user's trained model:\n{json.dumps(model_context, default=str)}",
        })
    if history:
        for turn in history[-6:]:
            messages.append({"role": "user" if turn.get("role") == "user" else "assistant",
                             "content": turn.get("content", "")})

    messages.append({"role": "user", "content": message})

    reply = llm.chat(messages, temperature=0.3, max_tokens=1500)
    return {"reply": reply, "context": context}
