"""Model-agnostic explainability: LIME-style local attributions + permutation importance.

Implemented with sklearn only (no external explainer deps) so it stays robust
across environments. Works with any fitted classifier that exposes ``predict``
/ ``predict_proba`` and any preprocessing ``pipeline`` dict from this app.
"""
from __future__ import annotations

import random

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import pairwise_distances

from . import preprocessing as pp


def _feature_kind(col: str, pipeline: dict) -> str:
    if col in pipeline.get("encoders", {}):
        return "categorical"
    if col in pipeline.get("onehot_meta", {}):
        return "categorical"
    return "numeric"


def _perturb_row(base: dict, widths: dict, classes_map: dict, rng: random.Random) -> dict:
    """Return a perturbed copy of the base row dict."""
    row = {}
    for col, val in base.items():
        kind = classes_map.get(col)
        if kind is not None:  # categorical
            choices = kind
            row[col] = choices[rng.randrange(len(choices))]
        else:
            w = widths.get(col, 0.1)
            if val is None or (isinstance(val, float) and pd.isna(val)):
                row[col] = round(float(rng.gauss(0, w)), 6)
            else:
                try:
                    v = float(val)
                    row[col] = round(float(rng.gauss(v, max(w, 1e-6))), 6)
                except (TypeError, ValueError):
                    row[col] = val
    return row


def local_explanation(
    model,
    pipeline: dict,
    input_values: dict,
    feature_names: list[str],
    class_names: list[str],
    df: pd.DataFrame | None = None,
    n_perturbations: int = 250,
    random_seed: int = 42,
) -> dict:
    """Explain a single prediction with a locally-fitted linear surrogate.

    Returns ``{"prediction", "probabilities", "attributions"}`` where each
    attribution maps an encoded coefficient back to a raw feature column.
    """
    feature_cols = list(pipeline.get("feature_columns", feature_names))
    if not feature_cols:
        return {"prediction": None, "probabilities": {}, "attributions": []}

    # Base row (raw space).
    base = {}
    for col in feature_cols:
        val = input_values.get(col)
        if val is None or (isinstance(val, float) and pd.isna(val)) or val == "":
            base[col] = None
        else:
            base[col] = str(val)

    # Perturbation widths from the dataset (fall back to sensible defaults).
    widths: dict[str, float] = {}
    classes_map: dict[str, list] = {}
    for col in feature_cols:
        if _feature_kind(col, pipeline) == "categorical":
            known = None
            if col in pipeline.get("encoders", {}):
                known = pipeline["encoders"][col].get("classes", [])
            elif col in pipeline.get("onehot_meta", {}):
                known = pipeline["onehot_meta"][col].get("categories", [])
            classes_map[col] = [str(c) for c in known] if known else ["Unknown"]
        else:
            if df is not None and col in df.columns:
                numeric = pd.to_numeric(df[col], errors="coerce").dropna()
                std = float(numeric.std()) if len(numeric) > 1 else 0.0
                widths[col] = max(std * 0.1, 0.1)
            else:
                widths[col] = 0.1

    rng = random.Random(random_seed)
    rows = [_perturb_row(base, widths, classes_map, rng) for _ in range(n_perturbations)]
    rows.append(dict(base))

    raw = pd.DataFrame(rows, columns=feature_cols)
    X = pp.apply_preprocessor(raw, pipeline)
    X_base = X[[-1]] if X.shape[0] > 1 else X  # last row is the base instance

    if not hasattr(model, "predict_proba"):
        return {"prediction": None, "probabilities": {}, "attributions": []}

    probs = model.predict_proba(X)
    target_idx = int(np.argmax(probs[-1]))
    target_label = class_names[target_idx] if target_idx < len(class_names) else str(target_idx)

    # RBF-like kernel weights from feature-space distance to the base instance.
    dist = pairwise_distances(X, X_base, metric="euclidean").ravel()
    dist_scale = np.percentile(dist, 75) if len(dist) > 1 else 1.0
    weights = np.exp(-(dist ** 2) / (2 * (dist_scale ** 2) + 1e-9))

    y = probs[:, target_idx]
    model_x = Ridge(alpha=1.0)
    model_x.fit(X[:-1], y[:-1], sample_weight=weights[:-1])

    # Aggregate encoded coefficients back to raw feature columns.
    contribution: dict[str, float] = {}
    idx = 0
    for col in feature_cols:
        if col in pipeline.get("onehot_meta", {}):
            n = len(pipeline["onehot_meta"][col]["categories"])
            contribution[col] = float(np.sum(np.abs(model_x.coef_[idx:idx + n]))) / n
            idx += n
        else:
            contribution[col] = float(model_x.coef_[idx])
            idx += 1

    attributions = sorted(
        ({"feature": c, "contribution": round(v, 4),
          "direction": "toward" if v >= 0 else "against",
          "description": "supports" if v >= 0 else "opposes"}
         for c, v in contribution.items()),
        key=lambda a: abs(a["contribution"]),
        reverse=True,
    )

    probabilities = {class_names[i]: float(probs[-1][i]) for i in range(len(class_names))}
    return {
        "prediction": target_label,
        "probabilities": probabilities,
        "attributions": attributions,
    }


def _encode_target(work: pd.DataFrame, pipeline: dict) -> np.ndarray:
    """Encode the target column to the same class indices the model was trained on."""
    tcol = pipeline.get("target_column")
    te = pipeline.get("target_encoder")
    if te is None:
        return np.array([])
    classes = getattr(te, "classes_", None)
    if classes is None:
        classes = te.get("classes") if isinstance(te, dict) else None
    classes = [str(c) for c in classes] if classes is not None else []
    if tcol and tcol in work.columns and classes:
        cat = work[tcol].astype(str)
        return np.array([classes.index(v) if v in classes else -1 for v in cat])
    return np.array([])


def permutation_importance(
    model,
    pipeline: dict,
    df: pd.DataFrame,
    max_rows: int = 300,
    repeats: int = 3,
    random_seed: int = 42,
) -> list[dict]:
    """Global feature importance by shuffling each raw feature and re-scoring.

    Rows are capped for speed; only numeric features are considered.
    """
    work = df.copy() if len(df) <= max_rows else df.sample(max_rows, random_state=random_seed).copy()
    if work.empty:
        return []

    y = _encode_target(work, pipeline)
    if y.size == 0:
        return []

    feature_cols = list(pipeline.get("feature_columns", work.columns))
    X = pp.apply_preprocessor(work, pipeline)
    try:
        baseline = float(np.mean(model.predict(X) == y))
    except Exception:  # noqa: BLE001
        return []

    results = []
    for col in feature_cols:
        if col not in work.columns or _feature_kind(col, pipeline) != "numeric":
            continue
        scores = []
        for _ in range(repeats):
            perm = work.copy()
            perm[col] = perm[col].sample(frac=1.0, random_state=random_seed).values
            Xp = pp.apply_preprocessor(perm, pipeline)
            try:
                acc = float(np.mean(model.predict(Xp) == y))
            except Exception:  # noqa: BLE001
                acc = 0.0
            scores.append(acc)
        drop = baseline - float(np.mean(scores))
        results.append({"feature": col, "importance": round(max(drop, 0.0), 4)})

    results.sort(key=lambda r: r["importance"], reverse=True)
    return results
