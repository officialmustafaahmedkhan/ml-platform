"""Interactive EDA data service.

Computes lightweight statistics that the frontend renders with charting
libraries (histograms, correlations, distributions). Everything is returned
as plain JSON-friendly numbers so the UI stays interactive without images.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd


def _numeric_stats(s: pd.Series, name: str) -> dict:
    s = pd.to_numeric(s, errors="coerce")
    clean = s.dropna()
    if clean.empty:
        return {"kind": "numeric", "missing": int(s.isna().sum()), "n": 0}
    q = clean.quantile([0.25, 0.5, 0.75])
    iqr = float(q[0.75] - q[0.25])
    lo, hi = float(q[0.25] - 1.5 * iqr), float(q[0.75] + 1.5 * iqr)
    outliers = int(((clean < lo) | (clean > hi)).sum())
    return {
        "kind": "numeric",
        "n": int(clean.shape[0]),
        "missing": int(s.isna().sum()),
        "mean": round(float(clean.mean()), 4),
        "std": round(float(clean.std()), 4) if len(clean) > 1 else 0.0,
        "min": round(float(clean.min()), 4),
        "q1": round(float(q[0.25]), 4),
        "median": round(float(q[0.5]), 4),
        "q3": round(float(q[0.75]), 4),
        "max": round(float(clean.max()), 4),
        "skew": round(float(clean.skew()), 4) if len(clean) > 2 else 0.0,
        "outliers": outliers,
        "corr_signature": name,
    }


def _categorical_stats(s: pd.Series) -> dict:
    clean = s.dropna()
    counts = clean.value_counts()
    top = counts.head(8)
    return {
        "kind": "categorical",
        "n": int(clean.shape[0]),
        "missing": int(s.isna().sum()),
        "nunique": int(clean.nunique()),
        "top": str(top.index[0]) if len(top) else None,
        "top_count": int(top.iloc[0]) if len(top) else 0,
        "top_share": round(float(top.iloc[0] / max(len(clean), 1)), 4) if len(top) else 0.0,
        "categories": [{"value": str(k), "count": int(v)} for k, v in top.items()],
    }


def _histogram(s: pd.Series, bins: int = 20) -> dict:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return {"counts": [], "labels": [], "min": 0, "max": 0}
    lo, hi = float(s.min()), float(s.max())
    if hi <= lo:
        counts = [int(s.shape[0])]
        labels = [round(lo, 4)]
    else:
        counts, edges = np.histogram(s, bins=min(bins, max(1, int(s.nunique()))))
        labels = [round(float((edges[i] + edges[i + 1]) / 2), 3) for i in range(len(counts))]
    return {"counts": [int(c) for c in counts], "labels": labels, "min": round(lo, 4), "max": round(hi, 4)}


def eda_dataset(df: pd.DataFrame, target_column: str | None = None) -> dict:
    """Compute interactive-EDA payload for a dataset."""
    df = df.copy()
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    column_stats: dict[str, dict] = {}

    for col in df.columns:
        s = df[col]
        try:
            as_num = pd.to_numeric(s, errors="coerce")
            is_numeric = as_num.notna().mean() > 0.6 and as_num.nunique() > 1
        except Exception:  # noqa: BLE001
            is_numeric = False
        if is_numeric:
            numeric_cols.append(str(col))
            column_stats[str(col)] = _numeric_stats(s, str(col))
        else:
            categorical_cols.append(str(col))
            column_stats[str(col)] = _categorical_stats(s)

    histograms = {c: _histogram(df[c]) for c in numeric_cols}

    corr_features = numeric_cols[:12]
    corr_matrix: list[list] = []
    if len(corr_features) >= 2:
        sub = df[corr_features].apply(pd.to_numeric, errors="coerce")
        cm = sub.corr()
        corr_matrix = [[None if pd.isna(v) else round(float(v), 3) for v in row]
                       for row in cm.to_numpy()]
    else:
        corr_matrix = [[1.0]]

    # Candidate scatter pairs (top correlated pairs for quick exploration).
    scatter_pairs: list[dict] = []
    if len(corr_features) >= 2:
        sub = df[corr_features].apply(pd.to_numeric, errors="coerce")
        cm = sub.corr()
        pairs = []
        for i in range(len(corr_features)):
            for j in range(i + 1, len(corr_features)):
                v = cm.iloc[i, j]
                if pd.notna(v):
                    pairs.append((corr_features[i], corr_features[j], float(v)))
        pairs.sort(key=lambda p: abs(p[2]), reverse=True)
        scatter_pairs = [{"x": a, "y": b, "corr": round(c, 3)} for a, b, c in pairs[:6]]

    class_counts = {}
    if target_column and target_column in df.columns:
        counts = df[target_column].value_counts(dropna=True)
        class_counts = {str(k): int(v) for k, v in counts.items()}

    return {
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "column_stats": column_stats,
        "histograms": histograms,
        "correlation_features": corr_features,
        "correlation_matrix": corr_matrix,
        "scatter_pairs": scatter_pairs,
        "target_column": target_column,
        "class_counts": class_counts,
    }


def eda_payload_json(eda: dict) -> str:
    """Serialize the EDA payload (used when caching alongside the dataset)."""
    return json.dumps(eda, default=str)
