"""Automated & manual data preprocessing engine.

Responsibilities
----------------
* Profile datasets (dtypes, missingness, cardinality, imbalance).
* Generate a sensible ``auto`` configuration from those heuristics.
* ``fit_preprocessor``: build + fit the full pipeline (imputation, encoding,
  scaling, optional SMOTE) and return (X, y) plus a serializable ``pipeline``
  dict used to reproduce the same transform on new prediction data.
* ``apply_preprocessor``: reuse a fitted pipeline on unseen data.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler

DEFAULT_RANDOM_STATE = 42

# --------------------------------------------------------------------------- #
# Column / dataset profiling
# --------------------------------------------------------------------------- #
def _is_numeric(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return True
    coerced = pd.to_numeric(series, errors="coerce")
    return coerced.notna().mean() >= 0.9


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    kind: str  # numeric | categorical
    missing: int
    missing_pct: float
    nunique: int
    sample: list = field(default_factory=list)


def profile_columns(df: pd.DataFrame) -> list[dict]:
    profiles = []
    for col in df.columns:
        s = df[col]
        kind = "numeric" if _is_numeric(s) else "categorical"
        missing = int(s.isna().sum())
        profiles.append(
            {
                "name": str(col),
                "dtype": str(s.dtype),
                "kind": kind,
                "missing": missing,
                "missing_pct": round(100.0 * missing / max(len(s), 1), 2),
                "nunique": int(s.nunique(dropna=True)),
                "sample": [str(v) for v in s.dropna().unique()[:5]],
            }
        )
    return profiles


def detect_target_warnings(df: pd.DataFrame, target_column: str) -> list[str]:
    """Heuristic warnings shown when the chosen target looks unsuitable for classification."""
    warnings: list[str] = []
    s = df[target_column].dropna()
    nunique = s.nunique()
    total = max(len(s), 1)

    if _is_numeric(s) and nunique > 20:
        ratio = nunique / total
        warnings.append(
            f"Column '{target_column}' has {nunique} distinct numeric values "
            f"(unique ratio {ratio:.2f}). This looks like a continuous variable and is "
            f"a poor fit for classification. Consider converting it to a regression "
            f"problem or binning it into fewer categories."
        )
    elif _is_numeric(s) and nunique > 10:
        warnings.append(
            f"Column '{target_column}' has {nunique} distinct numeric values. "
            f"If these represent ordered ranges rather than true categories, accuracy "
            f"may be misleading. Consider binning into fewer groups."
        )

    if len(s) < len(df):
        miss = len(df) - len(s)
        pct = miss / max(len(df), 1) * 100
        warnings.append(
            f"Target '{target_column}' has {miss} missing values ({pct:.1f}%). "
            f"Those rows will be dropped, leaving {len(s)} samples for training."
        )

    if nunique >= 2 and len(s) >= 10:
        counts = s.value_counts()
        imbalance = min(counts) / max(counts)
        if imbalance < 0.2:
            warnings.append(
                f"Target '{target_column}' is heavily imbalanced "
                f"(ratio {imbalance:.2f}). Consider enabling SMOTE or using "
                f"weighted metrics (precision/recall-weighted)."
            )

    return warnings


def profile_dataset(df: pd.DataFrame, target_column: Optional[str] = None) -> dict:
    """Lightweight dataset profile used by the recommendation engine."""
    cols = profile_columns(df)
    numeric_cols = [c["name"] for c in cols if c["kind"] == "numeric"]
    categorical_cols = [c["name"] for c in cols if c["kind"] == "categorical"]

    missing_total = int(df.isna().sum().sum())
    missing_pct = round(100.0 * missing_total / max(df.shape[0] * df.shape[1], 1), 2)

    target = target_column if target_column in df.columns else None
    class_counts: dict = {}
    imbalance_ratio: Optional[float] = None
    if target:
        counts = df[target].value_counts(dropna=True)
        class_counts = {str(k): int(v) for k, v in counts.items()}
        if len(counts) >= 2:
            imbalance_ratio = round(
                min(counts) / max(counts), 4
            )  # < 1 => imbalanced; closer to 0 = worse

    target_warnings: list[str] = []
    if target:
        target_warnings = detect_target_warnings(df, target)

    # Health score + automated insights -------------------------------------- #
    duplicates = int(df.duplicated().sum())
    dup_pct = 100.0 * duplicates / max(df.shape[0], 1)

    outlier_counts: dict[str, int] = {}
    for col in numeric_cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) >= 10:
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                outlier_counts[col] = int(((s < lo) | (s > hi)).sum())
    total_outliers = int(sum(outlier_counts.values()))

    high_card = [c["name"] for c in cols if c["kind"] == "categorical" and c["nunique"] > 10]
    imbalance = imbalance_ratio if imbalance_ratio is not None else 1.0

    health_score = 0.0
    health_score += max(0.0, 30.0 - missing_pct * 1.5)          # missing data: up to 30
    health_score += max(0.0, 30.0 - (1.0 - min(imbalance, 1.0)) * 37.5)  # imbalance: up to 30
    health_score += max(0.0, 20.0 - len(high_card) * 10.0)       # cardinality: up to 20
    health_score += max(0.0, 20.0 - dup_pct * 0.8)               # duplicates: up to 20
    health_score = round(min(100.0, max(0.0, health_score)), 1)

    insights: list[dict] = []
    if missing_pct > 20:
        insights.append({"level": "danger", "title": "High missing data",
                         "detail": f"{missing_pct:.1f}% of all cells are empty. Imputation is strongly recommended."})
    elif missing_pct > 5:
        insights.append({"level": "warn", "title": "Moderate missing data",
                         "detail": f"{missing_pct:.1f}% of cells are missing. Consider imputation."})
    elif missing_pct > 0:
        insights.append({"level": "good", "title": "Low missing data",
                         "detail": f"Only {missing_pct:.1f}% of cells are missing."})
    else:
        insights.append({"level": "good", "title": "No missing data",
                         "detail": "The dataset is complete — no imputation needed."})

    if imbalance < 0.2:
        insights.append({"level": "danger", "title": "Severe class imbalance",
                         "detail": f"Target ratio is {imbalance:.2f}. Enable SMOTE or class weighting."})
    elif imbalance < 0.6:
        insights.append({"level": "warn", "title": "Class imbalance",
                         "detail": f"Target ratio is {imbalance:.2f}. Weighted metrics may be more informative."})
    else:
        insights.append({"level": "good", "title": "Balanced target",
                         "detail": "Class distribution is reasonably balanced."})

    if high_card:
        insights.append({"level": "warn", "title": "High-cardinality categoricals",
                         "detail": f"{len(high_card)} categorical column(s) have >10 unique values: {', '.join(high_card[:4])}. Consider grouping rare categories."})
    else:
        insights.append({"level": "good", "title": "Clean categoricals",
                         "detail": "No high-cardinality categorical columns."})

    if dup_pct > 10:
        insights.append({"level": "warn", "title": "Duplicate rows",
                         "detail": f"{duplicates} duplicate rows ({dup_pct:.1f}%) found. Deduplication is recommended."})
    elif duplicates > 0:
        insights.append({"level": "good", "title": "Few duplicates",
                         "detail": f"{duplicates} duplicate rows detected."})
    else:
        insights.append({"level": "good", "title": "No duplicates",
                         "detail": "No fully duplicate rows were found."})

    if total_outliers > 0:
        top_out = sorted(outlier_counts.items(), key=lambda kv: kv[1], reverse=True)[:3]
        insights.append({"level": "warn", "title": "Outliers present",
                         "detail": f"{total_outliers} outlier cells across numeric columns (top: {', '.join(f'{c} ({n})' for c, n in top_out)})."})
    else:
        insights.append({"level": "good", "title": "No significant outliers",
                         "detail": "No IQR-based outliers detected in numeric columns."})

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "missing_total": missing_total,
        "missing_pct": missing_pct,
        "target_column": target,
        "target_classes": list(class_counts.keys()),
        "class_counts": class_counts,
        "num_classes": len(class_counts),
        "imbalance_ratio": imbalance_ratio,
        "high_cardinality_categoricals": high_card,
        "target_warnings": target_warnings,
        "duplicates": duplicates,
        "duplicate_pct": round(dup_pct, 2),
        "outlier_counts": outlier_counts,
        "health_score": health_score,
        "insights": insights,
    }


# --------------------------------------------------------------------------- #
# Auto configuration
# --------------------------------------------------------------------------- #
def auto_config(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    model_hint: Optional[str] = None,
    overrides: Optional[dict] = None,
) -> dict:
    """Build a recommended preprocessing config from dataset heuristics."""
    profile = profile_dataset(df, target_column)
    overrides = overrides or {}

    # Missing values ------------------------------------------------------- #
    missing_numeric = "median" if profile["missing_pct"] > 0 else "none"
    missing_categorical = "mode" if profile["missing_pct"] > 0 else "none"

    # Encoding -------------------------------------------------------------- #
    cat_cols = profile["categorical_columns"]
    if not cat_cols:
        encoding = "label"  # unused
    elif profile["high_cardinality_categoricals"]:
        encoding = "label"
    else:
        total_expansion = sum(
            1 + df[c].nunique(dropna=True) for c in cat_cols if c != target_column
        )
        encoding = "onehot" if total_expansion <= 30 else "label"

    # Scaling --------------------------------------------------------------- #
    scaling = "none"
    if model_hint == "knn":
        scaling = "standard"
    elif model_hint in ("dt", "rf", "voting", "stacking"):
        scaling = "none"
    elif not cat_cols or encoding == "label":
        scaling = "standard"

    # SMOTE ---------------------------------------------------------------- #
    smote = False
    ratio = profile["imbalance_ratio"]
    if ratio is not None and ratio < 0.35 and profile["rows"] >= 40:
        smote = True

    config = {
        "target_column": target_column,
        "missing_numeric": missing_numeric,
        "missing_categorical": missing_categorical,
        "encoding": encoding,
        "scaling": scaling,
        "smote": smote,
        "drop_columns": [],
        "random_state": DEFAULT_RANDOM_STATE,
    }
    # Model-aware overrides applied on top of heuristics
    for key, value in overrides.items():
        if key in config and value is not None:
            config[key] = value
    return config


# --------------------------------------------------------------------------- #
# Fit / apply pipeline
# --------------------------------------------------------------------------- #
def _clean_column_name(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_]+", "_", str(name)).strip("_") or "col"


def fit_preprocessor(
    df: pd.DataFrame,
    target_column: str,
    config: Optional[dict] = None,
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """Fit the preprocessing pipeline.

    Returns
    -------
    X : np.ndarray        feature matrix (float64)
    y : np.ndarray        encoded target (int)
    pipeline : dict       fitted transforms + metadata (picklable)
    report : dict         what happened (imputations, drops, SMOTE, ...)
    """
    config = config or {}
    report: dict = {
        "imputations": [],
        "dropped_rows": 0,
        "dropped_columns": [],
        "encoded": [],
        "scaled": config.get("scaling", "none"),
        "smote_applied": False,
        "smote_before": None,
        "smote_after": None,
    }

    df_work = df.copy()

    # Target ----------------------------------------------------------------- #
    if target_column not in df_work.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")

    feature_cols = [c for c in df_work.columns if c != target_column]
    drop_cols = list(config.get("drop_columns") or [])
    feature_cols = [c for c in feature_cols if c not in drop_cols]
    report["dropped_columns"] = drop_cols
    df_work = df_work.drop(columns=drop_cols, errors="ignore")

    missing_numeric = config.get("missing_numeric", "median")
    missing_categorical = config.get("missing_categorical", "mode")
    encoding = config.get("encoding", "label")
    scaling = config.get("scaling", "none")
    random_state = config.get("random_state", DEFAULT_RANDOM_STATE)

    # Drop rows with missing target (always safe)
    target_mask = df_work[target_column].notna()
    if not target_mask.all():
        report["dropped_rows"] += int((~target_mask).sum())
    df_work = df_work[target_mask].reset_index(drop=True)

    # Apply "drop" strategies as a single global row mask up-front so that all
    # transformed columns stay perfectly aligned.
    if missing_numeric == "drop":
        mask = df_work[feature_cols].apply(
            lambda c: pd.to_numeric(c, errors="coerce").notna() if _is_numeric(c) else pd.Series([True] * len(c))
        ).all(axis=1)
        report["dropped_rows"] += int((~mask).sum())
        df_work = df_work[mask].reset_index(drop=True)
    if missing_categorical == "drop":
        cat_cols = [c for c in feature_cols if not _is_numeric(df_work[c])]
        if cat_cols:
            mask = df_work[cat_cols].notna().all(axis=1)
            report["dropped_rows"] += int((~mask).sum())
            df_work = df_work[mask].reset_index(drop=True)

    # Encode the target AFTER row filtering so y stays aligned with X.
    target_encoder = LabelEncoder()
    y_raw = df_work[target_column].astype(str)
    y = target_encoder.fit_transform(y_raw)
    class_names = [str(c) for c in target_encoder.classes_]

    fitted_encoders: dict[str, Any] = {}
    onehot_meta: dict[str, dict] = {}
    fitted_scaler: Any = None
    column_order: list[str] = []
    transformed_cols: list[list[float]] = []

    for col in feature_cols:
        series = df_work[col]
        if _is_numeric(series):
            numeric = pd.to_numeric(series, errors="coerce")
            if missing_numeric and missing_numeric not in ("none", "drop") and numeric.isna().any():
                if missing_numeric in ("mean", "median", "mode"):
                    if missing_numeric == "mode":
                        fill = numeric.mode()
                        fill_value = float(fill.iloc[0]) if len(fill) else 0.0
                    else:
                        fill_value = float(getattr(numeric, missing_numeric)())
                else:
                    fill_value = 0.0
                numeric = numeric.fillna(fill_value)
                report["imputations"].append(f"{col} (numeric, {missing_numeric}={fill_value:.4g})")
            transformed_cols.append(numeric.astype(float).tolist())
            column_order.append(col)
        else:
            cat = series.astype(str)
            if missing_categorical == "constant":
                cat = series.astype(str).fillna("Missing")
            elif missing_categorical == "mode" and series.isna().any():
                mode_vals = series.mode()
                fill_value = str(mode_vals.iloc[0]) if len(mode_vals) else "Unknown"
                cat = series.astype(str).fillna(fill_value)
                report["imputations"].append(f"{col} (categorical, mode={fill_value})")

            if encoding == "onehot":
                ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                encoded = ohe.fit_transform(cat.to_numpy().reshape(-1, 1))
                cats = [str(c) for c in ohe.categories_[0]]
                for ci, cname in enumerate(cats):
                    clean = f"{_clean_column_name(col)}_{_clean_column_name(cname)}"
                    transformed_cols.append(encoded[:, ci].tolist())
                    column_order.append(clean)
                onehot_meta[col] = {"categories": cats, "columns": [
                    f"{_clean_column_name(col)}_{_clean_column_name(c)}" for c in cats
                ]}
            else:  # label encoding
                le = LabelEncoder()
                encoded = le.fit_transform(cat)
                fitted_encoders[col] = {"classes": [str(c) for c in le.classes_]}
                report["encoded"].append(col)
                transformed_cols.append(encoded.astype(float).tolist())
                column_order.append(col)

    if not transformed_cols:
        raise ValueError("No usable feature columns after preprocessing")

    X = np.column_stack(transformed_cols).astype(float)

    # Feature names --------------------------------------------------------- #
    feature_names: list[str] = []
    for col in feature_cols:
        if col in onehot_meta:
            feature_names.extend(onehot_meta[col]["columns"])
        else:
            feature_names.append(col)
    feature_names = [_clean_column_name(f) for f in feature_names]

    # Scaling ---------------------------------------------------------------- #
    if scaling in ("standard", "minmax") and X.shape[1] > 0:
        if scaling == "standard":
            fitted_scaler = StandardScaler()
        else:
            fitted_scaler = MinMaxScaler()
        X = fitted_scaler.fit_transform(X)

    # SMOTE ----------------------------------------------------------------- #
    smote_applied = False
    before_counts = np.bincount(y).tolist()
    if config.get("smote") and len(class_names) >= 2 and X.shape[0] >= 6:
        min_class = min(np.bincount(y))
        if min_class >= 6:
            smote = SMOTE(random_state=random_state, k_neighbors=min(5, min_class - 1))
            X, y = smote.fit_resample(X, y)
            smote_applied = True
    report["smote_applied"] = smote_applied
    report["smote_before"] = before_counts
    report["smote_after"] = np.bincount(y).tolist()
    report["feature_count"] = X.shape[1]
    report["sample_count"] = X.shape[0]

    pipeline = {
        "target_column": str(target_column),
        "feature_columns": [str(c) for c in feature_cols],
        "feature_names": feature_names,
        "column_order": column_order,
        "class_names": class_names,
        "target_encoder": target_encoder,
        "encoders": fitted_encoders,
        "onehot_meta": onehot_meta,
        "scaler": fitted_scaler,
        "scaling": scaling,
        "encoding": encoding,
        "config": config,
    }
    return X, y, pipeline, report


def apply_preprocessor(df: pd.DataFrame, pipeline: dict) -> np.ndarray:
    """Apply a fitted pipeline to a new dataframe (same schema). Returns X."""
    feature_cols = list(pipeline["feature_columns"])
    if not feature_cols:
        return np.empty((len(df), 0))
    df_work = df.copy()

    cols_out: list[list[float]] = []
    for col in feature_cols:
        if col not in df_work.columns:
            raise ValueError(f"Missing expected feature column: {col}")
        series = df_work[col]
        if col in pipeline["encoders"]:  # label-encoded categorical
            le_info = pipeline["encoders"][col]
            classes = le_info["classes"]
            cat = series.astype(str).fillna("Unknown")
            codes = []
            for v in cat:
                if v in classes:
                    codes.append(float(classes.index(v)))
                else:
                    codes.append(0.0)  # unseen category -> first class bucket
            cols_out.append(codes)
        elif col in pipeline["onehot_meta"]:
            ohe_info = pipeline["onehot_meta"][col]
            categories = ohe_info["categories"]
            cat = series.astype(str)
            for cname in categories:
                cols_out.append((cat == cname).astype(float).tolist())
        else:  # numeric
            numeric = pd.to_numeric(series, errors="coerce")
            numeric = numeric.fillna(0.0).astype(float)
            cols_out.append(numeric.tolist())

    X = np.column_stack(cols_out).astype(float)
    if pipeline.get("scaler") is not None:
        X = pipeline["scaler"].transform(X)
    return X


# --------------------------------------------------------------------------- #
# Public helpers
# --------------------------------------------------------------------------- #
def default_target(df: pd.DataFrame) -> Optional[str]:
    """Pick a sensible default target column (classification-first).

    Prefers genuine categorical columns and refuses to auto-select continuous
    numeric columns, which would otherwise force a meaningless high-cardinality
    classification problem (e.g. a 50-class target that trains to ~2% accuracy).
    """
    # Pass 1: real categorical columns (non-numeric), low cardinality.
    for col in df.columns:
        s = df[col]
        nunique = s.nunique(dropna=True)
        if nunique < 2:
            continue
        if not _is_numeric(s) and nunique <= 50:
            return str(col)
    # Pass 2: numeric columns that behave like categorical labels (0/1 flags,
    # small integer buckets), NOT continuous measurements.
    for col in df.columns:
        s = df[col]
        nunique = s.nunique(dropna=True)
        if nunique < 2:
            continue
        n = max(len(s), 1)
        if nunique <= 20 and nunique / n < 0.1:
            return str(col)
    return None
