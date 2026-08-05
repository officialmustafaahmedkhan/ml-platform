"""Intelligent recommendation engine.

Given a dataset profile (and optionally evaluation results) it produces:

* model recommendations with human-readable reasons
* preprocessing recommendations
* improvement suggestions

Heuristics are deliberately transparent and explainable so every suggestion
surfaces as ``{model/action, score, reasons: [...]}``.
"""
from __future__ import annotations

from typing import Any, Optional

from .preprocessing import profile_dataset


# --------------------------------------------------------------------------- #
# Model recommendation
# --------------------------------------------------------------------------- #
def recommend_model(profile: dict) -> list[dict]:
    rows = profile["rows"]
    n_features = profile["columns"]
    n_categorical = len(profile["categorical_columns"])
    n_numeric = len(profile["numeric_columns"])
    imbalance = profile.get("imbalance_ratio")
    num_classes = profile.get("num_classes", 0)

    scores: list[dict] = []

    def add(model_type, base, reasons):
        scores.append({"model_type": model_type, "score": base, "reasons": reasons})

    # --- Decision Tree ---------------------------------------------------- #
    dt_reasons = ["Interpretable, works well on small datasets"]
    dt_score = 60
    if rows <= 300:
        dt_score += 25
        dt_reasons.append("Small dataset (<300 rows) favors low-variance models")
    if n_categorical > n_numeric:
        dt_score += 10
        dt_reasons.append("Handles categorical features without explicit scaling")
    add("dt", dt_score, dt_reasons)

    # --- KNN -------------------------------------------------------------- #
    knn_reasons = []
    knn_score = 40
    if rows >= 100 and n_features <= 8:
        knn_score += 25
        knn_reasons.append("Compact numeric feature space (<=8 features)")
    if n_numeric == n_features:
        knn_score += 15
        knn_reasons.append("All-numeric dataset is naturally distance-friendly")
    if rows < 30:
        knn_score -= 20
        knn_reasons.append("Too few rows for reliable neighbor search")
    if n_features > 15:
        knn_score -= 15
        knn_reasons.append("High dimensionality degrades distance metrics (curse of dimensionality)")
    add("knn", max(0, knn_score), knn_reasons or ["Simple and competitive baseline"])

    # --- Random Forest ---------------------------------------------------- #
    rf_reasons = []
    rf_score = 55
    if n_features >= 12:
        rf_score += 25
        rf_reasons.append("High dimensionality recommended -> Random Forest")
    if rows >= 500:
        rf_score += 20
        rf_reasons.append("Larger dataset lets Random Forest shine")
    if imbalance is not None and imbalance < 0.5:
        rf_score += 10
        rf_reasons.append("Robust to class imbalance via bootstrapping")
    if num_classes >= 3:
        rf_score += 8
        rf_reasons.append("Multiclass problems handled natively by tree ensembles")
    add("rf", rf_score, rf_reasons or ["Robust, variance-reducing ensemble"])

    # --- Hybrid Voting ---------------------------------------------------- #
    hybrid_reasons = [
        "Combines DT + KNN + RF; usually the most robust single choice",
        "Soft voting averages calibrated probabilities across three families",
    ]
    add("voting", 75, hybrid_reasons)

    # Stacking gets a small premium on larger datasets
    stacking_score = 70 + (10 if rows >= 300 else 0) + (5 if n_features >= 8 else 0)
    add(
        "stacking",
        stacking_score,
        ["Meta-learner (Logistic Regression) learns how to combine base models best"],
    )

    scores.sort(key=lambda s: s["score"], reverse=True)
    for s in scores:
        s["score"] = round(s["score"], 1)
    return scores


# --------------------------------------------------------------------------- #
# Preprocessing recommendation
# --------------------------------------------------------------------------- #
def recommend_preprocessing(profile: dict, model_hint: Optional[str] = None) -> list[dict]:
    recs: list[dict] = []
    missing_pct = profile["missing_pct"]
    imbalance = profile.get("imbalance_ratio")
    n_features = profile["columns"]
    high_card = profile.get("high_cardinality_categoricals", [])

    if missing_pct > 5:
        recs.append(
            {
                "category": "missing_values",
                "action": "Impute missing values",
                "detail": "Median impute numeric columns, mode impute categorical columns",
                "priority": "high",
            }
        )
    else:
        recs.append(
            {
                "category": "missing_values",
                "action": "Missing values OK",
                "detail": "Missingness is below 5% — simple imputation is sufficient",
                "priority": "low",
            }
        )

    if model_hint == "knn" or (model_hint is None and n_features <= 8):
        recs.append(
            {
                "category": "scaling",
                "action": "Apply scaling before using KNN",
                "detail": "KNN relies on distances; StandardScaler equalizes feature ranges",
                "priority": "high",
            }
        )

    if imbalance is not None and imbalance < 0.35:
        recs.append(
            {
                "category": "imbalance",
                "action": "Use SMOTE to handle imbalance",
                "detail": f"Minority class is only {imbalance * 100:.0f}% of the majority — resample before training",
                "priority": "high",
            }
        )
    elif imbalance is not None and imbalance < 0.6:
        recs.append(
            {
                "category": "imbalance",
                "action": "Monitor class balance",
                "detail": "Mild imbalance detected; prefer macro-F1 over accuracy",
                "priority": "medium",
            }
        )

    if high_card:
        recs.append(
            {
                "category": "encoding",
                "action": "Label-encode high-cardinality categoricals",
                "detail": f"{', '.join(high_card)} have >10 unique values; One-Hot would explode dimensionality",
                "priority": "medium",
            }
        )

    if n_features >= 15:
        recs.append(
            {
                "category": "dimensionality",
                "action": "Consider feature selection",
                "detail": "High-dimensional dataset — inspect feature importance and drop noisy columns",
                "priority": "medium",
            }
        )

    return recs


# --------------------------------------------------------------------------- #
# Improvement suggestions (optionally informed by evaluation results)
# --------------------------------------------------------------------------- #
def suggest_improvements(
    profile: dict, metrics: Optional[dict] = None, model_type: Optional[str] = None
) -> list[dict]:
    suggestions: list[dict] = []
    f1 = metrics.get("f1_macro") if metrics else None
    accuracy = metrics.get("accuracy") if metrics else None

    if f1 is not None and f1 < 0.75:
        suggestions.append(
            {
                "category": "balance",
                "suggestion": "Add SMOTE or class_weight='balanced'",
                "impact": "Lifts recall on minority classes",
                "effort": "Low",
            }
        )
    if accuracy is not None and accuracy >= 0.95:
        suggestions.append(
            {
                "category": "deploy",
                "suggestion": "Model ready for deployment",
                "impact": "Export the model and expose the prediction API",
                "effort": "Low",
            }
        )
    if metrics is not None:
        suggestions.append(
            {
                "category": "tuning",
                "suggestion": "Run hyperparameter tuning (GridSearchCV)",
                "impact": "Usually +2-8% accuracy",
                "effort": "Medium",
            }
        )
        suggestions.append(
            {
                "category": "validation",
                "suggestion": "Use k-fold cross-validation",
                "impact": "More reliable performance estimates",
                "effort": "Low",
            }
        )
        if model_type == "knn":
            suggestions.append(
                {
                    "category": "model",
                    "suggestion": "Combine KNN into the hybrid voting ensemble",
                    "impact": "Smoothes KNN variance with tree-based models",
                    "effort": "Low",
                }
            )
        else:
            suggestions.append(
                {
                    "category": "model",
                    "suggestion": "Try the hybrid voting ensemble",
                    "impact": "Often beats any single model",
                    "effort": "Low",
                }
            )

    suggestions.append(
        {
            "category": "interpretability",
            "suggestion": "Review feature importance and prune noisy features",
            "impact": "Simpler, faster and often more generalizable",
            "effort": "Medium",
        }
    )
    return suggestions


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def recommend_all(df, target_column: Optional[str], model_hint: Optional[str] = None,
                  metrics: Optional[dict] = None, model_type: Optional[str] = None) -> dict:
    profile = profile_dataset(df, target_column)
    return {
        "dataset_profile": profile,
        "model_recommendations": recommend_model(profile),
        "preprocessing_recommendations": recommend_preprocessing(profile, model_hint),
        "improvement_suggestions": suggest_improvements(profile, metrics, model_type),
        "predicted_best_model": recommend_model(profile)[0],
    }
