"""Model evaluation metrics + matplotlib chart generation (base64 PNGs).

Uses the non-interactive ``Agg`` backend so it works headless (servers, CI).
"""
from __future__ import annotations

import base64
import io
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Consistent style ---------------------------------------------------------- #
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _to_data_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")


def evaluate_model(model, X_test, y_test, class_names: list[str]) -> dict:
    """Compute the full evaluation metric set + confusion matrix."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(class_names))))

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "samples": int(len(y_test)),
    }

    # Per-class breakdown
    report_dict = classification_report(
        y_test, y_pred, labels=list(range(len(class_names))), target_names=class_names,
        output_dict=True, zero_division=0,
    )
    per_class = []
    for i, name in enumerate(class_names):
        r = report_dict.get(name, {})
        per_class.append(
            {
                "class": name,
                "precision": round(float(r.get("precision", 0)), 4),
                "recall": round(float(r.get("recall", 0)), 4),
                "f1": round(float(r.get("f1-score", 0)), 4),
                "support": int(r.get("support", 0)),
            }
        )
    metrics["per_class"] = per_class
    metrics["classification_report_text"] = classification_report(
        y_test, y_pred, labels=list(range(len(class_names))), target_names=class_names,
        zero_division=0,
    )
    return {"metrics": metrics, "confusion_matrix": cm.tolist()}


def feature_importance(model, feature_names: list[str]) -> Optional[dict]:
    """Extract feature importances for tree-based models."""
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
        names = feature_names[: len(importances)]
        idx = np.argsort(importances)[::-1]
        return {
            "features": [str(names[i]) for i in idx],
            "importance": [float(importances[i]) for i in idx],
        }
    return None


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def chart_confusion_matrix(cm: list[list[int]], class_names: list[str]) -> str:
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = np.asarray(cm)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted Label",
        ylabel="True Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    return _to_data_uri(fig)


def chart_feature_importance(importance: dict, top: int = 15) -> str:
    fig, ax = plt.subplots(figsize=(7, max(4, 0.4 * min(len(importance["features"]), top))))
    feats = importance["features"][:top][::-1]
    vals = importance["importance"][:top][::-1]
    ax.barh(feats, vals, color="#6366f1")
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance")
    fig.tight_layout()
    return _to_data_uri(fig)


def chart_tree(model, feature_names: list[str], class_names: list[str]) -> str:
    if not isinstance(model, DecisionTreeClassifier):
        raise TypeError("Tree visualization requires a DecisionTreeClassifier")
    fig, ax = plt.subplots(figsize=(14, 8))
    plot_tree(
        model, filled=True, rounded=True, ax=ax,
        feature_names=feature_names, class_names=class_names, fontsize=8,
    )
    fig.tight_layout()
    return _to_data_uri(fig)


def chart_accuracy_comparison(rows: list[dict], title: str = "Model Comparison") -> str:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [r["model"] for r in rows]
    values = [r.get("accuracy", 0) * 100 for r in rows]
    bars = ax.bar(labels, values, color="#10b981", alpha=0.9)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.set_ylim(0, 105)
    fig.tight_layout()
    return _to_data_uri(fig)


def chart_metric_radar(rows: list[dict]) -> str:
    """Radar chart comparing accuracy/precision/recall/f1 across models."""
    metric_keys = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]
    labels = ["Accuracy", "Precision", "Recall", "F1"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
    for row in rows:
        vals = [row.get(k, 0) for k in metric_keys] + [row.get(metric_keys[0], 0)]
        ax.plot(angles, vals, linewidth=2, label=row["model"])
        ax.fill(angles, vals, alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title("Metric Profile", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    fig.tight_layout()
    return _to_data_uri(fig)


def chart_accuracy_trend(points: list[dict]) -> str:
    """Line chart of accuracy over recent experiments."""
    fig, ax = plt.subplots(figsize=(8, 4))
    x = [p["index"] for p in points]
    y = [p["accuracy"] * 100 for p in points]
    ax.plot(x, y, marker="o", color="#6366f1", linewidth=2)
    for xi, yi in zip(x, y):
        ax.text(xi, yi + 1, f"{yi:.1f}%", ha="center", fontsize=8)
    ax.set_xlabel("Experiment #")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy Trend")
    ax.set_ylim(0, 105)
    fig.tight_layout()
    return _to_data_uri(fig)


# --------------------------------------------------------------------------- #
# Prediction-oriented charts
# --------------------------------------------------------------------------- #
def _ovr_binary_targets(y_true: np.ndarray, n_classes: int):
    """Yield (binary_y, class_index) pairs, one per class (one-vs-rest)."""
    for i in range(n_classes):
        yield (y_true == i).astype(int), i


def chart_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, class_names: list[str]) -> str:
    """ROC curve — binary uses the positive class; multiclass uses one-vs-rest."""
    from sklearn.metrics import auc, roc_curve

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    if len(class_names) == 2:
        fpr, tpr, _ = roc_curve((y_true == 1).astype(int), y_proba[:, 1])
        ax.plot(fpr, tpr, lw=2, color="#6366f1", label=f"{class_names[1]} (AUC={auc(fpr, tpr):.3f})")
    else:
        for i, cls in enumerate(class_names):
            fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_proba[:, i])
            ax.plot(fpr, tpr, lw=2, label=f"{cls} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return _to_data_uri(fig)


def chart_precision_recall(y_true: np.ndarray, y_proba: np.ndarray, class_names: list[str]) -> str:
    """Precision-Recall curve — binary uses the positive class; multiclass OVR."""
    from sklearn.metrics import average_precision_score, precision_recall_curve

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    if len(class_names) == 2:
        prec, rec, _ = precision_recall_curve((y_true == 1).astype(int), y_proba[:, 1])
        ap = average_precision_score((y_true == 1).astype(int), y_proba[:, 1])
        ax.plot(rec, prec, lw=2, color="#10b981", label=f"{class_names[1]} (AP={ap:.3f})")
    else:
        for i, cls in enumerate(class_names):
            prec, rec, _ = precision_recall_curve((y_true == i).astype(int), y_proba[:, i])
            ap = average_precision_score((y_true == i).astype(int), y_proba[:, i])
            ax.plot(rec, prec, lw=2, label=f"{cls} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    return _to_data_uri(fig)


def chart_learning_curve(model, X: np.ndarray, y: np.ndarray, cv: int = 5) -> str:
    """Learning curve: training vs cross-validation score vs training size."""
    from sklearn.model_selection import learning_curve

    cv = max(2, min(int(cv), 5))
    if X.shape[0] < cv + 1:
        raise ValueError("Too few samples for a learning curve")
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, train_sizes=np.linspace(0.2, 1.0, 5),
        scoring="accuracy", n_jobs=-1,
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    tr_mean, tr_std = train_scores.mean(axis=1), train_scores.std(axis=1)
    te_mean, te_std = test_scores.mean(axis=1), test_scores.std(axis=1)
    ax.fill_between(train_sizes, tr_mean - tr_std, tr_mean + tr_std, alpha=0.15, color="#6366f1")
    ax.fill_between(train_sizes, te_mean - te_std, te_mean + te_std, alpha=0.15, color="#10b981")
    ax.plot(train_sizes, tr_mean, "o-", color="#6366f1", label="Training score")
    ax.plot(train_sizes, te_mean, "o-", color="#10b981", label="Cross-validation score")
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Accuracy")
    ax.set_title("Learning Curve")
    ax.legend(loc="best", fontsize=8)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    return _to_data_uri(fig)


def chart_class_balance(y: np.ndarray, class_names: list[str]) -> str:
    """Bar chart of the class distribution of the training target."""
    counts = np.bincount(np.asarray(y).astype(int), minlength=len(class_names))
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bars = ax.bar(class_names, counts, color="#f59e0b", alpha=0.9)
    for bar, v in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.2, int(v), ha="center", fontsize=9)
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution (Train Set)")
    fig.tight_layout()
    return _to_data_uri(fig)


def chart_predicted_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> str:
    """Grouped bars comparing actual vs predicted counts on the test set."""
    actual = np.bincount(np.asarray(y_true).astype(int), minlength=len(class_names))
    predicted = np.bincount(np.asarray(y_pred).astype(int), minlength=len(class_names))
    x = np.arange(len(class_names))
    width = 0.4
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - width / 2, actual, width, label="Actual", color="#6366f1", alpha=0.9)
    ax.bar(x + width / 2, predicted, width, label="Predicted", color="#f43f5e", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Predicted vs Actual (Test Set)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _to_data_uri(fig)


def chart_probability_histogram(y_true: np.ndarray, y_proba: np.ndarray, class_names: list[str]) -> str:
    """Histogram of model confidence (max predicted probability) on the test set."""
    conf = np.asarray(y_proba).max(axis=1)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(conf, bins=10, color="#8b5cf6", alpha=0.85, edgecolor="white")
    ax.set_xlabel("Confidence (max predicted probability)")
    ax.set_ylabel("Frequency")
    ax.set_title("Prediction Confidence Distribution")
    fig.tight_layout()
    return _to_data_uri(fig)


def chart_correlation_heatmap(df_numeric) -> str:
    """Heatmap of Pearson correlations across raw numeric columns."""
    import pandas as pd

    if not isinstance(df_numeric, pd.DataFrame):
        df_numeric = pd.DataFrame(df_numeric)
    if df_numeric.shape[1] < 2:
        raise ValueError("Need at least two numeric columns for a correlation heatmap")
    corr = df_numeric.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(max(5, 0.85 * corr.shape[1]), max(4, 0.75 * corr.shape[0])))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.figure.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns, fontsize=8)
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    return _to_data_uri(fig)
