"""Model registry, builder and trainer for the supported estimators.

Supported model types
---------------------
* ``dt``       Decision Tree
* ``knn``      K-Nearest Neighbors
* ``rf``       Random Forest
* ``voting``   Hard/Soft Voting hybrid ensemble (DT + KNN + RF)
* ``stacking`` Stacking ensemble (DT + KNN + RF -> meta Logistic Regression)
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from ..config import settings

# --------------------------------------------------------------------------- #
# Registry (drives the UI controls + validation)
# --------------------------------------------------------------------------- #
MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "dt": {
        "label": "Decision Tree",
        "category": "tree",
        "supports_importance": True,
        "supports_tree_plot": True,
        "params": [
            {
                "name": "max_depth",
                "type": "int",
                "label": "Max Depth",
                "default": 8,
                "min": 1,
                "max": 30,
                "allow_null": True,
                "help": "None = unlimited depth",
            },
            {
                "name": "criterion",
                "type": "select",
                "label": "Criterion",
                "default": "entropy",
                "options": ["gini", "entropy"],
            },
            {
                "name": "min_samples_split",
                "type": "int",
                "label": "Min Samples Split",
                "default": 2,
                "min": 2,
                "max": 20,
            },
        ],
    },
    "knn": {
        "label": "K-Nearest Neighbors",
        "category": "instance",
        "supports_importance": False,
        "supports_tree_plot": False,
        "params": [
            {
                "name": "n_neighbors",
                "type": "int",
                "label": "K (neighbors)",
                "default": 5,
                "min": 1,
                "max": 50,
            },
            {
                "name": "weights",
                "type": "select",
                "label": "Weights",
                "default": "distance",
                "options": ["uniform", "distance"],
            },
            {
                "name": "metric",
                "type": "select",
                "label": "Distance Metric",
                "default": "minkowski",
                "options": ["minkowski", "euclidean", "manhattan"],
            },
        ],
    },
    "rf": {
        "label": "Random Forest",
        "category": "ensemble",
        "supports_importance": True,
        "supports_tree_plot": False,
        "params": [
            {
                "name": "n_estimators",
                "type": "int",
                "label": "N Estimators",
                "default": 100,
                "min": 10,
                "max": 500,
            },
            {
                "name": "max_depth",
                "type": "int",
                "label": "Max Depth",
                "default": 8,
                "min": 1,
                "max": 30,
                "allow_null": True,
                "help": "None = unlimited depth",
            },
            {
                "name": "criterion",
                "type": "select",
                "label": "Criterion",
                "default": "gini",
                "options": ["gini", "entropy"],
            },
        ],
    },
    "voting": {
        "label": "Hybrid Voting Ensemble",
        "category": "hybrid",
        "supports_importance": False,
        "supports_tree_plot": False,
        "params": [
            {
                "name": "voting",
                "type": "select",
                "label": "Voting Strategy",
                "default": "soft",
                "options": ["soft", "hard"],
                "help": "soft = averaged probabilities, hard = majority class",
            },
            {
                "name": "n_estimators",
                "type": "int",
                "label": "RF Base N Estimators",
                "default": 100,
                "min": 10,
                "max": 500,
            },
            {
                "name": "n_neighbors",
                "type": "int",
                "label": "KNN Base K",
                "default": 5,
                "min": 1,
                "max": 50,
            },
            {
                "name": "max_depth",
                "type": "int",
                "label": "Tree Base Max Depth",
                "default": 8,
                "min": 1,
                "max": 30,
                "allow_null": True,
            },
        ],
    },
    "stacking": {
        "label": "Stacking Ensemble",
        "category": "hybrid",
        "supports_importance": False,
        "supports_tree_plot": False,
        "params": [
            {
                "name": "n_estimators",
                "type": "int",
                "label": "RF Base N Estimators",
                "default": 100,
                "min": 10,
                "max": 500,
            },
            {
                "name": "n_neighbors",
                "type": "int",
                "label": "KNN Base K",
                "default": 5,
                "min": 1,
                "max": 50,
            },
            {
                "name": "max_depth",
                "type": "int",
                "label": "Tree Base Max Depth",
                "default": 8,
                "min": 1,
                "max": 30,
                "allow_null": True,
            },
            {
                "name": "cv",
                "type": "int",
                "label": "Meta-Learner CV Folds",
                "default": 5,
                "min": 2,
                "max": 10,
            },
        ],
    },
}

MODEL_TYPES = list(MODEL_REGISTRY.keys())


def get_model_registry() -> dict:
    return MODEL_REGISTRY


def _clean_params(model_type: str, params: Optional[dict]) -> dict:
    """Validate/coerce hyperparameters against the registry defaults."""
    params = dict(params or {})
    for spec in MODEL_REGISTRY[model_type]["params"]:
        name, default = spec["name"], spec["default"]
        if name not in params or params[name] in (None, "", "null"):
            params[name] = default
        elif spec["type"] == "int":
            try:
                params[name] = int(params[name])
            except (ValueError, TypeError):
                params[name] = default
    return params


def build_model(model_type: str, params: Optional[dict] = None, random_state: int = 42):
    """Instantiate a fresh, unfitted estimator."""
    params = _clean_params(model_type, params)
    rs = random_state

    if model_type == "dt":
        return DecisionTreeClassifier(
            criterion=params["criterion"],
            max_depth=params.get("max_depth") or None,
            min_samples_split=params["min_samples_split"],
            random_state=rs,
        )
    if model_type == "knn":
        return KNeighborsClassifier(
            n_neighbors=params["n_neighbors"],
            weights=params["weights"],
            metric=params["metric"],
        )
    if model_type == "rf":
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params.get("max_depth") or None,
            criterion=params["criterion"],
            random_state=rs,
            n_jobs=-1,
        )
    if model_type == "voting":
        return VotingClassifier(
            estimators=[
                ("dt", DecisionTreeClassifier(max_depth=params.get("max_depth") or None, random_state=rs)),
                ("knn", KNeighborsClassifier(n_neighbors=params["n_neighbors"])),
                ("rf", RandomForestClassifier(n_estimators=params["n_estimators"], random_state=rs, n_jobs=-1)),
            ],
            voting=params["voting"],
        )
    if model_type == "stacking":
        return StackingClassifier(
            estimators=[
                ("dt", DecisionTreeClassifier(max_depth=params.get("max_depth") or None, random_state=rs)),
                ("knn", KNeighborsClassifier(n_neighbors=params["n_neighbors"])),
                ("rf", RandomForestClassifier(n_estimators=params["n_estimators"], random_state=rs, n_jobs=-1)),
            ],
            final_estimator=LogisticRegression(max_iter=1000),
            cv=params["cv"],
            stack_method="auto",
            n_jobs=-1,
        )
    raise ValueError(f"Unknown model type: {model_type}")


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str,
    params: Optional[dict] = None,
    random_state: int = 42,
    tune: bool = False,
    cv_folds: int = 5,
):
    """Train the model; optionally runs GridSearchCV-lite for the 3 base models."""
    model = build_model(model_type, params, random_state)
    if tune and model_type in ("dt", "knn", "rf"):
        grid = _auto_grid(model_type)
        if grid:
            gs = GridSearchCV(model, grid, cv=min(cv_folds, 3), scoring="accuracy", n_jobs=-1)
            gs.fit(X, y)
            return gs.best_estimator_, gs.best_params_, gs.cv_results_["mean_test_score"].mean()
    model.fit(X, y)
    return model, _clean_params(model_type, params), None


def _auto_grid(model_type: str) -> Optional[dict]:
    if model_type == "dt":
        return {"max_depth": [3, 5, 8, None], "criterion": ["gini", "entropy"]}
    if model_type == "knn":
        return {"n_neighbors": [3, 5, 7, 11], "weights": ["uniform", "distance"]}
    if model_type == "rf":
        return {"n_estimators": [50, 100], "max_depth": [5, 8, None]}
    return None


def describe_model(model_type: str) -> dict:
    return MODEL_REGISTRY[model_type]
