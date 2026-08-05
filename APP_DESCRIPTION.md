# ModelMind AI — Complete Application Description

A user-personalized, web-based machine-learning platform that takes raw CSV data through the full ML lifecycle — upload, preprocessing, training, evaluation, comparison, prediction, and now LLM-assisted labeling — without writing a single line of code.

**Stack:** FastAPI (Python) backend + SQLAlchemy/SQLite + React 18 / Tailwind / Vite frontend. Single monorepo (`backend/`, `frontend/`, `sample_data/`, `scripts/`). Runs in dev mode (Vite on :5173, API on :8000) or as a single production server (FastAPI serves the built React app).

---

## 1. Authentication & User Experience

- **Register / Login / Email OTP verification.** New accounts are created unverified; a 6-digit code is emailed (real Gmail SMTP, or returned inline in dev mode) and must be entered within 10 minutes (max 5 attempts). Verification auto-logs the user in.
- **JWT sessions** (7-day expiry) with PBKDF2 password hashing and bearer-token auth on every API call.
- **Private per-user workspaces.** Every dataset, model, and experiment is isolated by the logged-in user — nothing is shared.
- **Dark / light theme** with OS-preference detection and persisted choice.
- **Protected routes** — unauthenticated visitors are redirected to login.

## 2. Dataset Management

- **CSV upload** with type validation and automatic profiling (rows, columns, dtypes, missing %, imbalance, class counts, target warnings).
- **Dataset list / preview / delete** per user.
- **Versioning** — each dataset tracks a history of versions (original upload, re-processing, LLM-generated columns). Selecting an older version is a roadmap item.
- **Live preview** with `head` re-read from disk.

## 3. Automated Preprocessing

- **Dataset profiling engine** — per-column types, missing-value stats, imbalance detection, continuous-variable target warnings.
- **Smart auto-config** — the engine recommends, with human-readable reasons:
  - Missing-value strategy (mean/median/mode/drop for numerics; mode/constant/drop for categoricals)
  - Label vs. one-hot encoding
  - Standard / MinMax scaling
  - SMOTE oversampling when class imbalance is severe
  - Auto-detection of the best target column (classification-first)
- **Manual override mode** — the user can override every decision (imputation, encoding, scaling, SMOTE toggle, column dropping).
- **Transformation report** — shows exactly what happened (samples, features, dropped rows, imputations, SMOTE status).

## 4. Multi-Model Training

Five algorithms, including hybrid ensembles:

| Model | Type | Highlights |
|---|---|---|
| Decision Tree | Base | Full hyperparameter form |
| K-Nearest Neighbors | Base | Full hyperparameter form |
| Random Forest | Base | Full hyperparameter form |
| Voting Ensemble | Hybrid | Soft/hard voting of DT/KNN/RF |
| Stacking Ensemble | Hybrid | Logistic-Regression meta-learner |

- **Hyperparameter controls** rendered dynamically from a model registry (int inputs, dropdowns, nullable "None" fields with tooltips).
- **Auto-tuning** via a lightweight GridSearchCV on a configurable number of folds.
- **Train/test split** slider (10–40%) with stratified split (automatic fallback when stratification is impossible).
- Result shows accuracy + macro/weighted precision/recall/F1 + feature + class names.

## 5. Intelligent Recommendations

A built-in recommendation engine scores models and preprocessing steps against the dataset and explains its reasoning:

- **Model recommendations** — DT/KNN/RF/Voting/Stacking ranked with reasons ("dataset is small and low-dimensional → KNN may overfit").
- **Preprocessing recommendations** — priority-tagged steps.
- **Improvement suggestions** — SMOTE/class weighting, tuning, cross-validation, ensembling, interpretability, deployment readiness.
- **Predicted best model** for the given target.

## 6. Evaluation & Visualization (13 charts)

Every trained model gets a full evaluation regenerated on demand:

- Metric cards (macro + weighted precision/recall/F1), per-class metrics table
- **Confusion matrix**
- **Feature importance**
- **Decision-tree structure plot** (tree models)
- **ROC curve**
- **Precision-recall curve**
- **Learning curve**
- **Class balance**
- **Predicted-vs-actual**
- **Probability (confidence) histogram**
- **Correlation heatmap**
- All charts are headless-matplotlib base64 PNGs, downloadable from the UI.

## 7. Model Comparison Engine

- Select any subset of the 5 algorithms (or enable "Hybrid mode" to auto-include voting/stacking).
- All models train on the **same split** for a fair comparison.
- **Leaderboard** table with accuracy + macro/weighted P/R/F1.
- Winner highlight + **Accuracy Comparison bar** and **Metric Profile radar** charts.
- One-click export of the best model as a `.pkl` file.

## 8. Prediction (Live & Batch)

- **Single-row prediction** — a form is generated per feature (categoricals → dropdowns from data values, numerics → inputs). Result shows the predicted class, per-class probabilities, and top contributing features.
- **Batch prediction** — upload a CSV, the app validates required columns, predicts every row, prepends `prediction` + `confidence` columns, and lets you download the results CSV.

## 9. LLM-Powered Outcome Labeling

- **Provider-agnostic** — works with OpenAI (e.g. gpt-4o-mini) or local Ollama (llama3.2), no extra Python dependencies.
- **Designs categories** — the LLM proposes 2–6 outcome classes with descriptions based on the dataset's schema and sample rows.
- **Labels every row** — batched classification (configurable batch size and max rows).
- **Persists as a new versioned column** — the result becomes a new dataset version usable for training. UI shows per-category counts and preview.
- Configurable via `.env` (`LLM_PROVIDER`, `OPENAI_API_KEY`, `OLLAMA_MODEL`, etc.).

## 10. Personalized Dashboard

- Stat cards (datasets, models, avg. accuracy, experiments)
- **Accuracy trend** line chart (last 8 models) + **models-by-type** pie chart
- **Suggested improvements** (from the recommendation engine)
- **Recent models** with accuracy badges
- **Activity timeline** — every action (upload, preprocess, train, predict, compare) is logged and shown.

## 11. Experiment History & Data Management

- Full append-only activity log powering the dashboard and a dedicated History page.
- Trained-model table (name, type, accuracy, created, delete with confirmation).
- Dataset table (rows/cols, delete) with cascade cleanup of dependent models.

## 12. Exports & Reports

- **Model** export (`.pkl`) — downloadable, reusable artifact.
- **Evaluation report** (`.csv`) — includes per-class precision/recall/F1 rows.
- **Batch prediction results** (`.csv`).

---

## Guided 5-Step Workflow

The core experience is a wizard: **Upload → Preprocess → Train → Evaluate → Predict**, with a step indicator, back-navigation, and "start new experiment" flow. All other pages (Dashboard, Compare, History) orbit this workflow.

---

## What's Already Askable for Feedback / Possible Roadmap Features

**Priority 1 — highest impact (recommended focus):**

1. **AI Assistant** — chat about datasets and models (ask questions, get answers from the data/model in plain language).
2. **SHAP/LIME explainability** — per-prediction and global feature-attribution explanations.
3. **Interactive EDA dashboard** — clickable scatter/box/histogram exploration before modeling.
4. **Dataset Health Score** — a single score + automated insights (missing %, imbalance, drift, cardinality).
5. **Experiment workspace with versioning and notes** — track runs, log notes/params, compare versions side-by-side.
6. **Automatic PDF report generation** — export a polished PDF summary of dataset, preprocessing, model, metrics, charts.
7. **Natural-language ML commands** — type "train a random forest with 80/20 split" and have the platform do it.
8. **Pipeline visualization** — drag-and-drop node graph of data → preprocess → train → evaluate.
9. **Model deployment and monitoring** — one-click inference endpoint, live usage/accuracy monitoring, drift alerts.
10. **Notebook and portfolio export** — export the whole experiment as a clean Jupyter notebook / shareable project portfolio.

**Other candidates (lower priority):**

- **More models:** XGBoost, LightGBM, Logistic Regression, SVC, Gaussian Naive Bayes, MLP neural network, regression support (currently classification-only).
- **Regression support** — continuous targets currently get warnings, not full training.
- **Multi-class & multilabel experiments, class-weight handling.**
- **Feature engineering:** feature selection, PCA, automated feature creation, outlier detection.
- **Data cleaning UX:** column editing, duplicate-row removal, row sampling, custom data upload formats (Excel/JSON), larger-file streaming.
- **Dataset version browsing / rollback** (versioning already exists under the hood).
- **Experiment tracking** — A/B compare runs, log hyperparameters/loss curves, MLflow-style history.
- **Model persistence sharing** — public links, cross-user sharing, import/export of pipelines.
- **Batch prediction on stored datasets** (not just file upload).
- **LLM enhancements** — generate feature descriptions / dataset summaries, natural-language data questions, auto-generated column names, editable categories before labeling, chunked full-dataset labeling progress, cost controls.
- **Teams / roles** — organizations, role-based access, shared datasets.
- **Deployment** — one-click model deployment to an inference endpoint / Docker image export.
- **Retraining & monitoring** — scheduled retraining, drift detection, prediction caching.
- **Explainability** — SHAP/LIME, partial dependence plots, counterfactuals.
- **Scheduling / automation** — CRON-like automated pipelines.
- **Progressive UI** — multi-file training sets, drag-drop reordering, richer EDA tab, dataset statistics gallery.
- **Enterprise polish** — SSO, audit logs, quotas, backup/restore, PostgreSQL deployment guide.
