# User-Personalized Intelligent Machine Learning Platform with Hybrid Ensemble Learning

An upgraded, production-ready evolution of the original
[Decision Tree Classifier Web App](https://github.com/hassanshoaib29-lang/Decision-Tree-Classifier-Web-App).
A full-stack ML platform where users upload datasets, get smart preprocessing
recommendations, train **multiple models** (Decision Tree, KNN, Random Forest)
plus **hybrid voting/stacking ensembles**, evaluate, compare, export and run
live predictions — all behind a per-user dashboard.

---

## ✨ Feature Highlights

| Area | What it does |
| --- | --- |
| **Multi-model training** | Decision Tree, K-Nearest Neighbors, Random Forest with hyperparameter controls (K, n_estimators, max_depth, …) |
| **Hybrid ensemble learning** | Hard/Soft **Voting** classifier + **Stacking** classifier combining DT + KNN + RF |
| **Automated preprocessing** | Auto mode (recommendation-driven) + Manual mode: missing-value handling (mean/median/mode/drop), Label & One-Hot encoding, Standard/MinMax scaling, **SMOTE** for imbalance |
| **Intelligent recommendations** | Suggests the best model, preprocessing steps, and improvements with human-readable reasons ("Random Forest recommended due to high dimensionality", "Apply scaling before using KNN", "Use SMOTE to handle imbalance") |
| **Evaluation & visualization** | Accuracy, Precision, Recall, F1 (macro + weighted + per-class), Confusion Matrix, Feature Importance, Tree plot, **ROC/PR curves, learning curve, class balance, predicted-vs-actual, probability histogram, correlation heatmap**, comparison bar & radar charts |
| **Model comparison engine** | Trains all models on the same split → leaderboard table + charts + one-click export of the winner; pick specific algorithms (DT/KNN/RF) or enable hybrid voting/stacking |
| **Email OTP verification** | Register creates an unverified account and emails a 6-digit code (SMTP); verify once to activate + auto-login. Empty `SMTP_HOST` = dev mode that returns the code in the API response |
| **Live prediction** | Manual input with class probabilities + batch prediction from CSV with downloadable results |
| **Personalized dashboards** | Per-user stats, accuracy trend, model-type distribution, activity timeline, suggested improvements |
| **Auth & persistence** | JWT authentication; users get private datasets, trained models and experiment history (SQLAlchemy + SQLite) |
| **Export** | Download trained models as `.pkl`, evaluation reports as `.csv`, batch results as `.csv` |
| **AutoML-lite** | GridSearchCV hyperparameter tuning toggle on train |

---

## 🧱 Architecture

Clean separation of concerns with a modular monorepo:

```
ml-platform/
├── backend/                      # FastAPI application
│   ├── app/
│   │   ├── main.py               # app factory, CORS, router registration
│   │   ├── config.py             # env-driven settings
│   │   ├── database.py           # SQLAlchemy engine / session
│   │   ├── models/               # ORM: User, Dataset, ModelArtifact, Experiment
│   │   ├── schemas/              # Pydantic request/response models
│   │   ├── routes/               # auth, datasets, preprocess, train, evaluate,
│   │   │                         #   predict, compare, recommendation, dashboard, export
│   │   ├── services/             # preprocessing, models, hybrid, evaluation,
│   │   │                         #   recommendation, storage, pipeline
│   │   └── utils/                # security (JWT/hash), serialization
│   ├── data/                     # uploads/, models/, reports/ (gitignored)
│   └── tests/                    # pytest end-to-end API suite
├── frontend/                     # React + Tailwind (Vite)
│   └── src/
│       ├── pages/                # Login, Dashboard, Workflow, Compare, History
│       ├── components/           # Layout, StepIndicator, StatCard, charts, …
│       │   └── workflow/         # Upload, Preprocess, Train, Evaluate, Predict
│       ├── context/              # AuthContext, ThemeContext (dark/light)
│       ├── services/             # typed API client
│       └── api/                  # axios instance + interceptors
├── sample_data/                  # ready-to-test datasets
│   ├── play_tennis.csv           # classic small categorical dataset
│   └── loan_approval.csv         # mixed-type, ~12% missing, imbalanced (SMOTE demo)
└── scripts/                      # dataset generator + no-pandas smoke tests
```

**Technology stack**

- **Backend:** FastAPI, SQLAlchemy (SQLite out-of-the-box, Postgres-ready), PyJWT, Scikit-learn, imbalanced-learn, Matplotlib
- **Frontend:** React 18, Tailwind CSS 3, Vite, React Router, Recharts, Axios
- **Database:** SQLite via SQLAlchemy (`DATABASE_URL` swaps to Postgres/Mongo-compatible SQL easily)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+ (tested on 3.14)
- Node.js 18+

### 1. Backend

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

- Interactive API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/api/health

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 — the dev server proxies `/api` to the backend.

### 3. Try it end-to-end

1. **Register** a new account. Without SMTP configured the 6-digit OTP is shown
   on the login screen (dev mode); otherwise check the email we send.
2. **Verify** the OTP — the account is activated and you are logged in.
3. Open **ML Workflow**, upload `sample_data/loan_approval.csv` (or `play_tennis.csv`), pick the target (`loan_approved` / `Play`).
4. **Preprocess** — watch the engine recommend scaling/SMOTE; run auto or manual.
5. **Train** — pick a model or the hybrid voting/stacking ensemble.
6. **Evaluate** — confusion matrix, ROC/PR curves, learning curve, class balance,
   predicted-vs-actual, probability histogram, correlation heatmap, export.
7. **Predict** — manual input or batch CSV. In **Compare**, untick *Hybrid mode*
   to pick exactly which algorithms to compare.

---

## 🔌 API Overview

| Method | Endpoint | Purpose |
| --- | --- | --- |
| POST | `/api/auth/register` `/api/auth/login` `/api/auth/me` | JWT auth + email OTP |
| POST | `/api/auth/send-otp` `/api/auth/verify-otp` | Resend / verify the 6-digit OTP (verify activates + logs in) |
| POST | `/api/datasets/upload` | Upload CSV (per user) |
| GET | `/api/datasets` `/api/datasets/{id}` `/api/datasets/{id}/head` | List/inspect |
| POST | `/api/preprocess/auto-config/{id}` | Preview recommended config |
| POST | `/api/preprocess` | Run preprocessing (auto/manual) |
| GET | `/api/preprocess/profile/{id}` | Dataset profile |
| POST | `/api/train` | Train a model (`dt`, `knn`, `rf`, `voting`, `stacking`) |
| GET | `/api/train/models` `/api/train/registry` | List models / registry |
| GET | `/api/evaluate/{model_id}` | Metrics + charts |
| POST | `/api/predict` `/api/predict/batch` | Single + batch prediction |
| POST | `/api/compare` | Train-all + leaderboard |
| GET | `/api/recommend/{dataset_id}` | Recommendation engine |
| GET | `/api/dashboard` | Personalized dashboard payload |
| GET | `/api/export/model/{id}` `/api/export/report/{id}` | Downloads |

All endpoints (except auth) require `Authorization: Bearer <token>`.

---

## ✅ Running the Tests

The end-to-end suite (`backend/tests/test_api.py`) covers auth **with the OTP
verification flow**, upload, preprocessing, training of **all five** model types,
evaluation (**including the new prediction charts**), single/batch prediction,
comparison (**explicit algorithm selection**), recommendations, dashboard and
exports. **Status: verified — 16/16 tests pass** and the full journey was
exercised over live HTTP (`scripts/live_smoke.py`, 26 checks):

```bash
cd backend
pytest -v
```

> **Note:** Some Windows machines with **Smart App Control** enabled can
> transiently reputation-block pandas' compiled DLLs. It clears once the files
> gain reputation (or on Linux/macOS/CI). A pandas-free smoke test of the core
> logic is included as a fallback:
>
> ```bash
> python scripts/smoke_no_pandas.py
> ```

---

## 🧪 Sample Datasets

| File | Rows | Type | Why it's useful |
| --- | --- | --- | --- |
| `sample_data/play_tennis.csv` | 14 | All-categorical | Classic small dataset; one-hot encoding + Decision Tree |
| `sample_data/loan_approval.csv` | 240 | Mixed, ~12% missing, imbalanced | Exercises scaling, SMOTE and imbalance handling |

Regenerate the loan dataset deterministically with `python scripts/generate_loan_data.py`.

---

## ☁️ Deployment

- **Frontend:** build with `npm run build` then deploy `frontend/dist` to Vercel/Netlify.
  Set `VITE_API_URL` to the deployed backend URL.
- **Backend:** `uvicorn app.main:app` on Render/Railway/Docker. Set `DATABASE_URL`,
  `JWT_SECRET_KEY`, `CORS_ORIGINS`.
- **Single-server mode:** if `frontend/dist` exists, FastAPI automatically serves
  the built React app, so you can deploy one service only.

---

## 🗺️ Advanced Enhancements (roadmap)

- **AutoML-lite auto-selection** — train top-k recommended models and pick the best automatically (already partially available via the Compare engine).
- **Hyperparameter tuning** — GridSearchCV toggle already shipped on train.
- **Explainable AI** — SHAP/LIME integration for instance explanations.
- **Experiment tracking** — MLflow-style run comparison (history table is the first step).
- **Dataset versioning** — the schema already stores a `versions` array.
- **API key access** for third-party prediction endpoints.

---

## 📄 License

Demo/project scaffold — original upstream project by hassanshoaib29-lang.
