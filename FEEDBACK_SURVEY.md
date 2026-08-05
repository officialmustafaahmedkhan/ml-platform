# ML Platform — Feature Feedback Survey

**Audience:** existing users of the platform (someone who has used the app at least once).
**Goal:** decide which roadmap features to build next, and how to prioritize them by perceived value and ease.

---

## Section 1 — About you

1. **How often do you use the platform?**
   - Weekly or more
   - A few times a month
   - Rarely
   - Tried once

2. **What is your primary role?**
   - Data scientist / ML engineer
   - Analyst / business user
   - Student / researcher
   - Developer (non-ML)
   - Other

3. **What is your main use case?** (open text)
   - Classification experiments, teaching, quick prototyping, end-to-end ML learning, etc.

---

## Section 2 — How valuable is each feature to you?

For each, rate **Value** (1 = not valuable, 5 = essential) and **Would I use it in the next month?** (Yes / Maybe / No).

### Priority features (highest potential impact)

| # | Feature | Value (1–5) | Would you use it? | Notes |
|---|---------|-------------|-------------------|-------|
| 1 | **AI Assistant** — chat about your datasets and models in plain language ("why is my model confused about class X?") | | | |
| 2 | **SHAP/LIME explainability** — see which features drive each prediction, and overall model behavior | | | |
| 3 | **Interactive EDA dashboard** — clickable scatter plots, histograms, box plots to explore data before training | | | |
| 4 | **Dataset Health Score** — one score + automated alerts (missing data, imbalance, high cardinality, drift) | | | |
| 5 | **Experiment workspace** — save runs, add notes, compare versions side-by-side | | | |
| 6 | **Automatic PDF report** — one-click polished PDF of dataset, preprocessing, model metrics and charts | | | |
| 7 | **Natural-language ML commands** — type "train a random forest, 80/20 split" and the app does it | | | |
| 8 | **Pipeline visualization** — visual drag-and-drop graph: data → preprocess → train → evaluate | | | |
| 9 | **Model deployment & monitoring** — deploy to a live API endpoint, track accuracy, drift alerts | | | |
| 10 | **Notebook / portfolio export** — export the whole experiment as a clean Jupyter notebook | | | |

### Secondary features

| Feature | Value (1–5) | Would you use it? | Notes |
|---------|-------------|-------------------|-------|
| More algorithms (XGBoost, LightGBM, Logistic Regression, SVC, MLP, Naive Bayes) | | | |
| Regression support (continuous targets) | | | |
| Feature engineering (feature selection, PCA, outlier removal) | | | |
| Data cleaning UX (edit columns, remove duplicates, sample rows, Excel/JSON import) | | | |
| Dataset version rollback | | | |
| Batch prediction on stored datasets (not only file upload) | | | |
| Cross-user model/dataset sharing | | | |
| Teams / role-based access | | | |
| Scheduled retraining & drift detection | | | |
| SSO / enterprise login | | | |

---

## Section 3 — Open questions

11. **What is the #1 thing that would make you use this platform more often?** (open text)

12. **What frustrates you most about the current workflow?** (open text)

13. **If you could only pick ONE feature from the priority list, which would it be and why?** (open text)

14. **What feature would you pay for (subscription/one-time)?** (open text)

---

## Section 4 — Prioritization

15. **Rank your top 3 priorities from the priority list above** (just the numbers, e.g. "1, 3, 7").

16. **Would you prefer** (pick one):
    - Depth: fewer features, done really well
    - Breadth: many features, more surface area

17. **How important is each to you?** (1–5)
    - Speed of the workflow (faster from upload → model)
    - Explainability / trust in the results
    - Automation ("set and forget")
    - Visual polish / presentation (reports, charts)
    - Team collaboration
