"""Generates sample_data/loan_approval.csv — a synthetic, deterministic dataset.

Includes numeric + categorical features, ~12% missing values and a class
imbalance (~35:65) so the platform's scaling / SMOTE / imbalance features can
be exercised out of the box. Deterministic (seeded). Uses only numpy + csv so
it runs even where pandas is unavailable.
"""
from __future__ import annotations

import csv

import numpy as np

rng = np.random.default_rng(2026)
n = 240

age = rng.integers(21, 68, n)
income = rng.normal(58_000, 22_000, n).clip(12_000, 180_000).round(0)
credit_score = rng.integers(350, 850, n)
loan_amount = rng.normal(25_000, 14_000, n).clip(2_000, 90_000).round(0)
employment_years = rng.integers(0, 30, n)

genders = rng.choice(["Male", "Female"], n, p=[0.52, 0.48])
education = rng.choice(["High School", "Bachelor", "Master", "PhD"], n, p=[0.3, 0.4, 0.22, 0.08])

# Loan approved if good credit + income relative to loan + employment
score = (
    0.5 * (credit_score / 850)
    + 0.25 * np.clip(income / loan_amount, 0, 6) / 2
    + 0.15 * np.clip(employment_years / 25, 0, 1)
    + rng.normal(0, 0.12, n)
)
approved = (score > 0.55).astype(int)

# Pick a flip probability that yields a clearly imbalanced target
# (minority/majority ratio < 0.35 so SMOTE gets recommended).
def final_balance(flip_prob: float) -> tuple[np.ndarray, dict]:
    flips = rng.random(n) < flip_prob
    app = approved.copy()
    app[flips & (app == 1)] = 0
    return app, {"No": int((app == 0).sum()), "Yes": int((app == 1).sum())}

chosen = None
for fp in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
    app, bal = final_balance(fp)
    ratio = min(bal.values()) / max(bal.values())
    if ratio < 0.35:
        chosen = (fp, app, bal)
        break
assert chosen is not None, "could not reach imbalance target"
_, approved, balance = chosen
print(f"flip prob = {chosen[0]}, final balance = {balance}")

rows = [
    {
        "age": int(age[i]),
        "income": int(income[i]),
        "credit_score": int(credit_score[i]),
        "loan_amount": int(loan_amount[i]),
        "employment_years": int(employment_years[i]),
        "gender": str(genders[i]),
        "education": str(education[i]),
        "loan_approved": "Yes" if approved[i] else "No",
    }
    for i in range(n)
]

# Inject missing values (~12%)
missing = {
    "income": rng.random(n) < 0.12,
    "credit_score": rng.random(n) < 0.12,
    "gender": rng.random(n) < 0.12,
    "employment_years": rng.random(n) < 0.12,
}
for i, row in enumerate(rows):
    for col, mask in missing.items():
        if mask[i]:
            row[col] = ""

out = "sample_data/loan_approval.csv"
with open(out, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

balance = {}
for row in rows:
    balance[row["loan_approved"]] = balance.get(row["loan_approved"], 0) + 1
print(f"Wrote {out}: {n} rows x {len(rows[0])} cols, class balance = {balance}")
