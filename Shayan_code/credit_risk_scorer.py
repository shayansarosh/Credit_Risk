# credit_risk_scorecard_baseline.py (or a notebook cell)

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

#Load data
# Replace with your file path
df = pd.read_csv("credit_data.csv")

# Assume 'target' is the binary target column indicating default (1) or not (0)
df = df.dropna(subset=["target"]).copy()

y = df["target"].astype(int)
X = df.drop(columns=["target"])

# Keep numeric only for a clean beginner baseline
X = X.select_dtypes(include=[np.number]).fillna(0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#Model pipeline (scaling + logistic regression)
pipe = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))
])

pipe.fit(X_train, y_train)

#Evaluate
proba = pipe.predict_proba(X_test)[:, 1]  # P(default=1)
auc = roc_auc_score(y_test, proba)
print("AUC:", round(auc, 4))

# Choose a threshold (beginner: start with 0.5; later tune for business needs)
threshold = 0.5
pred = (proba >= threshold).astype(int)

print("\nConfusion Matrix:\n", confusion_matrix(y_test, pred))
print("\nReport:\n", classification_report(y_test, pred, digits=3))

# Convert probability to a "score" (simple scorecard-style scaling)
# Higher score = lower default risk
# Common simple mapping:
# score = offset + factor * logit(p_good / p_bad)
# Here we use logit(p_default) then invert it to make higher = better.

eps = 1e-6
p = np.clip(proba, eps, 1-eps)
logit_default = np.log(p / (1 - p))

# Scale settings (you can tweak)
BASE_SCORE = 600
FACTOR = 40  # larger = wider spread of scores

score = BASE_SCORE - FACTOR * logit_default  # invert so higher = safer

out = pd.DataFrame({
    "p_default": proba,
    "score": score,
    "y_true": y_test.values
})

# Risk bands (simple, explainable)
out["band"] = pd.cut(out["score"],
                     bins=[-np.inf, 520, 580, 640, np.inf],
                     labels=["High Risk", "Medium Risk", "Low Risk", "Very Low Risk"])

band_summary = out.groupby("band").agg(
    count=("score", "size"),
    avg_score=("score", "mean"),
    avg_p_default=("p_default", "mean"),
    default_rate=("y_true", "mean")
).reset_index()

print("\nBand summary:\n", band_summary)
out.to_csv("scored_customers_sample.csv", index=False)

# Optional) Feature impact table for interpretability
lr = pipe.named_steps["lr"]
feature_impact = pd.DataFrame({
    "feature": X.columns,
    "coef": lr.coef_[0]
}).sort_values("coef", ascending=False)

print("\nTop positive coefs (increase default odds):\n", feature_impact.head(10))
print("\nTop negative coefs (reduce default odds):\n", feature_impact.tail(10))