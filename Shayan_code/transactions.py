# customer_segmentation_rfm.py

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1) Load transactions
tx = pd.read_csv("transactions.csv")
tx["date"] = pd.to_datetime(tx["date"], errors="coerce")
tx = tx.dropna(subset=["customer_id", "date", "amount"]).copy()

# Keep only positive amounts for a clean baseline
tx = tx[tx["amount"] > 0]

# 2) Build RFM features
snapshot_date = tx["date"].max() + pd.Timedelta(days=1)

rfm = tx.groupby("customer_id").agg(
    Recency=("date", lambda x: (snapshot_date - x.max()).days),
    Frequency=("date", "count"),
    Monetary=("amount", "sum")
).reset_index()

# Optional: log transform Monetary/Frequency for stability
rfm["Frequency"] = np.log1p(rfm["Frequency"])
rfm["Monetary"] = np.log1p(rfm["Monetary"])

X = rfm[["Recency", "Frequency", "Monetary"]].copy()

# 3) Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4) Choose K with silhouette (beginner-friendly)
scores = []
k_values = range(2, 8)

for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(X_scaled)
    s = silhouette_score(X_scaled, labels)
    scores.append((k, s))

print("Silhouette scores:", scores)
best_k = max(scores, key=lambda t: t[1])[0]
print("Best k:", best_k)

# 5) Fit final model
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
rfm["segment"] = kmeans.fit_predict(X_scaled)

# 6) Profile segments (so it’s business-readable)
profile = rfm.groupby("segment").agg(
    customers=("customer_id", "nunique"),
    avg_recency=("Recency", "mean"),
    avg_frequency=("Frequency", "mean"),
    avg_monetary=("Monetary", "mean")
).reset_index().sort_values("avg_monetary", ascending=False)

print("\nSegment profile:\n", profile)

rfm.to_csv("customer_segments.csv", index=False)
profile.to_csv("segment_profile.csv", index=False)