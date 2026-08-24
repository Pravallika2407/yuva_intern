"""
Stage 3b: Predictive Modeling - Warehouse/Zone Segmentation (Clustering)
Groups delivery zones by demand density, distance, and delay risk to
inform warehouse-to-zone assignment and route planning.
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

zones = pd.read_csv("zone_summary.csv")

features = ["avg_daily_orders", "avg_delivery_distance_km", "delay_rate"]
X_scaled = StandardScaler().fit_transform(zones[features])

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
zones["segment"] = kmeans.fit_predict(X_scaled)

print(zones.groupby("segment")[features].mean())
