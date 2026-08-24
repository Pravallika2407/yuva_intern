"""
Week 2: Data Collection, Cleaning, and Preprocessing
Full preprocessing pipeline for the 3PL shipment history dataset:
load, deduplicate, handle missing values, flag outliers, normalize
and scale. Produces shipment_history_clean.csv for downstream use
by the Week 1 modeling scripts (03-05).
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------
# 1. Load and inspect
# ---------------------------------------------------------------------
shipments = pd.read_csv(
    "shipment_history_raw.csv", parse_dates=["order_time", "delivery_time"]
)

print(shipments.shape)
print(shipments.isnull().sum())
print(shipments.duplicated(subset="shipment_id").sum())

# ---------------------------------------------------------------------
# 2. Remove duplicates (keep earliest record per shipment_id)
# ---------------------------------------------------------------------
shipments = shipments.sort_values("order_time")
shipments = shipments.drop_duplicates(subset="shipment_id", keep="first")

# ---------------------------------------------------------------------
# 3. Handle missing values
# ---------------------------------------------------------------------
shipments["destination_zone"] = shipments["destination_zone"].fillna("Unknown")

shipments["distance_mi"] = shipments.groupby(
    ["origin_warehouse", "destination_zone"]
)["distance_mi"].transform(lambda s: s.fillna(s.median()))

shipments["distance_mi"] = shipments["distance_mi"].fillna(
    shipments["distance_mi"].median()
)

# ---------------------------------------------------------------------
# 4. Detect and flag outliers (IQR method on delivery duration)
# ---------------------------------------------------------------------
shipments["delivery_hours"] = (
    shipments["delivery_time"] - shipments["order_time"]
).dt.total_seconds() / 3600

q1 = shipments["delivery_hours"].quantile(0.25)
q3 = shipments["delivery_hours"].quantile(0.75)
iqr = q3 - q1
upper_bound = q3 + 1.5 * iqr

shipments["is_outlier"] = shipments["delivery_hours"] > upper_bound

# ---------------------------------------------------------------------
# 5. Normalize units and scale features for clustering
# ---------------------------------------------------------------------
shipments["distance_km"] = shipments["distance_mi"] * 1.60934

cluster_features = ["distance_km", "order_volume"]
scaler = StandardScaler()
shipments[[f"{c}_scaled" for c in cluster_features]] = scaler.fit_transform(
    shipments[cluster_features]
)

shipments.to_csv("shipment_history_clean.csv", index=False)
