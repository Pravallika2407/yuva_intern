"""
Stage 1: Data Cleaning
Loads raw shipment history and prepares it for exploratory analysis
and modeling: deduplication, missing-value handling, unit
standardization, and derived fields.
"""

import pandas as pd

shipments = pd.read_csv(
    "shipment_history.csv", parse_dates=["order_time", "delivery_time"]
)

# Drop duplicate shipment records and rows missing critical fields
shipments = shipments.drop_duplicates(subset="shipment_id")
shipments = shipments.dropna(subset=["origin_warehouse", "destination_zone"])

# Standardize distance units and compute delivery duration in hours
shipments["distance_km"] = shipments["distance_mi"] * 1.60934
shipments["delivery_hours"] = (
    shipments["delivery_time"] - shipments["order_time"]
).dt.total_seconds() / 3600

if __name__ == "__main__":
    print(shipments.info())
    print(shipments.head())
