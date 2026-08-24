"""
Generates a synthetic, but realistic, shipment-level dataset consistent
with the Week 1/Week 2 scenario: a 3PL provider operating 5 warehouses.
Used as the shared dataset for Week 3 (EDA/visualization) and
Week 4 (predictive modeling).
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

N = 2000
warehouses = ["WH-North", "WH-South", "WH-East", "WH-West", "WH-Central"]
zones = ["Zone-A", "Zone-B", "Zone-C", "Zone-D", "Zone-E", "Zone-F"]

wh_base_distance = {"WH-North": 38, "WH-South": 42, "WH-East": 30, "WH-West": 50, "WH-Central": 22}
wh_efficiency = {"WH-North": 1.0, "WH-South": 1.15, "WH-East": 0.9, "WH-West": 1.25, "WH-Central": 0.85}

dates = pd.date_range("2026-01-01", periods=180, freq="D")

rows = []
for i in range(N):
    wh = rng.choice(warehouses, p=[0.24, 0.20, 0.22, 0.16, 0.18])
    zone = rng.choice(zones)
    order_date = rng.choice(dates)
    dow = pd.Timestamp(order_date).dayofweek  # 0=Mon
    order_volume = max(1, int(rng.normal(45, 18)))
    distance_km = max(3, rng.normal(wh_base_distance[wh], 12))

    # weekend + higher volume + longer distance + warehouse inefficiency -> more delay
    weekend_factor = 1.25 if dow >= 5 else 1.0
    base_hours = (
        6
        + distance_km * 0.18
        + order_volume * 0.05
        + rng.normal(0, 3)
    ) * wh_efficiency[wh] * weekend_factor
    delivery_hours = max(1.5, base_hours)

    # occasional extreme delay events (traffic/breakdown)
    if rng.random() < 0.03:
        delivery_hours += rng.uniform(15, 40)

    transport_cost = 8 + distance_km * 0.55 + order_volume * 0.35 + rng.normal(0, 4)
    transport_cost = max(5, transport_cost)

    delay_flag = delivery_hours > 24

    rows.append({
        "shipment_id": f"SHP{i:05d}",
        "order_date": pd.Timestamp(order_date),
        "day_of_week": pd.Timestamp(order_date).day_name(),
        "origin_warehouse": wh,
        "destination_zone": zone,
        "distance_km": round(distance_km, 2),
        "order_volume": order_volume,
        "delivery_hours": round(delivery_hours, 2),
        "transport_cost": round(transport_cost, 2),
        "delay_flag": delay_flag,
    })

df = pd.DataFrame(rows)
df.to_csv("shipment_dataset.csv", index=False)
print(df.shape)
print(df.head())
print(df.describe())
