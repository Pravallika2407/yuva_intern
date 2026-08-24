"""
Stage 2: Exploratory Data Analysis
Profiles delivery delay patterns to surface seasonality and
operational bottlenecks ahead of modeling.
"""

import matplotlib.pyplot as plt
import pandas as pd

# Assumes `shipments` has already been produced by 01_data_cleaning.py
shipments = pd.read_csv(
    "shipment_history_clean.csv", parse_dates=["order_time", "delivery_time"]
)

# Delivery delay patterns by day of week
shipments["order_dow"] = shipments["order_time"].dt.day_name()
delay_by_day = shipments.groupby("order_dow")["delivery_hours"].mean()

delay_by_day.plot(kind="bar", title="Average Delivery Time by Day of Week")
plt.ylabel("Hours")
plt.tight_layout()
plt.show()
