"""
Stage 3a: Predictive Modeling - Demand Forecasting (Regression)
Predicts daily order volume per warehouse to support inventory
replenishment decisions and improve the Inventory Turnover Ratio KPI.
"""

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

daily_demand = pd.read_csv("daily_demand.csv")

features = ["order_dow_num", "warehouse_id", "promo_flag", "past_7d_avg_orders"]
X = daily_demand[features]
y = daily_demand["order_volume"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, predictions))
