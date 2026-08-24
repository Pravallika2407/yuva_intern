"""
Week 4: Predictive Modeling and Optimization in Logistics Systems
Trains models to forecast delivery time (hours) from shipment features,
evaluates them, and produces supporting charts.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import OneHotEncoder

sns.set_theme(style="whitegrid", palette="deep")
ACCENT = "#1F4E79"
PALETTE = ["#1F4E79", "#2E75B6", "#5B9BD5", "#9DC3E6", "#BDD7EE"]
OUT = "./week4_assets"
import os
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv("shipment_dataset.csv", parse_dates=["order_date"])

# ---------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------
df["is_weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)
wh_dummies = pd.get_dummies(df["origin_warehouse"], prefix="wh", drop_first=True)
features = pd.concat([df[["distance_km", "order_volume", "is_weekend"]], wh_dummies], axis=1)
target = df["delivery_hours"]

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# ---------------------------------------------------------------------
# Model 1: Linear Regression (baseline)
# ---------------------------------------------------------------------
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

lr_mae = mean_absolute_error(y_test, lr_pred)
lr_rmse = mean_squared_error(y_test, lr_pred) ** 0.5
lr_r2 = r2_score(y_test, lr_pred)
lr_cv = cross_val_score(lr, features, target, cv=5, scoring="r2")

print("Linear Regression -> MAE: %.3f RMSE: %.3f R2: %.3f CV-R2 mean: %.3f" % (
    lr_mae, lr_rmse, lr_r2, lr_cv.mean()))

# ---------------------------------------------------------------------
# Model 2: Random Forest with light hyperparameter tuning
# ---------------------------------------------------------------------
param_grid = {"n_estimators": [100, 200], "max_depth": [4, 6, 8]}
rf_search = GridSearchCV(
    RandomForestRegressor(random_state=42), param_grid, cv=3, scoring="neg_mean_absolute_error"
)
rf_search.fit(X_train, y_train)
rf = rf_search.best_estimator_
rf_pred = rf.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = mean_squared_error(y_test, rf_pred) ** 0.5
rf_r2 = r2_score(y_test, rf_pred)
rf_cv = cross_val_score(rf, features, target, cv=5, scoring="r2")

print("Best RF params:", rf_search.best_params_)
print("Random Forest -> MAE: %.3f RMSE: %.3f R2: %.3f CV-R2 mean: %.3f" % (
    rf_mae, rf_rmse, rf_r2, rf_cv.mean()))

# ---------------------------------------------------------------------
# Fig 1: Actual vs predicted (Random Forest, the better model)
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=150)
ax.scatter(y_test, rf_pred, alpha=0.4, s=22, color=ACCENT)
lims = [min(y_test.min(), rf_pred.min()), max(y_test.max(), rf_pred.max())]
ax.plot(lims, lims, "--", color="#C00000", linewidth=1.5, label="Perfect prediction")
ax.set_xlabel("Actual Delivery Hours")
ax.set_ylabel("Predicted Delivery Hours")
ax.set_title("Random Forest: Actual vs. Predicted Delivery Time", fontsize=12.5, weight="bold", color=ACCENT)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/fig1_actual_vs_predicted.png")
plt.close()

# ---------------------------------------------------------------------
# Fig 2: Feature importance (Random Forest)
# ---------------------------------------------------------------------
importances = pd.Series(rf.feature_importances_, index=features.columns).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=150)
ax.barh(importances.index, importances.values, color=ACCENT)
ax.set_title("Random Forest Feature Importance", fontsize=12.5, weight="bold", color=ACCENT)
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{OUT}/fig2_feature_importance.png")
plt.close()

# ---------------------------------------------------------------------
# Fig 3: Model comparison bar chart (MAE / RMSE)
# ---------------------------------------------------------------------
comp = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "MAE": [lr_mae, rf_mae],
    "RMSE": [lr_rmse, rf_rmse],
})
fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=150)
x = np.arange(len(comp))
width = 0.32
ax.bar(x - width / 2, comp["MAE"], width, label="MAE", color=PALETTE[0])
ax.bar(x + width / 2, comp["RMSE"], width, label="RMSE", color=PALETTE[2])
ax.set_xticks(x)
ax.set_xticklabels(comp["Model"])
ax.set_ylabel("Hours")
ax.set_title("Model Error Comparison", fontsize=12.5, weight="bold", color=ACCENT)
ax.legend()
for i, v in enumerate(comp["MAE"]):
    ax.text(i - width / 2, v + 0.05, f"{v:.2f}", ha="center", fontsize=8.5)
for i, v in enumerate(comp["RMSE"]):
    ax.text(i + width / 2, v + 0.05, f"{v:.2f}", ha="center", fontsize=8.5)
plt.tight_layout()
plt.savefig(f"{OUT}/fig3_model_comparison.png")
plt.close()

print("charts saved")
print(importances)
print(comp)
