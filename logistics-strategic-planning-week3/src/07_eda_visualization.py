"""
Week 3: Advanced Data Analysis and Visualization in Logistics
Performs EDA (central tendencies, distributions, correlations) on the
shared shipment dataset and produces the visualizations referenced in
the Week 3 report.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")
ACCENT = "#1F4E79"
PALETTE = ["#1F4E79", "#2E75B6", "#5B9BD5", "#9DC3E6", "#BDD7EE"]

df = pd.read_csv("shipment_dataset.csv", parse_dates=["order_date"])
OUT = "./week3_assets"
import os
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------
# Central tendency / summary statistics
# ---------------------------------------------------------------------
summary = df[["distance_km", "order_volume", "delivery_hours", "transport_cost"]].describe()
summary.to_csv(f"{OUT}/summary_stats.csv")
print(summary)

corr = df[["distance_km", "order_volume", "delivery_hours", "transport_cost"]].corr()
print(corr)

# ---------------------------------------------------------------------
# Fig 1: Distribution of delivery hours
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
sns.histplot(df["delivery_hours"], bins=40, kde=True, color=ACCENT, ax=ax)
ax.set_title("Distribution of Delivery Time (Hours)", fontsize=13, weight="bold", color=ACCENT)
ax.set_xlabel("Delivery Hours")
ax.set_ylabel("Number of Shipments")
plt.tight_layout()
plt.savefig(f"{OUT}/fig1_delivery_hours_distribution.png")
plt.close()

# ---------------------------------------------------------------------
# Fig 2: Delivery hours by warehouse (boxplot)
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
order = df.groupby("origin_warehouse")["delivery_hours"].median().sort_values().index
sns.boxplot(data=df, x="origin_warehouse", y="delivery_hours", order=order, palette=PALETTE, ax=ax)
ax.set_title("Delivery Time by Origin Warehouse", fontsize=13, weight="bold", color=ACCENT)
ax.set_xlabel("Warehouse")
ax.set_ylabel("Delivery Hours")
plt.tight_layout()
plt.savefig(f"{OUT}/fig2_delivery_by_warehouse.png")
plt.close()

# ---------------------------------------------------------------------
# Fig 3: Distance vs delivery hours (scatter, colored by warehouse)
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
sns.scatterplot(
    data=df, x="distance_km", y="delivery_hours", hue="origin_warehouse",
    palette=PALETTE, alpha=0.6, s=28, ax=ax
)
ax.set_title("Delivery Distance vs. Delivery Time", fontsize=13, weight="bold", color=ACCENT)
ax.set_xlabel("Distance (km)")
ax.set_ylabel("Delivery Hours")
ax.legend(title="Warehouse", fontsize=8, title_fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/fig3_distance_vs_delivery.png")
plt.close()

# ---------------------------------------------------------------------
# Fig 4: Correlation heatmap
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", square=True, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title("Correlation Matrix — Key Logistics Variables", fontsize=12, weight="bold", color=ACCENT)
plt.tight_layout()
plt.savefig(f"{OUT}/fig4_correlation_heatmap.png")
plt.close()

# ---------------------------------------------------------------------
# Fig 5: Average transport cost by warehouse (bar)
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
avg_cost = df.groupby("origin_warehouse")["transport_cost"].mean().sort_values(ascending=False)
sns.barplot(x=avg_cost.index, y=avg_cost.values, palette=PALETTE, ax=ax)
ax.set_title("Average Transport Cost by Warehouse", fontsize=13, weight="bold", color=ACCENT)
ax.set_xlabel("Warehouse")
ax.set_ylabel("Average Cost ($)")
for i, v in enumerate(avg_cost.values):
    ax.text(i, v + 0.5, f"${v:.0f}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/fig5_avg_cost_by_warehouse.png")
plt.close()

# ---------------------------------------------------------------------
# Fig 6: Daily order volume trend over time
# ---------------------------------------------------------------------
daily = df.groupby(df["order_date"].dt.to_period("W"))["order_volume"].sum()
daily.index = daily.index.to_timestamp()
fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
ax.plot(daily.index, daily.values, color=ACCENT, linewidth=2, marker="o", markersize=3)
ax.set_title("Weekly Order Volume Trend", fontsize=13, weight="bold", color=ACCENT)
ax.set_xlabel("Week")
ax.set_ylabel("Total Units Ordered")
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig(f"{OUT}/fig6_weekly_volume_trend.png")
plt.close()

print("charts saved")

# print key EDA numbers used in the report narrative
print("\n--- KEY STATS ---")
print("Mean delivery hours:", df["delivery_hours"].mean())
print("Median delivery hours:", df["delivery_hours"].median())
print("Std delivery hours:", df["delivery_hours"].std())
print("Delay rate:", df["delay_flag"].mean())
print("Corr distance vs delivery_hours:", corr.loc["distance_km", "delivery_hours"])
print("Corr order_volume vs transport_cost:", corr.loc["order_volume", "transport_cost"])
print(df.groupby("origin_warehouse")["delivery_hours"].median().sort_values())
print(df.groupby("origin_warehouse")["transport_cost"].mean().sort_values(ascending=False))
