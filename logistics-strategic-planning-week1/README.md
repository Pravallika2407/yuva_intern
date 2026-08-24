# Logistics Strategic Planning — Week 1

Supporting code for the Week 1 task: **Strategic Planning and Data Exploration in Logistics** (Logistics Data Analyst internship track).

This repo contains the Python pseudocode/scripts referenced in the accompanying strategic planning report (`Week1_Strategic_Planning_Logistics_Report.docx`), illustrating the proposed end-to-end analytics approach for a mid-size 3PL provider.

## Scenario

A mid-size third-party logistics (3PL) provider operating a five-warehouse regional distribution network currently plans routes manually and replenishes inventory from historical averages rather than demand forecasts. This leads to inefficient routing, inventory imbalances, and limited visibility into delivery delay drivers.

## KPIs Tracked

- **On-Time Delivery Rate (OTD)** — % of shipments delivered within the promised window
- **Inventory Turnover Ratio** — cost of goods moved vs. average inventory held
- **Cost per Mile Delivered** — total fleet operating cost / total delivery miles

## Strategic Roadmap & Scripts

| Stage | Script | Technique |
|---|---|---|
| 1. Data Collection | — | Aggregation from ERP/WMS/TMS systems and public datasets |
| 2. Data Cleaning | [`src/01_data_cleaning.py`](src/01_data_cleaning.py) | pandas |
| 3. Exploratory Data Analysis | [`src/02_exploratory_analysis.py`](src/02_exploratory_analysis.py) | pandas, matplotlib |
| 4a. Demand Forecasting | [`src/03_demand_forecasting.py`](src/03_demand_forecasting.py) | Regression (GradientBoostingRegressor) |
| 4b. Warehouse/Zone Segmentation | [`src/04_warehouse_segmentation.py`](src/04_warehouse_segmentation.py) | Clustering (KMeans) |
| 4c. Route Optimization | [`src/05_route_optimization.py`](src/05_route_optimization.py) | Optimization (OR-Tools VRP) |
| 5. Evaluation & Reporting | — | Benchmarking against baseline KPIs |

## Setup

```bash
pip install -r requirements.txt
```

Scripts are illustrative pseudocode intended to accompany the strategic report; they assume cleaned, tabular shipment/inventory CSVs (`shipment_history.csv`, `daily_demand.csv`, `zone_summary.csv`) as input, which are not included in this repo.

## Report

The full strategic planning report — including scenario definition, KPI rationale, literature/data research, and the strategic roadmap — is submitted separately as a Word document per the task deliverables.
