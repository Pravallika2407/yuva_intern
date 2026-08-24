# Week 4 — Predictive Modeling and Optimization in Logistics Systems

| Script | Purpose |
|---|---|
| `src/00_generate_dataset.py` | Generates the synthetic shipment dataset (`shipment_dataset.csv`, 2,000 records across 5 warehouses) used as model input. |
| `src/08_predictive_modeling.py` | Trains and evaluates Linear Regression and Random Forest models to forecast delivery time, with grid-search hyperparameter tuning, cross-validation, and comparison/feature-importance charts. |

## Setup

```bash
pip install -r requirements.txt
python src/00_generate_dataset.py     # generates shipment_dataset.csv
python src/08_predictive_modeling.py  # charts saved to week4_assets/
```

Report: `Week4_Predictive_Modeling_Optimization_Report.docx` (submitted separately).
