# Predictive-individual-assignment

# Hotel Booking Cancellation Prediction
### MSIN0097: Predictive Analytics 25/26 — Individual Assignment
**Anonymous Candidate Number:** TNKC5

---

## Project Overview
This project builds an end-to-end machine learning pipeline to predict hotel booking 
cancellations using a dataset of 119,390 booking records from a City Hotel and a Resort 
Hotel. The pipeline covers data cleaning, exploratory data analysis, feature engineering, 
model training, fine-tuning, error analysis, and business insights.

The final deployed model (XGBoost, threshold-tuned at t = 0.5165) achieves:
- **Cancellation Recall:** 0.708
- **F1 Score:** 0.616
- **ROC-AUC:** 0.833

---

## Repository Structure
```
├── assignment.ipynb        # Main notebook (all sections)
├── hotel_bookings.csv      # Dataset (not included — see below)
├── README.md               # This file
└── requirements.txt        # Python dependencies
```

---

## Dataset
The dataset is sourced from the publicly available Hotel Booking Demand dataset.  
Place `hotel_bookings.csv` in the same directory as `assignment.ipynb` before running.

---

## How to Run
1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Place `hotel_bookings.csv` in the project directory
4. Open and run `assignment.ipynb` from top to bottom

> **Note:** TabNet requires sufficient CPU/GPU memory.  
> Full notebook runtime is approximately 15–30 minutes.

---

## Pipeline Summary

| Section | Description |
|---------|-------------|
| 1. Problem Framing | Define target variable, success metrics, assumptions, agent tooling plan |
| 2. EDA | Correlation analysis, skewness analysis, cancellation vs key features |
| 3. Feature Engineering | 6 new features engineered, VIF-based pruning to final 14 features |
| 4. Model Selection | Random Forest (baseline), XGBoost, TabNet |
| 5. Fine-tuning & Error Analysis | SMOTE, hyperparameter tuning, threshold tuning, error analysis |
| 6. Final Solution | Model recommendation, cost saving calculation, business insights |

---

## Key Results

| Model | Accuracy | Recall | F1 | ROC-AUC |
|-------|----------|--------|-----|---------|
| Random Forest (baseline) | 0.791 | 0.452 | 0.543 | 0.831 |
| XGBoost | 0.718 | 0.779 | 0.603 | 0.832 |
| TabNet | 0.788 | 0.448 | 0.540 | 0.828 |
| **XGBoost Tuned (t=0.5165)** | **0.757** | **0.708** | **0.616** | **0.833** |
| Ensemble | 0.770 | 0.654 | 0.610 | 0.833 |

---

## Agent Tooling
Codex and Claude were used as coding agents throughout this project.  
All agent outputs were verified manually — cleaning logic validated against dataset 
statistics, model suggestions confirmed through experimental comparison.  
Full decision register documented in report Appendix 7.1.

---

## Dependencies
See `requirements.txt` for full list of dependencies.
