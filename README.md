# Hotel Booking Cancellation Prediction

**Course**: MSIN0097 Predictive Analytics | UCL School of Management  
**Author**: Kuei-Chun Liu  
**Tools**: Python · XGBoost · Random Forest · TabNet · SMOTE · scikit-learn

---

## Overview

An end-to-end ML pipeline to predict hotel booking cancellations using 119,390 records from a City Hotel and Resort Hotel. The pipeline covers data cleaning, EDA, feature engineering, model training, fine-tuning, error analysis, and business insights.

**Final model**: XGBoost threshold-tuned at t = 0.5165
- Cancellation Recall: **0.708**
- F1 Score: **0.616**
- ROC-AUC: **0.833**
- Estimated recoverable revenue: **€1,315,813** (70.8% of cancellations detected)

---

## Contents

| File | Description |
|---|---|
| `assignment.ipynb` | Full end-to-end pipeline notebook |
| `hotel_bookings.csv` | Hotel booking dataset (119,390 records) |
| `requirements.txt` | Python dependencies |

---

## Pipeline

| Section | Description |
|---|---|
| 1. Problem Framing | Business objective, success metrics, data leakage prevention |
| 2. EDA | Correlation heatmap, skewness analysis, cancellation vs key features |
| 3. Feature Engineering | 6 engineered features, VIF pruning to final 14 features |
| 4. Model Selection | Random Forest (baseline), XGBoost, TabNet |
| 5. Fine-tuning | SMOTE, hyperparameter tuning, threshold optimisation, error analysis |
| 6. Business Insights | Segment recommendations, cost saving calculation |

---

## Model Results

| Model | Accuracy | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Random Forest (baseline) | 0.791 | 0.452 | 0.543 | 0.831 |
| XGBoost | 0.718 | 0.779 | 0.603 | 0.832 |
| TabNet | 0.788 | 0.448 | 0.540 | 0.828 |
| **XGBoost Tuned (t=0.5165)** | **0.757** | **0.708** | **0.616** | **0.833** |
| Ensemble (Soft Voting) | 0.770 | 0.654 | 0.610 | 0.833 |

---

## Key Findings

- **Lead time** is the strongest cancellation predictor (RF importance 0.203) — risk escalates from 8.4% (last-minute) to 36.9% (long-term bookings)
- **Parking requests** are the most reliable commitment proxy — near-zero cancellation rate; dominates XGBoost importance (0.407)
- **No Deposit bookings** are the hardest segment to predict (F1 0.599), comprising 98% of the test set
- SMOTE improved Recall across all models but did not improve ROC-AUC, confirming it shifts decision boundaries without improving discriminative ability

---

## How to Run

1. Clone the repo and install dependencies:
```bash
pip install -r requirements.txt
```
2. Place `hotel_bookings.csv` in the project directory
3. Open `assignment.ipynb` and run all cells top to bottom

> ⚠️ TabNet requires sufficient CPU/GPU memory. Full runtime ~15–30 minutes.
