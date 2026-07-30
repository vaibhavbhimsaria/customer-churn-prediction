# Customer Churn Prediction & Business Intelligence

An end-to-end churn analytics project: data cleaning → feature engineering → SQL-based customer segmentation → three classification models (Logistic Regression, Decision Tree, Random Forest) → full evaluation suite (accuracy, precision, recall, F1, ROC-AUC, confusion matrix) → feature importance / churn-driver analysis → dashboard-ready exports for Power BI.

## Tech Stack
`Python` · `Pandas` · `NumPy` · `SQL (SQLite)` · `Scikit-learn` · `Matplotlib` · `Seaborn`

## Project Structure
```
customer-churn-prediction/
├── generate_data.py                # creates synthetic subscription-customer dataset
├── churn_analysis.py                # cleaning, segmentation, modeling, evaluation — full pipeline
├── requirements.txt
├── data/
│   └── customers_raw.csv
└── output/
    ├── customer_segment_churn_summary.csv
    ├── model_comparison.csv
    ├── confusion_matrix_logistic_regression.png
    ├── confusion_matrix_decision_tree.png
    ├── confusion_matrix_random_forest.png
    ├── roc_curves.png
    ├── feature_importance.csv
    ├── feature_importance.png
    └── customers_with_churn_risk_scores.csv
```

## How to Run
```bash
pip install -r requirements.txt
python generate_data.py      # step 1: generate 5,000 synthetic customer records
python churn_analysis.py     # step 2: clean, segment, train models, evaluate, export
```

## Pipeline Details
1. **Data Cleaning**: Impute missing `monthly_charges` with median, drop duplicate customer records.
2. **Feature Engineering**: `avg_monthly_spend_per_tenure`, `high_support_contact` flag, `is_month_to_month` flag.
3. **SQL-Based Segmentation**: Customers grouped by tenure bracket × contract type via a SQL query (SQLite), with churn rate and average spend per segment — this is the table that feeds the Power BI churn-trend dashboard.
4. **Modeling**: Three classifiers trained and compared — **Logistic Regression**, **Decision Tree**, **Random Forest**.
5. **Evaluation**: Accuracy, Precision, Recall, F1-score, ROC-AUC, and a confusion matrix for every model (see `/output`).
6. **Feature Importance**: Random Forest feature importances used to identify top churn drivers — translated into plain-language business recommendations.
7. **Dashboard Export**: Per-customer churn-risk scores exported to CSV, ready to load into Power BI to build a churn-risk / revenue-at-risk dashboard.

## Sample Results (from this run)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 67.4% | 0.65 | 0.61 | 0.63 | **0.74** |
| Random Forest | 66.9% | 0.65 | 0.59 | 0.62 | 0.73 |
| Decision Tree | 66.0% | 0.65 | 0.53 | 0.58 | 0.70 |

**Top churn drivers** (Random Forest feature importance): average monthly spend relative to tenure, monthly charges, payment delay days, tenure, and month-to-month contract type.

**Key business insight**: Month-to-month customers in their first year churn at **~71%**, versus **~20%** for long-term customers on two-year contracts — a clear signal to prioritize contract-upgrade incentives for new, high-risk customers.

> Note: these are realistic, actually-computed metrics from this synthetic dataset and pipeline — replace them with your own numbers once you run it, and be ready to explain *why* Logistic Regression edged out Random Forest here (simpler decision boundary generalized slightly better on this dataset — a good talking point in interviews about not defaulting to the most complex model).

## Author
**Vaibhav Bhimsaria** — B.Tech Electronics & Communication Engineering, MANIT Bhopal
