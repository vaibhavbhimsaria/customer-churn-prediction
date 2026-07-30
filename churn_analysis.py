"""
churn_analysis.py
------------------
Full customer churn pipeline:
    1. Data cleaning
    2. Feature engineering
    3. SQL-based customer segmentation (via SQLite)
    4. Model training: Logistic Regression, Decision Tree, Random Forest
    5. Evaluation: accuracy, precision, recall, F1, ROC-AUC, confusion matrix
    6. Feature importance analysis (churn drivers)
    7. Dashboard-ready exports for Power BI

Run:
    python churn_analysis.py
"""

import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay,
)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# 1. Data cleaning
# ---------------------------------------------------------------------------
df = pd.read_csv("data/customers_raw.csv")
before = len(df)
df["monthly_charges"] = df["monthly_charges"].fillna(df["monthly_charges"].median())
df = df.drop_duplicates(subset=["customer_id"])
print(f"[CLEAN] {before:,} -> {len(df):,} rows (missing values imputed, duplicates removed)")


# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------
df["avg_monthly_spend_per_tenure"] = df["monthly_charges"] / df["tenure_months"].replace(0, 1)
df["high_support_contact"] = (df["support_calls"] >= 3).astype(int)
df["is_month_to_month"] = (df["contract_type"] == "Month-to-month").astype(int)


# ---------------------------------------------------------------------------
# 3. SQL-based customer segmentation (via SQLite — same pattern used to
#    segment customers for the churn dashboard / stakeholder reporting)
# ---------------------------------------------------------------------------
conn = sqlite3.connect(":memory:")
df.to_sql("customers", conn, index=False, if_exists="replace")

segment_query = """
    SELECT
        CASE
            WHEN tenure_months < 12 THEN 'New (<1yr)'
            WHEN tenure_months < 36 THEN 'Established (1-3yr)'
            ELSE 'Long-term (3yr+)'
        END AS tenure_segment,
        contract_type,
        COUNT(*) AS num_customers,
        ROUND(AVG(churned) * 100, 1) AS churn_rate_pct,
        ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges
    FROM customers
    GROUP BY tenure_segment, contract_type
    ORDER BY churn_rate_pct DESC;
"""
segment_summary = pd.read_sql_query(segment_query, conn)
segment_summary.to_csv("output/customer_segment_churn_summary.csv", index=False)
print("\n[SEGMENTATION] Customer segments by tenure & contract type:")
print(segment_summary.to_string(index=False))
conn.close()


# ---------------------------------------------------------------------------
# 4. Prepare features for modeling
# ---------------------------------------------------------------------------
feature_cols = [
    "tenure_months", "monthly_charges", "support_calls", "payment_delay_days",
    "is_senior_citizen", "has_addon_services", "avg_monthly_spend_per_tenure",
    "high_support_contact", "is_month_to_month",
]
X = df[feature_cols]
y = df["churned"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ---------------------------------------------------------------------------
# 5. Train models
# ---------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=RANDOM_STATE),
}

results = []
roc_data = {}

for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    results.append({"Model": name, "Accuracy": acc, "Precision": prec,
                     "Recall": rec, "F1_Score": f1, "ROC_AUC": auc})
    roc_data[name] = roc_curve(y_test, y_proba)

    # Confusion matrix for each model
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    plt.savefig(f"output/confusion_matrix_{name.replace(' ', '_').lower()}.png")
    plt.close()

results_df = pd.DataFrame(results).round(3).sort_values("ROC_AUC", ascending=False)
results_df.to_csv("output/model_comparison.csv", index=False)
print("\n[MODEL RESULTS]")
print(results_df.to_string(index=False))


# ---------------------------------------------------------------------------
# 6. ROC curves (all models on one chart)
# ---------------------------------------------------------------------------
plt.figure(figsize=(7, 5.5))
for name, (fpr, tpr, _) in roc_data.items():
    auc = results_df.loc[results_df["Model"] == name, "ROC_AUC"].values[0]
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — Model Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("output/roc_curves.png")
plt.close()


# ---------------------------------------------------------------------------
# 7. Feature importance (churn drivers) — from Random Forest, the best model
# ---------------------------------------------------------------------------
best_model = models["Random Forest"]
importances = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
importances.to_csv("output/feature_importance.csv", header=["importance"])

plt.figure(figsize=(7, 5))
sns.barplot(x=importances.values, y=importances.index, hue=importances.index,
            palette="viridis", legend=False)
plt.xlabel("Importance")
plt.title("Key Churn Drivers — Random Forest Feature Importance")
plt.tight_layout()
plt.savefig("output/feature_importance.png")
plt.close()

print("\n[FEATURE IMPORTANCE] Top churn drivers:")
print(importances.head(5).to_string())


# ---------------------------------------------------------------------------
# 8. Dashboard-ready summary export (for Power BI)
# ---------------------------------------------------------------------------
df["churn_risk_score"] = best_model.predict_proba(df[feature_cols])[:, 1].round(3)
df.to_csv("output/customers_with_churn_risk_scores.csv", index=False)

best_row = results_df.iloc[0]
print("\nKEY INSIGHTS")
print(f"- Best model: {best_row['Model']} (Accuracy={best_row['Accuracy']:.1%}, "
      f"F1={best_row['F1_Score']:.2f}, ROC-AUC={best_row['ROC_AUC']:.2f})")
print(f"- Top churn driver: '{importances.index[0]}'")
print("- All outputs (model comparison, confusion matrices, ROC curves, "
      "feature importance, per-customer risk scores) saved to /output")
