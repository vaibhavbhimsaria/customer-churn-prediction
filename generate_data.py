"""
generate_data.py
----------------
Creates a synthetic customer dataset for churn prediction — mimics a
telecom/subscription business (tenure, monthly charges, contract type,
support calls, etc.) with realistic churn patterns baked in.

Run:
    python generate_data.py
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(21)
N = 5000


def generate():
    customer_id = [f"CUST_{i:05d}" for i in range(1, N + 1)]

    tenure_months = RNG.integers(1, 72, size=N)
    contract_type = RNG.choice(["Month-to-month", "One year", "Two year"], size=N, p=[0.55, 0.25, 0.20])
    monthly_charges = RNG.uniform(300, 3000, size=N).round(2)
    support_calls = RNG.poisson(1.5, size=N)
    payment_delay_days = RNG.integers(0, 30, size=N)
    is_senior_citizen = RNG.choice([0, 1], size=N, p=[0.84, 0.16])
    has_addon_services = RNG.choice([0, 1], size=N, p=[0.5, 0.5])

    # Build churn probability from realistic drivers so the models have
    # genuine signal to learn from (not random noise)
    churn_score = (
        -0.03 * tenure_months
        + 0.35 * support_calls
        + 0.05 * payment_delay_days
        + np.where(contract_type == "Month-to-month", 1.4, 0)
        + np.where(contract_type == "One year", 0.4, 0)
        + 0.0006 * monthly_charges
        - 0.6 * has_addon_services
        + 0.3 * is_senior_citizen
        + RNG.normal(0, 1.0, size=N)  # noise
    )
    churn_prob = 1 / (1 + np.exp(-(churn_score - 2)))
    churned = (RNG.uniform(0, 1, size=N) < churn_prob).astype(int)

    df = pd.DataFrame({
        "customer_id": customer_id,
        "tenure_months": tenure_months,
        "contract_type": contract_type,
        "monthly_charges": monthly_charges,
        "support_calls": support_calls,
        "payment_delay_days": payment_delay_days,
        "is_senior_citizen": is_senior_citizen,
        "has_addon_services": has_addon_services,
        "churned": churned,
    })

    # A little realistic messiness for the cleaning step
    missing_idx = RNG.choice(N, size=int(N * 0.02), replace=False)
    df.loc[missing_idx, "monthly_charges"] = np.nan

    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv("data/customers_raw.csv", index=False)
    print(f"Generated {len(df):,} customer records -> data/customers_raw.csv")
    print(f"Churn rate: {df['churned'].mean() * 100:.1f}%")
