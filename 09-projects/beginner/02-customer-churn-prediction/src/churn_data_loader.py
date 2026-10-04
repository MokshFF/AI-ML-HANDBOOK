"""Synthetic Customer Churn Dataset Generator."""
import numpy as np
import pandas as pd

def generate_churn_data(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    tenure = np.random.exponential(scale=20, size=n_samples).clip(1, 72).round().astype(int)
    monthly_charges = np.random.normal(65, 25, n_samples).clip(20, 140)
    total_charges = (tenure * monthly_charges * np.random.uniform(0.9, 1.05, n_samples)).round(2)
    contract = np.random.choice(["Month-to-month", "One year", "Two year"], size=n_samples, p=[0.55, 0.25, 0.20])
    tickets = np.random.poisson(lam=1.5, size=n_samples).clip(0, 10)
    payment = np.random.choice(["Credit Card", "Electronic Check", "Bank Transfer"], size=n_samples)
    
    # Churn probability logit
    contract_risk = {"Month-to-month": 1.2, "One year": -0.4, "Two year": -1.5}
    z = (
        -1.5 
        - 0.04 * tenure 
        + 0.02 * monthly_charges 
        + 0.35 * tickets 
        + np.array([contract_risk[c] for c in contract])
    )
    prob = 1.0 / (1.0 + np.exp(-z))
    churn = (np.random.rand(n_samples) < prob).astype(int)
    
    return pd.DataFrame({
        "tenure_months": tenure,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "contract_type": contract,
        "tech_support_tickets": tickets,
        "payment_method": payment,
        "churn": churn
    })
