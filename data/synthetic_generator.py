"""
Synthetic Insurance Panel Generator

Generates panel data with:
- Heterogeneous policyholders
- Count-based claim frequency (Poisson)
- Heavy-tailed claim severity (Lognormal)
- Time dimension for walk-forward testing
"""

import numpy as np
import pandas as pd


def generate_synthetic_panel(
    n_policies: int = 5000,
    n_periods: int = 8,
    random_state: int = 42,
):
    np.random.seed(random_state)

    policy_ids = np.arange(n_policies)

    records = []

    for t in range(n_periods):

        # Static characteristics
        age = np.random.normal(45, 12, n_policies)
        vehicle_value = np.random.lognormal(mean=9, sigma=0.5, size=n_policies)
        risk_score = np.random.uniform(0, 1, n_policies)

        # Latent risk intensity
        lambda_ = np.exp(
            -3
            + 0.02 * age
            + 0.000002 * vehicle_value
            + 1.5 * risk_score
        )

        claims = np.random.poisson(lambda_)

        # Severity only if claim occurs
        severity = np.where(
            claims > 0,
            np.random.lognormal(mean=8, sigma=1.0, size=n_policies),
            0.0,
        )

        for i in range(n_policies):
            records.append(
                {
                    "policy_id": policy_ids[i],
                    "time": t,
                    "age": age[i],
                    "vehicle_value": vehicle_value[i],
                    "risk_score": risk_score[i],
                    "claims": claims[i],
                    "severity": severity[i],
                }
            )

    df = pd.DataFrame(records)
    df["pure_premium"] = df["claims"] * df["severity"]

    return df