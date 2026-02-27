"""
Expanding Window Backtest Engine

Implements walk-forward evaluation for cross-sectional
insurance prediction models.
"""

import numpy as np
import pandas as pd
from models.frequency import FrequencyModel

def expanding_window_backtest(df, start_period=2):
    results = []

    periods = sorted(df["time"].unique())

    for t in periods[start_period:]:

        train = df[df["time"] < t]
        test = df[df["time"] == t]

        X_train = train[["age", "vehicle_value", "risk_score"]]
        y_train = train["claims"]

        X_test = test[["age", "vehicle_value", "risk_score"]]
        y_test = test["claims"]

        model = FrequencyModel()

        # ---- GLM ----
        model.fit_glm(X_train, y_train)
        glm_preds = model.predict_glm(X_test)
        glm_score = model.evaluate(y_test, glm_preds)

        # ---- LightGBM ----
        model.fit_lgbm(X_train, y_train)
        lgb_preds = model.predict_lgbm(X_test)
        lgb_score = model.evaluate(y_test, lgb_preds)

        results.append({
            "period": t,
            "glm_deviance": glm_score,
            "lgbm_deviance": lgb_score
        })

        print(
            f"Period {t} | "
            f"GLM: {glm_score:.4f} | "
            f"LGBM: {lgb_score:.4f}"
        )

    return pd.DataFrame(results)