"""
Expanding Window Backtest Engine

Implements walk-forward evaluation for cross-sectional
insurance prediction models.
"""

import numpy as np
import pandas as pd
from models.frequency import FrequencyModel
from models.severity import SeverityModel

import logging
logger = logging.getLogger(__name__)

def expanding_window_backtest(df, start_period=2):
    results = []

    periods = sorted(df["time"].unique())

    for t in periods[start_period:]:

        train = df[df["time"] < t]
        test = df[df["time"] == t]

        features = ["age", "vehicle_value", "risk_score"]

        # -----------------------
        # FREQUENCY
        # -----------------------

        freq_model = FrequencyModel()

        X_train_freq = train[features]
        y_train_freq = train["claims"]

        X_test_freq = test[features]
        y_test_freq = test["claims"]

        freq_model.fit_glm(X_train_freq, y_train_freq)
        glm_freq_preds = freq_model.predict_glm(X_test_freq)
        glm_freq_score = freq_model.evaluate(y_test_freq, glm_freq_preds)

        freq_model.fit_lgbm(X_train_freq, y_train_freq)
        lgbm_freq_preds = freq_model.predict_lgbm(X_test_freq)
        lgbm_freq_score = freq_model.evaluate(y_test_freq, lgbm_freq_preds)

        # -----------------------
        # SEVERITY (only claims > 0)
        # -----------------------

        train_sev = train[train["claims"] > 0]
        test_sev = test[test["claims"] > 0]

        sev_model = SeverityModel()

        if len(train_sev) > 0 and len(test_sev) > 0:

            X_train_sev = train_sev[features]
            y_train_sev = train_sev["severity"]

            X_test_sev = test_sev[features]
            y_test_sev = test_sev["severity"]

            sev_model.fit_glm(X_train_sev, y_train_sev)
            glm_sev_preds = sev_model.predict_glm(X_test_sev)
            glm_sev_score = sev_model.evaluate(y_test_sev, glm_sev_preds)

            sev_model.fit_lgbm(X_train_sev, y_train_sev)
            lgbm_sev_preds = sev_model.predict_lgbm(X_test_sev)
            lgbm_sev_score = sev_model.evaluate(y_test_sev, lgbm_sev_preds)

        else:
            glm_sev_score = np.nan
            lgbm_sev_score = np.nan

        logger.info(
            f"Period {t} | "
            f"Freq GLM: {glm_freq_score:.4f} | "
            f"Freq LGBM: {lgbm_freq_score:.4f} | "
            f"Sev GLM: {glm_sev_score:.4f} | "
            f"Sev LGBM: {lgbm_sev_score:.4f}"
        )

        results.append({
            "period": t,
            "glm_freq_dev": glm_freq_score,
            "lgbm_freq_dev": lgbm_freq_score,
            "glm_sev_dev": glm_sev_score,
            "lgbm_sev_dev": lgbm_sev_score,
        })

    return pd.DataFrame(results)