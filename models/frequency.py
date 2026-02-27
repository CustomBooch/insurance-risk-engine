"""
Frequency Modelling Module

Implements:
- Poisson GLM
- Negative Binomial (optional extension)
- LightGBM regressor (count objective)

Designed for walk-forward expanding window estimation.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance
import lightgbm as lgb


class FrequencyModel:
    def __init__(self):
        self.glm = PoissonRegressor(alpha=0.0, max_iter=1000)
        self.lgbm = None

    def fit_glm(self, X, y):
        self.glm.fit(X, y)

    def predict_glm(self, X):
        return self.glm.predict(X)

    def fit_lgbm(self, X, y):
        self.lgbm = lgb.LGBMRegressor(
            objective="poisson",
            n_estimators=200,
            learning_rate=0.05,
        )
        self.lgbm.fit(X, y)

    def predict_lgbm(self, X):
        return self.lgbm.predict(X)

    @staticmethod
    def evaluate(y_true, y_pred):
        return mean_poisson_deviance(y_true, y_pred)