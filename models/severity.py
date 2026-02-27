"""
Severity Modelling Module

Models claim severity conditional on at least one claim.
Implements:
- Gamma GLM (log link)
- LightGBM regressor
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import GammaRegressor
from sklearn.metrics import mean_gamma_deviance
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb


class SeverityModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.glm = GammaRegressor(alpha=1.0, max_iter=1000)
        self.lgbm = None

    def fit_glm(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.glm.fit(X_scaled, y)

    def predict_glm(self, X):
        X_scaled = self.scaler.transform(X)
        return self.glm.predict(X_scaled)

    def fit_lgbm(self, X, y):
        self.lgbm = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=300,
            learning_rate=0.05,
            verbosity=-1
        )
        self.lgbm.fit(X, np.log(y))

    def predict_lgbm(self, X):
        log_preds = self.lgbm.predict(X)
        return np.exp(log_preds)

    @staticmethod
    def evaluate(y_true, y_pred):
        return mean_gamma_deviance(y_true, y_pred)