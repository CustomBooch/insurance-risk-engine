import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb


class FrequencyModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.glm = PoissonRegressor(alpha=1.0, max_iter=1000)
        self.lgbm = None

    def fit_glm(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.glm.fit(X_scaled, y)

    def predict_glm(self, X):
        X_scaled = self.scaler.transform(X)
        return self.glm.predict(X_scaled)

    def fit_lgbm(self, X, y):
        self.lgbm = lgb.LGBMRegressor(
            objective="poisson",
            n_estimators=200,
            learning_rate=0.05,
            verbosity=-1
        )
        self.lgbm.fit(X, y)

    def predict_lgbm(self, X):
        return self.lgbm.predict(X)

    @staticmethod
    def evaluate(y_true, y_pred):
        return mean_poisson_deviance(y_true, y_pred)