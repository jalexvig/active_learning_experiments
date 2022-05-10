"""
Gradient boosting regression.

Idea comes from https://towardsdatascience.com/how-to-generate-prediction-intervals-with-scikit-learn-and-python
-ab3899f992ed.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


@dataclass
class GBREnsemble:

    lower_quantile: float = 0.1
    upper_quantile: float = 0.9

    def __post_init__(self):
        self.lower_gbr = GradientBoostingRegressor(
            loss="quantile", alpha=self.lower_quantile
        )
        self.mid_gbr = GradientBoostingRegressor(loss="ls")
        self.upper_gbr = GradientBoostingRegressor(
            loss="quantile", alpha=self.upper_quantile
        )

    def fit(self, X, y):
        self.lower_gbr.fit(X, y)
        self.mid_gbr.fit(X, y)
        self.upper_gbr.fit(X, y)

    def predict(self, X) -> np.ndarray:
        res = np.array(
            [
                self.lower_gbr.predict(X),
                self.mid_gbr.predict(X),
                self.upper_gbr.predict(X),
            ]
        ).T

        return res
