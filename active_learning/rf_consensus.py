import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


class RF(RandomForestRegressor):
    def predict(self, X) -> np.ndarray:
        res = []

        for dt in self.estimators_:
            dt: DecisionTreeRegressor
            res.append(dt.predict(X))

        res = np.array(res).T

        return res
