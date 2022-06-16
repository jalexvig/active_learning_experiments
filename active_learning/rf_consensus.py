import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class DisaggregatedRFRegressor(RandomForestRegressor):
    def predict(self, X) -> np.ndarray:
        res = []

        for dt in self.estimators_:
            dt: DecisionTreeRegressor
            res.append(dt.predict(X))

        res = np.array(res).T

        return res


class DisaggregatedRFClassifier(RandomForestClassifier):
    def predict_proba(self, X) -> np.ndarray:
        res = []

        for dt in self.estimators_:
            dt: DecisionTreeClassifier
            res.append(dt.predict_proba(X))

        res = np.array(res).T

        return res
