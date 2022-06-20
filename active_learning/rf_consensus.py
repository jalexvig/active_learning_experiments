import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import BaseDecisionTree


# Note: Doesn't actually make sense to have different classes for clf/reg since underlying decision trees will
# predict proba 0/1 anyways, so might as well predict
class DisaggregatedRFMixin(RandomForestRegressor):
    def predict(self, X) -> np.ndarray:
        res = []

        for dt in self.estimators_:
            dt: BaseDecisionTree
            res.append(dt.predict(X))

        res = np.array(res).T

        return res


class DisaggregatedRFRegressor(DisaggregatedRFMixin, RandomForestRegressor):
    pass


class DisaggregatedRFClassifier(DisaggregatedRFMixin, RandomForestClassifier):
    pass
