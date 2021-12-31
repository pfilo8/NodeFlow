import catboost
import numpy as np

from scipy.stats import multivariate_normal


class IndependentCatBoost:
    def __init__(self, params_tree=None, params_ngboost=None):
        self.params_tree = params_tree
        self.params_ngboost = params_ngboost
        self.models = []

    def fit(self, x, y):
        for i in range(y.shape[1]):
            self.models.append(catboost.CatBoostRegressor(**self.params_ngboost))
            self.models[i].fit(
                X=x,
                y=y[:, i]
            )
        return self

    def scipy_distribution(self, X_test, cmat_output=False):
        pred_dists = [
            m.predict(X_test) for m in self.models
        ]

        means = np.concatenate([dist[:, 0].reshape(-1, 1) for dist in pred_dists], axis=1)
        vs = np.concatenate([dist[:, 1].reshape(-1, 1) for dist in pred_dists], axis=1)
        cmat = [np.diag(vs[i, :]) for i in range(vs.shape[0])]

        if cmat_output:
            out = [means, np.stack(cmat)]
        else:
            out = [
                multivariate_normal(means[i, :], cov=cmat[i])
                for i in range(vs.shape[0])
            ]
        return out
