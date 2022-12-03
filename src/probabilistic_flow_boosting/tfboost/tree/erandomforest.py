import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder


class EmbeddableRandomForest(RandomForestRegressor):

    def fit(self, X, y, sample_weight=None):
        super().fit(X, y, sample_weight=None)
        leafs = self.apply(X)
        self._fit_encoder(leafs)
        self._y_dims = y.shape[1] if len(y.shape) == 2 else 1  # Improve that!
        return self

    def _fit_encoder(self, leafs):
        self._leafs_encoder = OneHotEncoder(
            categories=[range(leafs.max() + 1) for _ in range(leafs.shape[1])],
            drop='if_binary',
            sparse=False
        ).fit(leafs)

    def _transform_encoder(self, leafs):
        return self._leafs_encoder.transform(leafs)

    def embed(self, X):
        leafs = self.apply(X)
        embeddings = self._transform_encoder(leafs)
        return embeddings

    def pred_dist_param(self, X):
        """ Method for predicting distribution parameters for CNF module. Here Normal distribution (0, 1)."""
        return np.zeros((X.shape[0], 2 * self._y_dims))

    def extract_thresholds(self) -> np.ndarray:
        """ Method for extracting threshold values on leaves in trees."""
        return np.array([
            tree.tree_.threshold[tree.tree_.feature > -1]
            for tree in self.estimators_
        ])

    def update_thresholds(self, thresholds: np.ndarray):
        """ Method for updating thresholds in trees.

        Please keep in mind that this method destroys other statistics of trees, i.e., they are no longer valid.
        Example of such a statistics: impurity, samples.

        thresholds: np.ndarray - [n_estimators, 2**(max_depth-1) + 1] array of new thresholds; formula for second
                    dimension comes from the fact that we take all nodes except leaves and assume fully grown trees.
        """
        assert thresholds.shape[0] == len(self.estimators_)
        assert thresholds.shape[1] == 2 ** (self.max_depth - 1) + 1

        for idx, row in enumerate(thresholds):
            self.estimators_[idx].tree_.threshold[self.estimators_[idx].tree_.feature > -1] = row
