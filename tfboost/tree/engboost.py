import ngboost as ng
import numpy as np

from sklearn.preprocessing import OneHotEncoder


class EmbeddableNGBoost(ng.NGBoost):

    def fit(
            self,
            X,
            Y,
            X_val=None,
            Y_val=None,
            sample_weight=None,
            val_sample_weight=None,
            train_loss_monitor=None,
            val_loss_monitor=None,
            early_stopping_rounds=None,
    ):
        super().fit(
            X,
            Y,
            X_val=X_val,
            Y_val=Y_val,
            sample_weight=sample_weight,
            val_sample_weight=val_sample_weight,
            train_loss_monitor=train_loss_monitor,
            val_loss_monitor=val_loss_monitor,
            early_stopping_rounds=early_stopping_rounds
        )
        leafs = self.apply(X)
        self._fit_encoder(leafs)
        return self

    def apply(self, X):
        mu_models = [n[0] for n in self.base_models]
        logvar_models = [n[1] for n in self.base_models]

        mu_leafs = np.vstack([m.apply(X) for m in mu_models]).T
        logvar_leafs = np.vstack([m.apply(X) for m in logvar_models]).T
        return [mu_leafs, logvar_leafs]

    def _fit_encoder(self, leafs):
        mu_leafs, logvar_leafs = leafs
        self._mu_leafs_encoder = OneHotEncoder(sparse=False).fit(mu_leafs)
        self._logvar_leafs_encoder = OneHotEncoder(sparse=False).fit(logvar_leafs)

    def embed(self, X):
        leafs = self.apply(X)

        mu_leafs, logvar_leafs = leafs
        mu_leafs_encoded = self._mu_leafs_encoder.transform(mu_leafs)
        logvar_leafs_encoded = self._logvar_leafs_encoder.transform(logvar_leafs)

        embedding = np.hstack([mu_leafs_encoded, logvar_leafs_encoded])
        return embedding

    def pred_dist_param(self, X):
        return None
