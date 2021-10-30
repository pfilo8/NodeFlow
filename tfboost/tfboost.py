import numpy as np
import torch.nn as nn

from sklearn.base import BaseEstimator


class TreeFlowBoost(BaseEstimator):

    def __init__(self, tree_model, flow_model, embedding_size: int = 20):
        self.tree_model = tree_model
        self.flow_model = flow_model

        self.embedding_size = embedding_size

    def fit(self, X: np.ndarray, y: np.ndarray, n_epochs=100):
        self.tree_model.fit(X, y)

        context: np.ndarray = self.tree_model.embed(X)
        self.flow_model.setup_context_encoder(nn.Sequential(
            nn.Linear(context.shape[1], self.embedding_size),
            nn.Tanh(),
        ))

        params: np.ndarray = self.tree_model.pred_dist_param(X)
        y: np.ndarray = y if len(y.shape) == 2 else y.reshape(-1, 1)

        self.flow_model.fit(y, context, params, n_epochs=n_epochs)
        return self

    def sample(self, X: np.ndarray, num_samples: int = 10) -> np.ndarray:
        context: np.ndarray = self.tree_model.embed(X)
        params: np.ndarray = self.tree_model.pred_dist_param(X)
        samples: np.ndarray = self.flow_model.sample(num_samples=num_samples, context=context, params=params)
        return samples

    def predict(self, X: np.ndarray, num_samples: int = 10) -> np.ndarray:
        samples: np.ndarray = self.sample(X=X, num_samples=num_samples)
        y_hat: np.ndarray = samples.mean(axis=1)
        return y_hat
