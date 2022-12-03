from typing import Iterable, List, Union

import numpy as np
import torch
import torch.optim as optim

from sklearn.base import BaseEstimator
from torch.utils.data import DataLoader, TensorDataset

from .flow.flow2 import ContinuousNormalizingFlow
from .shallow_feature_extractor import ShallowFeatureExtractor
from .soft_decision_tree import SoftDecisionTree


class SoftTreeFlow(BaseEstimator):

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            sfe_context_dim: int = 40,
            tree_depth: int = 5,
            tree_lambda: float = 1e-3,
            flow_hidden_dims: Iterable[int] = (80, 40),
            flow_num_blocks: int = 3,
            flow_layer_type: str = "concatsquash",
            flow_nonlinearity: str = "tanh",
            device: str = None
    ):
        """
        Initialization of SoftTreeFlow model.

        :param input_dim: X data dimensionality.
        :param output_dim: Y data dimensionality.
        :param sfe_context_dim: Shallow Feature Extractor dimensionality.
        :param tree_depth: Soft Decision Tree depth parameter.
        :param tree_lambda: Soft Decision Tree lambda parameter.
        :param flow_hidden_dims: Continuous Normalizing Flow hidden dimensionality.
        :param flow_num_blocks: Continuous Normalizing Flow number of blocks of flow.
        :param flow_layer_type: Continuous Normalizing Flow layer type.
        :param flow_nonlinearity : Continuous Normalizing Flow nonlinearity function.
        """
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sfe_context_dim = sfe_context_dim
        self.tree_depth = tree_depth
        self.tree_lambda = tree_lambda
        self.flow_hidden_dims = flow_hidden_dims
        self.flow_num_blocks = flow_num_blocks
        self.flow_layer_type = flow_layer_type
        self.flow_nonlinearity = flow_nonlinearity

        self.tree_model = SoftDecisionTree(
            input_dim=input_dim,
            output_dim=output_dim,
            depth=tree_depth,
            lamda=tree_lambda,
            device=self.device
        )

        self.shallow_feature_extractor = ShallowFeatureExtractor(
            input_dim=2 ** tree_depth,
            output_dim=sfe_context_dim,
            device=self.device
        )

        self.flow_model = ContinuousNormalizingFlow(
            input_dim=output_dim,
            hidden_dims=flow_hidden_dims,
            context_dim=sfe_context_dim,
            num_blocks=flow_num_blocks,
            conditional=True,  # It must be true as we are using Conditional CNF model for SoftTreeFlow.
            layer_type=flow_layer_type,
            nonlinearity=flow_nonlinearity,
            device=self.device
        )

    def _log_prob(self, X: torch.Tensor, y: torch.Tensor, return_penalty=False):
        """ Calculate the log probability of the model (batch)."""
        x, penalty = self.tree_model.forward_leaves(X)
        x = self.shallow_feature_extractor(x)
        x = self.flow_model.log_prob(y, x)

        if return_penalty:
            return x, penalty
        return x

    @torch.no_grad()
    def log_prob(self, X: torch.Tensor, y: torch.Tensor, batch_size: int = 128):
        """ Calculate the log probability of the model."""
        dataset_loader: DataLoader = DataLoader(
            dataset=TensorDataset(X, y),
            shuffle=False,
            batch_size=batch_size
        )

        logpxs: List[torch.Tensor] = [self._log_prob(X=x_batch, y=y_batch) for x_batch, y_batch in dataset_loader]
        logpx: torch.Tensor = torch.cat(logpxs, dim=0)
        logpx: np.ndarray = logpx.detach().cpu().numpy()
        return logpx

    @torch.no_grad()
    def _sample(self, X: torch.Tensor, num_samples: int) -> torch.Tensor:
        x, penalty = self.tree_model.forward_leaves(X)
        x = self.shallow_feature_extractor(x)
        x = self.flow_model.sample(x, num_samples=num_samples)
        return x

    @torch.no_grad()
    def sample(self, X: torch.Tensor, num_samples: int = 10, batch_size: int = 128) -> torch.Tensor:
        dataset_loader: DataLoader = DataLoader(
            dataset=TensorDataset(X),
            shuffle=False,
            batch_size=batch_size
        )

        all_samples: List[torch.Tensor] = []

        for x in dataset_loader:
            sample: torch.Tensor = self._sample(x[0], num_samples)
            all_samples.append(sample)

        samples: torch.Tensor = torch.cat(all_samples, dim=0)
        samples: torch.Tensor = samples.detach().cpu()
        return samples

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            X_val: Union[np.ndarray, None] = None,
            y_val: Union[np.ndarray, None] = None,
            n_epochs: int = 100,
            batch_size: int = 128,
            verbose: bool = False
    ):
        self.optimizer = optim.Adam([
            *self.tree_model.parameters(),
            *self.shallow_feature_extractor.parameters(),
            *self.flow_model.parameters()
        ])

        X: torch.Tensor = torch.as_tensor(data=X, dtype=torch.float, device=self.device)
        y: torch.Tensor = torch.as_tensor(data=y, dtype=torch.float, device=self.device)

        if X_val is not None and y_val is not None:
            X_val: torch.Tensor = torch.as_tensor(data=X_val, dtype=torch.float, device=self.device)
            y_val: torch.Tensor = torch.as_tensor(data=y_val, dtype=torch.float, device=self.device)

        dataset_loader_train: DataLoader = DataLoader(
            dataset=TensorDataset(X, y),
            shuffle=True,
            batch_size=batch_size
        )

        loss_best = np.inf

        for i in range(n_epochs):
            print(i)
            for x_batch, y_batch in dataset_loader_train:
                self.optimizer.zero_grad()

                logpx, penalty = self._log_prob(x_batch, y_batch, return_penalty=True)
                loss = -logpx.mean() + penalty
                print(f'Loss {loss}')

                loss.backward()
                self.optimizer.step()

            if X_val is not None and y_val is not None:
                loss_val = -self.log_prob(X_val, y_val, batch_size=batch_size).mean().item()
                print(f'Loss validation {loss_val}')

                # Save model if better
                if loss_val < loss_best:
                    self.epoch_best = i
                    loss_best = loss_val

        return self
