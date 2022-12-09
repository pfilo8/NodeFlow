import uuid

from tqdm import tqdm
from typing import Iterable, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.base import BaseEstimator, RegressorMixin
from torch.utils.data import DataLoader, TensorDataset

from .flow.flow2 import ContinuousNormalizingFlow
from .soft_decision_tree import SoftDecisionTree


class SoftTreeFlow(BaseEstimator, RegressorMixin, nn.Module):

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
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
        :param tree_depth: Soft Decision Tree depth parameter.
        :param tree_lambda: Soft Decision Tree lambda parameter.
        :param flow_hidden_dims: Continuous Normalizing Flow hidden dimensionality.
        :param flow_num_blocks: Continuous Normalizing Flow number of blocks of flow.
        :param flow_layer_type: Continuous Normalizing Flow layer type.
        :param flow_nonlinearity : Continuous Normalizing Flow nonlinearity function.
        """
        nn.Module.__init__(self)

        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.input_dim = input_dim
        self.output_dim = output_dim
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

        self.flow_model = ContinuousNormalizingFlow(
            input_dim=output_dim,
            hidden_dims=flow_hidden_dims,
            context_dim=2 ** tree_depth,
            num_blocks=flow_num_blocks,
            conditional=True,  # It must be true as we are using Conditional CNF model for SoftTreeFlow.
            layer_type=flow_layer_type,
            nonlinearity=flow_nonlinearity,
            device=self.device
        )

    def _log_prob(self, X: torch.Tensor, y: torch.Tensor, return_penalty=False):
        """ Calculate the log probability of the model (batch). Internal method used for training."""
        x, penalty = self.tree_model.forward_leaves(X)
        x = self.flow_model.log_prob(y, x)

        if return_penalty:
            return x, penalty
        return x

    @torch.no_grad()
    def log_prob(self, X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor],
                 batch_size: int = 128) -> np.ndarray:
        """ Calculate the log probability of the model."""
        X: torch.Tensor = torch.as_tensor(data=X, dtype=torch.float, device=self.device)
        y: torch.Tensor = torch.as_tensor(data=y, dtype=torch.float, device=self.device)

        dataset_loader: DataLoader = DataLoader(
            dataset=TensorDataset(X, y),
            shuffle=False,
            batch_size=batch_size
        )

        logpxs: List[torch.Tensor] = [self._log_prob(X=x_batch, y=y_batch) for x_batch, y_batch in dataset_loader]
        logpx: torch.Tensor = torch.cat(logpxs, dim=0)
        logpx: torch.Tensor = logpx.detach().cpu()
        logpx: np.ndarray = logpx.numpy()
        return logpx

    @torch.no_grad()
    def _sample(self, X: torch.Tensor, num_samples: int) -> torch.Tensor:
        x, penalty = self.tree_model.forward_leaves(X)
        x = self.flow_model.sample(x, num_samples=num_samples)
        return x

    @torch.no_grad()
    def sample(self, X: Union[np.ndarray, torch.Tensor], num_samples: int = 10, batch_size: int = 128) -> np.ndarray:
        """Sample from the model."""
        X: torch.Tensor = torch.as_tensor(data=X, dtype=torch.float, device=self.device)
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
        samples: torch.Tensor = samples.detach().cpu().squeeze()
        samples: np.ndarray = samples.numpy()
        return samples

    def fit(self, X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor],
            X_val: Union[np.ndarray, torch.Tensor, None] = None, y_val: Union[np.ndarray, torch.Tensor, None] = None,
            n_epochs: int = 100, batch_size: int = 128, max_patience: int = 50, verbose: bool = False):
        """ Fit SoftTreeFlow model.

        Method supports the best epoch model selection and early stopping (max_patience param)
        if validation dataset is available.
        """
        X: torch.Tensor = torch.as_tensor(data=X, dtype=torch.float, device=self.device)
        y: torch.Tensor = torch.as_tensor(data=y, dtype=torch.float, device=self.device)

        if X_val is not None and y_val is not None:
            X_val: torch.Tensor = torch.as_tensor(data=X_val, dtype=torch.float, device=self.device)
            y_val: torch.Tensor = torch.as_tensor(data=y_val, dtype=torch.float, device=self.device)

        self.optimizer_ = optim.Adam(self.parameters())

        dataset_loader_train: DataLoader = DataLoader(
            dataset=TensorDataset(X, y),
            shuffle=True,
            batch_size=batch_size
        )

        patience: int = 0
        mid: str = uuid.uuid4()  # To be able to run multiple experiments in parallel.
        loss_best: float = np.inf

        for _ in tqdm(range(n_epochs)):
            self.train()
            for x_batch, y_batch in dataset_loader_train:
                self.optimizer_.zero_grad()

                logpx, penalty = self._log_prob(x_batch, y_batch, return_penalty=True)
                loss = -logpx.mean() + penalty

                loss.backward()
                self.optimizer_.step()

            self.eval()
            if X_val is not None and y_val is not None:
                loss_val: float = self.nll(X_val, y_val)

                # Save model if better
                if loss_val < loss_best:
                    loss_best = loss_val
                    self._save_temp(mid)
                    patience = 0

                else:
                    patience += 1

                if patience > max_patience:
                    break

        if X_val is not None and y_val is not None:
            return self._load_temp(mid)
        return self

    @torch.no_grad()
    def predict(self, X: Union[np.ndarray, torch.Tensor], method: str = 'mean', num_samples: int = 1000,
                batch_size: int = 128, **kwargs) -> np.ndarray:
        X: torch.Tensor = torch.as_tensor(data=X, dtype=torch.float, device=self.device)
        samples: np.ndarray = self.sample(X=X, num_samples=num_samples, batch_size=batch_size)

        if method == 'mean':
            y_pred: np.ndarray = samples.mean(axis=1)
        else:
            raise ValueError(f'Method {method} not supported.')

        y_pred: np.ndarray = np.array(y_pred)
        return y_pred

    @torch.no_grad()
    def nll(self, X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]) -> float:
        return - self.log_prob(X, y).mean()

    @torch.no_grad()
    def crps(self, X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]) -> float:
        return None

    def predict_tree_path(self, X: np.ndarray):
        """ Method for predicting the tree path from Soft Decision Tree component."""
        X: torch.Tensor = torch.as_tensor(data=X, dtype=torch.float, device=self.device)
        paths, _ = self.tree_model.forward_leaves(X)
        paths: np.ndarray = paths.detach().cpu().numpy()
        return paths

    def _save_temp(self, mid: str):
        torch.save(self, f"/tmp/model_{mid}.pt")

    def _load_temp(self, mid: str):
        return torch.load(f"/tmp/model_{mid}.pt")
