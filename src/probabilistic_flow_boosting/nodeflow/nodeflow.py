import uuid

from tqdm import tqdm
from typing import Callable, Iterable, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer
from torch.utils.data import DataLoader, TensorDataset

from .flow import ContinuousNormalizingFlow
from .node import DenseODSTBlock
from .node.activations import sparsemax, sparsemoid


class NodeFlow(BaseEstimator, RegressorMixin, nn.Module):

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            num_trees: int = 200,
            depth: int = 6,
            tree_output_dim: int = 1,
            choice_function: Callable = sparsemax,
            bin_function: Callable = sparsemoid,
            initialize_response_: Callable = nn.init.normal_,
            initialize_selection_logits_: Callable = nn.init.uniform_,
            threshold_init_beta: float = 1.0,
            threshold_init_cutoff: float = 1.0,
            num_layers: int = 6,
            max_features: Union[None, int] = None,
            input_dropout: float = 0.0,
            flow_hidden_dims: Iterable[int] = (80, 40),
            flow_num_blocks: int = 3,
            flow_layer_type: str = "concatsquash",
            flow_nonlinearity: str = "tanh",
            device: str = None,
            random_state: int = 0
    ):
        nn.Module.__init__(self)

        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_trees = num_trees
        self.depth = depth
        self.tree_output_dim = tree_output_dim
        self.choice_function = choice_function
        self.bin_function = bin_function
        self.initialize_response_ = initialize_response_
        self.initialize_selection_logits_ = initialize_selection_logits_
        self.threshold_init_beta = threshold_init_beta
        self.threshold_init_cutoff = threshold_init_cutoff
        self.num_layers = num_layers
        self.max_features = max_features
        self.input_dropout = input_dropout
        self.flow_hidden_dims = flow_hidden_dims
        self.flow_num_blocks = flow_num_blocks
        self.flow_layer_type = flow_layer_type
        self.flow_nonlinearity = flow_nonlinearity
        self.random_state = random_state
        self._best_epoch = None

        self.feature_scaler = Pipeline([
            ('quantile', QuantileTransformer(random_state=random_state, output_distribution='normal')),
            ('standarize', StandardScaler())
        ])
        self.target_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.tree_model = DenseODSTBlock(
            input_dim,
            num_trees,
            depth=depth,
            tree_output_dim=tree_output_dim,
            choice_function=choice_function,
            bin_function=bin_function,
            initialize_response_=initialize_response_,
            initialize_selection_logits_=initialize_selection_logits_,
            threshold_init_beta=threshold_init_beta,
            threshold_init_cutoff=threshold_init_cutoff,
            num_layers=num_layers,
            max_features=max_features,
            input_dropout=input_dropout,
            flatten_output=True,
        ).to(device)

        self.flow_model = ContinuousNormalizingFlow(
            input_dim=output_dim,
            hidden_dims=flow_hidden_dims,
            context_dim=num_layers * tree_output_dim * num_trees,
            num_blocks=flow_num_blocks,
            conditional=True,  # It must be true as we are using Conditional CNF model.
            layer_type=flow_layer_type,
            nonlinearity=flow_nonlinearity,
            device=self.device
        )

    def _log_prob(self, X: torch.Tensor, y: torch.Tensor):
        """ Calculate the log probability of the model (batch). Internal method used for training."""
        x = self.tree_model(X)
        x = self.flow_model.log_prob(y, x)
        x += np.log(np.abs(np.prod(self.target_scaler.scale_)))  # Target scaling correction. log(abs(det(jacobian)))
        return x

    @torch.no_grad()
    def log_prob(self, X: np.ndarray, y: np.ndarray, batch_size: int = 128) -> np.ndarray:
        """ Calculate the log probability of the model."""
        X: np.ndarray = self.feature_scaler.transform(X)
        y: np.ndarray = self.target_scaler.transform(y)

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
        x = self.tree_model(X)
        x = self.flow_model.sample(x, num_samples=num_samples)
        return x

    @torch.no_grad()
    def sample(self, X: np.ndarray, num_samples: int = 10, batch_size: int = 128) -> np.ndarray:
        """Sample from the model."""
        X: np.ndarray = self.feature_scaler.transform(X)

        X: torch.Tensor = torch.as_tensor(data=X, dtype=torch.float, device=self.device)
        dataset_loader: DataLoader = DataLoader(
            dataset=TensorDataset(X),
            shuffle=False,
            batch_size=batch_size
        )

        all_samples: List[torch.Tensor] = []

        for x in tqdm(dataset_loader):
            sample: torch.Tensor = self._sample(x[0], num_samples)
            all_samples.append(sample)

        samples: torch.Tensor = torch.cat(all_samples, dim=0)
        samples: torch.Tensor = samples.detach().cpu()
        samples: np.ndarray = samples.numpy()

        # Inverse target transformation
        samples_size = samples.shape

        samples: np.ndarray = samples.reshape((samples_size[0] * samples_size[1], samples_size[2]))
        samples: np.ndarray = self.target_scaler.inverse_transform(samples)
        samples: np.ndarray = samples.reshape((samples_size[0], samples_size[1], samples_size[2]))

        samples: np.ndarray = samples.squeeze()
        return samples

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Union[np.ndarray, None] = None,
            y_val: Union[np.ndarray, None] = None, n_epochs: int = 100, batch_size: int = 128, max_patience: int = 50,
            verbose: bool = False):
        """ Fit SoftTreeFlow model.

        Method supports the best epoch model selection and early stopping (max_patience param)
        if validation dataset is available.
        """
        X: np.ndarray = self.feature_scaler.fit_transform(X)
        y: np.ndarray = self.target_scaler.fit_transform(y)

        X: torch.Tensor = torch.as_tensor(data=X, dtype=torch.float, device=self.device)
        y: torch.Tensor = torch.as_tensor(data=y, dtype=torch.float, device=self.device)

        self.optimizer_ = optim.RAdam(self.parameters(), lr=0.003)

        dataset_loader_train: DataLoader = DataLoader(
            dataset=TensorDataset(X, y),
            shuffle=True,
            batch_size=batch_size
        )

        patience: int = 0
        mid: str = str(uuid.uuid4())  # To be able to run multiple experiments in parallel.
        self.mid = mid
        loss_best: float = np.inf

        with tqdm(range(n_epochs), disable=(not verbose)) as pbar:
            for epoch in pbar:
                self.train()
                for x_batch, y_batch in dataset_loader_train:
                    self.optimizer_.zero_grad()

                    logpx = self._log_prob(x_batch, y_batch)
                    loss = -logpx.mean()

                    loss.backward()
                    self.optimizer_.step()

                self.eval()
                if X_val is not None and y_val is not None:
                    loss_val: float = self.nll(X_val, y_val)
                    pbar.set_description(f"Validation loss: {round(loss_val, 4)}.")

                    # Save model if better
                    if loss_val < loss_best:
                        loss_best = loss_val
                        self._save_temp(mid)
                        patience = 0
                        self._best_epoch = epoch

                    else:
                        patience += 1

                    if patience > max_patience:
                        break

        if X_val is not None and y_val is not None:
            return self._load_temp(mid)
        return self

    @torch.no_grad()
    def predict(self, X: np.ndarray, method: str = 'mean', num_samples: int = 1000, batch_size: int = 128,
                **kwargs) -> np.ndarray:
        samples: np.ndarray = self.sample(X=X, num_samples=num_samples, batch_size=batch_size)

        if method == 'mean':
            y_pred: np.ndarray = samples.mean(axis=1)
        else:
            raise ValueError(f'Method {method} not supported.')

        y_pred: np.ndarray = np.array(y_pred)
        return y_pred

    @torch.no_grad()
    def nll(self, X: np.ndarray, y: np.ndarray) -> float:
        return - self.log_prob(X, y).mean()

    @torch.no_grad()
    def crps(self, X: np.ndarray, y: np.ndarray, n_samples: int = 1000) -> float:
        return None

    def predict_tree_path(self, X: np.ndarray):
        """ Method for predicting the tree path from Soft Decision Tree component."""
        X: torch.Tensor = torch.as_tensor(data=X, dtype=torch.float, device=self.device)
        paths, _ = self.tree_model(X)
        paths: np.ndarray = paths.detach().cpu().numpy()
        return paths

    def _save_temp(self, mid: str):
        torch.save(self, f"/tmp/model_{mid}.pt")

    def _load_temp(self, mid: str):
        return torch.load(f"/tmp/model_{mid}.pt")
    
    def save(self, filename: str):
        torch.save(self, f"{filename}-nodeflow.pt")

    @classmethod
    def load(cls, filename: str):
        return torch.load(f"{filename}-nodeflow.pt")