from typing import Any, Callable, Iterable, List, Union, Optional
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer
from torch.utils.data import DataLoader, TensorDataset

from probabilistic_flow_boosting.models.flow import ContinuousNormalizingFlow
from probabilistic_flow_boosting.models.node import DenseODSTBlock
from probabilistic_flow_boosting.models.node.activations import sparsemax, sparsemoid


class NodeFlowDataModule(pl.LightningDataModule):
    def __init__(
            self,
            X_train: pd.DataFrame,
            y_train: pd.DataFrame,
            X_test: Optional[pd.DataFrame] = None,
            y_test: Optional[pd.DataFrame] = None,
            split_size: float = 0.8,
            batch_size: int = 1024
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.split_size = split_size

        self.X_train = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else X_train
        self.y_train = y_train.to_numpy() if isinstance(y_train, pd.DataFrame) else y_train

        if X_test is not None:
            self.X_test = X_test.to_numpy() if isinstance(X_test, pd.DataFrame) else X_test
            self.y_test = y_test.to_numpy() if isinstance(y_test, pd.DataFrame) else y_test

        if self.split_size is not None:
            num_training_examples = int(self.split_size * self.X_train.shape[0])
            self.x_tr, self.x_val = self.X_train[:num_training_examples], self.X_train[num_training_examples:]
            self.y_tr, self.y_val = self.y_train[:num_training_examples], self.y_train[num_training_examples:]
        else:
            self.x_tr = self.X_train
            self.y_tr = self.y_train

        self.feature_scaler = Pipeline(
            [
                ("quantile", QuantileTransformer(output_distribution="normal")),
                ("standarize", StandardScaler()),
            ]
        )
        self.target_scaler = MinMaxScaler(feature_range=(-1, 1))

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.x_tr: np.ndarray = self.feature_scaler.fit_transform(self.x_tr)
            self.y_tr: np.ndarray = self.target_scaler.fit_transform(self.y_tr)
            if self.split_size is not None:
                self.x_val: np.ndarray = self.feature_scaler.transform(self.x_val)
                self.y_val: np.ndarray = self.target_scaler.transform(self.y_val)
        if stage == "validate":
            self.x_val: np.ndarray = self.feature_scaler.transform(self.x_val)
            self.y_val: np.ndarray = self.target_scaler.transform(self.y_val)
        if stage == "test":
            self.x_tr: np.ndarray = self.feature_scaler.fit_transform(self.x_tr)
            self.y_tr: np.ndarray = self.target_scaler.fit_transform(self.y_tr)
            self.X_test = self.feature_scaler.transform(self.X_test)
            self.y_test = self.target_scaler.transform(self.y_test)

    def _to_dataloader(self, X, y):
        X: torch.Tensor = torch.as_tensor(X, dtype=torch.float32)
        y: torch.Tensor = torch.as_tensor(y, dtype=torch.float32)
        return DataLoader(
            dataset=TensorDataset(X, y),
            shuffle=False,
            batch_size=self.batch_size,
        )

    def train_dataloader(self):
        return self._to_dataloader(X=self.x_tr, y=self.y_tr)

    def val_dataloader(self):
        return self._to_dataloader(X=self.x_val, y=self.y_val)

    def test_dataloader(self):
        return self._to_dataloader(X=self.X_test, y=self.y_test)

    def predict_dataloader(self):
        return self._to_dataloader(X=self.X_test, y=self.y_test)


class NodeFlow(pl.LightningModule):
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
            random_state: int = 0,
    ):
        super().__init__()
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
        )

        self.flow_model = ContinuousNormalizingFlow(
            input_dim=output_dim,
            hidden_dims=flow_hidden_dims,
            context_dim=num_layers * tree_output_dim * num_trees,  # + input_dim,
            num_blocks=flow_num_blocks,
            conditional=True,  # It must be true as we are using Conditional CNF model.
            layer_type=flow_layer_type,
            nonlinearity=flow_nonlinearity,
        )

    @torch.enable_grad()
    def forward(self, X, y):
        """Calculate the log probability of the model (batch). Method used only for training and validation."""
        x = self.tree_model(X)
        x = self.flow_model.log_prob(y, x)
        x += np.log(np.abs(np.prod(
            self.trainer.datamodule.target_scaler.scale_)))  # Target scaling correction. log(abs(det(jacobian)))
        return x

    @torch.enable_grad()
    def _log_prob(self, X, y):
        """Calculate the log probability of the model (batch). Method used only for testing."""
        grad_x = X.clone().requires_grad_()
        x = self.tree_model(grad_x)
        x = self.flow_model.log_prob(y, x)
        x += np.log(np.abs(np.prod(
            self.trainer.datamodule.target_scaler.scale_)))  # Target scaling correction. log(abs(det(jacobian)))
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logpx = self(x, y)
        loss = -logpx.mean()
        self.log("train_nll", loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logpx = self(x, y)
        loss = -logpx.mean()
        self.log("val_nll", loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logpx = self._log_prob(x, y)
        loss = -logpx.mean()
        self.log("test_nll", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    @torch.enable_grad()
    def _sample(self, X: torch.Tensor, num_samples: int) -> torch.Tensor:
        grad_x = X.clone().requires_grad_()
        x = self.tree_model(grad_x)
        x = self.flow_model.sample(x, num_samples=num_samples)
        return x

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0, num_samples: int = 1000) -> Any:
        X, y = batch
        samples = self._sample(X, num_samples)

        # Inverse target transformation
        # samples: torch.Tensor = torch.cat(all_samples, dim=0)
        samples_size = samples.shape
        samples: np.ndarray = samples.detach().cpu().numpy()
        samples: np.ndarray = samples.reshape((samples_size[0] * samples_size[1], samples_size[2]))
        samples: np.ndarray = self.trainer.datamodule.target_scaler.inverse_transform(samples)
        samples: np.ndarray = samples.reshape((samples_size[0], samples_size[1], samples_size[2]))
        samples: np.ndarray = samples.squeeze()
        return samples

    def configure_optimizers(self) -> Any:
        optimizer = optim.RAdam(self.parameters(), lr=0.003)
        return optimizer

    def save(self, filename: str):
        torch.save(self, f"{filename}-nodeflow.pt")

    @classmethod
    def load(cls, filename: str, map_location=None):
        return torch.load(f"{filename}-nodeflow.pt", map_location=map_location)
