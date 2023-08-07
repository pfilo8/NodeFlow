from typing import Any, Callable, Iterable, List, Union, Optional

import numpy as np
import pandas as pd
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
    def __init__(self, X: pd.DataFrame, y: pd.DataFrame, split_size: Optional[float], batch_size: int = 1024) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.split_size = split_size
        self.X = X.to_numpy()
        self.y = y.to_numpy()
        if self.split_size:
            num_training_examples = int(self.split_size * self.X.shape[0])
            self.x_tr, self.x_val = self.X[:num_training_examples], self.X[num_training_examples:]
            self.y_tr, self.y_val = self.y[:num_training_examples], self.y[num_training_examples:]
        
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
            self.x_val: np.ndarray = self.feature_scaler.transform(self.x_val)
            self.y_val: np.ndarray = self.target_scaler.transform(self.y_val)
        if stage == "validate":
            self.x_val: np.ndarray = self.feature_scaler.transform(self.x_val)
            self.y_val: np.ndarray = self.target_scaler.transform(self.y_val)
        if stage == "test":
            self.X = self.feature_scaler.transform(self.X)
            self.y = self.target_scaler.transform(self.y)

    def train_dataloader(self):
        X: torch.Tensor = torch.as_tensor(self.x_tr, dtype=torch.float32)
        y: torch.Tensor = torch.as_tensor(self.y_tr, dtype=torch.float32)
        return DataLoader(
            dataset=TensorDataset(X, y),
            shuffle=False,
            batch_size=self.batch_size
        )

    def val_dataloader(self):
        X: torch.Tensor = torch.as_tensor(self.x_val, dtype=torch.float32)
        y: torch.Tensor = torch.as_tensor(self.y_val, dtype=torch.float32)
        return DataLoader(
            dataset=TensorDataset(X, y),
            shuffle=False,
            batch_size=self.batch_size
        )
        


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
            context_dim=num_layers * tree_output_dim * num_trees,
            num_blocks=flow_num_blocks,
            conditional=True,  # It must be true as we are using Conditional CNF model.
            layer_type=flow_layer_type,
            nonlinearity=flow_nonlinearity,
        )

    def forward(self, X, y):
        """Calculate the log probability of the model (batch). Method used only for training and validation."""
        x = self.tree_model(X)
        x = self.flow_model.log_prob(y, x)
        x += np.log(np.abs(np.prod(self.trainer.datamodule.target_scaler.scale_))) # Target scaling correction. log(abs(det(jacobian)))
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logpx = self(x, y)
        loss = -logpx.mean()
        self.log("train_nll", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logpx = self(x, y)
        loss = -logpx.mean()
        self.log("val_nll", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self) -> Any:
        optimizer = optim.RAdam(self.parameters(), lr=0.003)
        return optimizer

    # @torch.no_grad()
    # def _sample(self, X: torch.Tensor, num_samples: int) -> torch.Tensor:
    #     x = self.tree_model(X)
    #     x = self.flow_model.sample(x, num_samples=num_samples)
    #     return x

    # @torch.no_grad()
    # def sample(self, X: np.ndarray, num_samples: int = 10, batch_size: int = 128) -> np.ndarray:
    #     """Sample from the model."""
    #     X: np.ndarray = self.feature_scaler.transform(X)

    #     X: torch.Tensor = torch.as_tensor(data=X, dtype=torch.float, device=self.device)
    #     dataset_loader: DataLoader = DataLoader(dataset=TensorDataset(X), shuffle=False, batch_size=batch_size)

    #     all_samples: List[torch.Tensor] = []

    #     for x in tqdm(dataset_loader):
    #         sample: torch.Tensor = self._sample(x[0], num_samples)
    #         all_samples.append(sample)

    #     samples: torch.Tensor = torch.cat(all_samples, dim=0)
    #     samples: torch.Tensor = samples.detach().cpu()
    #     samples: np.ndarray = samples.numpy()

    #     # Inverse target transformation
    #     samples_size = samples.shape

    #     samples: np.ndarray = samples.reshape((samples_size[0] * samples_size[1], samples_size[2]))
    #     samples: np.ndarray = self.target_scaler.inverse_transform(samples)
    #     samples: np.ndarray = samples.reshape((samples_size[0], samples_size[1], samples_size[2]))

    #     samples: np.ndarray = samples.squeeze()
    #     return samples

    # @torch.no_grad()
    # def predict(
    #     self, X: np.ndarray, method: str = "mean", num_samples: int = 1000, batch_size: int = 128, **kwargs
    # ) -> np.ndarray:
    #     samples: np.ndarray = self.sample(X=X, num_samples=num_samples, batch_size=batch_size)

    #     if method == "mean":
    #         y_pred: np.ndarray = samples.mean(axis=1)
    #     else:
    #         raise ValueError(f"Method {method} not supported.")

    #     y_pred: np.ndarray = np.array(y_pred)
    #     return y_pred

    # def predict_tree_path(self, X: np.ndarray):
    #     """Method for predicting the tree path from Soft Decision Tree component."""
    #     X: torch.Tensor = torch.as_tensor(data=X, dtype=torch.float, device=self.device)
    #     paths, _ = self.tree_model(X)
    #     paths: np.ndarray = paths.detach().cpu().numpy()
    #     return paths

    # def _save_temp(self, mid: str):
    #     torch.save(self, f"/tmp/model_{mid}.pt")

    # def _load_temp(self, mid: str):
    #     return torch.load(f"/tmp/model_{mid}.pt")

    # def save(self, filename: str):
    #     torch.save(self, f"{filename}-nodeflow.pt")

    # @classmethod
    # def load(cls, filename: str):
    #     return torch.load(f"{filename}-nodeflow.pt")
