import uuid

from tqdm import tqdm
from typing import Iterable, List, Union, Optional, Any

import numpy as np
import pandas as pd
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset

from probabilistic_flow_boosting.models.flow import ContinuousNormalizingFlow

class CNFDataModule(pl.LightningDataModule):
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
        
        self.feature_scaler = MinMaxScaler(feature_range=(-1, 1))
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


class ContinuousNormalizingFlowRegressor(pl.LightningModule):

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            embedding_dim: int = 40,
            hidden_dims: Iterable[int] = (80, 40),
            num_blocks: int = 3,
            layer_type: str = "concatsquash",
            nonlinearity: str = "tanh",
            device: str = None
    ):
        """
        Initialization of Continuous Normalizing Flow model.
        """
        # nn.Module.__init__(self)

        # if device:
        #     self.device = device
        # elif torch.cuda.is_available():
        #     self.device = torch.device('cuda')
        # else:
        #     self.device = torch.device('cpu')
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.num_blocks = num_blocks
        self.layer_type = layer_type
        self.nonlinearity = nonlinearity

        # self.feature_scaler = MinMaxScaler(feature_range=(-1, 1))
        # self.target_scaler = MinMaxScaler(feature_range=(-1, 1))

        if embedding_dim > 0:
            self.feature_extractor = nn.Sequential(
                nn.Linear(input_dim, embedding_dim),
                nn.Tanh(),
            )#.to(self.device)

            self.flow_model = ContinuousNormalizingFlow(
                input_dim=output_dim,
                hidden_dims=hidden_dims,
                context_dim=embedding_dim,
                num_blocks=num_blocks,
                conditional=True,  # It must be true as we are using Conditional CNF model.
                layer_type=layer_type,
                nonlinearity=nonlinearity,
                device=self.device
            )
        elif embedding_dim == 0:
            self.feature_extractor = nn.Identity()#.to(self.device)

            self.flow_model = ContinuousNormalizingFlow(
                input_dim=output_dim,
                hidden_dims=hidden_dims,
                context_dim=input_dim,
                num_blocks=num_blocks,
                conditional=True,  # It must be true as we are using Conditional CNF model.
                layer_type=layer_type,
                nonlinearity=nonlinearity,
                # device=self.device
            )
        else:
            ValueError(f"Embedding dim must be greater or equal to zero. Provided value: f{embedding_dim}.")
    
    @torch.enable_grad() 
    def forward(self, X, y):
        """Calculate the log probability of the model (batch). Method used only for training and validation."""
        x = self.feature_extractor(X)
        x = self.flow_model.log_prob(y, x)
        x += np.log(np.abs(np.prod(self.trainer.datamodule.target_scaler.scale_)))  # Target scaling correction. log(abs(det(jacobian)))
        return x
    
    @torch.enable_grad()
    def _log_prob(self, X, y):
        """Calculate the log probability of the model (batch). Method used only for testing."""
        grad_x = X.clone().requires_grad_()
        x = self.feature_extractor(grad_x)
        x = self.flow_model.log_prob(y, x)
        x += np.log(np.abs(np.prod(self.trainer.datamodule.target_scaler.scale_)))  # Target scaling correction. log(abs(det(jacobian)))
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
        x = self.feature_extractor(grad_x)
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
        torch.save(self, f"{filename}-cnf.pt")

    @classmethod
    def load(cls, filename: str):
        return torch.load(f"{filename}-cnf.pt")
