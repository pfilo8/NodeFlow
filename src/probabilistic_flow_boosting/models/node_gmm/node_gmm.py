from typing import Any, Callable, Optional, Union, Iterable, List
from lightning.pytorch.utilities.types import STEP_OUTPUT

import numpy as np

import torch
from torch import nn
from torch.distributions import Categorical, MixtureSameFamily, Normal
import torch.optim as optim
import lightning.pytorch as pl

# from sklearn.mixture import GaussianMixture
from probabilistic_flow_boosting.models.gmm import GaussianMixture
from probabilistic_flow_boosting.models.node import DenseODSTBlock
from probabilistic_flow_boosting.models.node.activations import sparsemax, sparsemoid


class NodeGMM(pl.LightningModule):
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
        n_components: int = 2,
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
        self.n_components = n_components
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
        self.gauss_model = GaussianMixture(
            n_components=n_components,
            context_dim=num_layers * tree_output_dim * num_trees,
        ).to("cuda")
        

    def forward(self, X, y):
        """Calculate the log probability of the model (batch). Method used only for training and validation."""
        x = self.tree_model(X)
        logpx = self.gauss_model.log_prob(x, y)
        logpx += np.log(np.abs(np.prod(self.trainer.datamodule.target_scaler.scale_))) # Target scaling correction. log(abs(det(jacobian)))
        return logpx
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logpx = self(x, y)
        nll = -logpx.mean()
        self.log("train_nll", nll, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        return nll
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logpx = self(x, y)
        nll = -logpx.mean()
        self.log("val_nll", nll, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        return nll
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logpx = self(x, y)
        nll = -logpx.mean()
        self.log("test_nll", nll, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return nll
    
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0, num_samples: int = 1000) -> Any:
        X, y = batch
        x = self.tree_model(X)
        samples = self.gauss_model.sample(x, num_samples=num_samples)

        samples_size = samples.shape
        samples: np.ndarray = samples.detach().cpu().numpy()
        samples: np.ndarray = samples.reshape((samples_size[0] * samples_size[1], samples_size[2]))
        samples: np.ndarray = self.trainer.datamodule.target_scaler.inverse_transform(samples)
        samples: np.ndarray = samples.reshape((samples_size[0], samples_size[1], samples_size[2]))
        samples: np.ndarray = samples.squeeze()
        return samples

    def configure_optimizers(self):
        return optim.RAdam(self.parameters(), lr=1e-3)
    
    def save(self, filename: str):
        torch.save(self, f"{filename}-nodegmm.pt")

    @classmethod
    def load(cls, filename: str):
        return torch.load(f"{filename}-nodegmm.pt")