from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from nflows.distributions import ConditionalDiagonalNormal


class ContinuousNormalizingFlow:

    def __init__(self, flow):
        self.DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.flow = flow.to(self.DEVICE)
        self.context_encoder = nn.Identity().to(self.DEVICE)
        self.distribution = ConditionalDiagonalNormal(shape=[1]).to(self.DEVICE)

    def setup_context_encoder(self, context_encoder: nn.Module):
        self.context_encoder = context_encoder.to(self.DEVICE)

    def fit(
            self,
            X: np.ndarray,
            context: np.ndarray,
            params: Union[np.ndarray, None] = None,
            n_epochs: int = 100
    ):
        # Data
        X: torch.Tensor = torch.as_tensor(data=X, dtype=torch.float, device=self.DEVICE)
        context: torch.Tensor = torch.as_tensor(data=context, dtype=torch.float, device=self.DEVICE)
        params: torch.Tensor = torch.as_tensor(
            data=params,
            dtype=torch.float,
            device=self.DEVICE
        ) if params is not None else torch.zeros(X.shape[0], 2, device=self.DEVICE)  # Assuming Normal(0, 1) prior

        # Optimizer
        self.optimizer = optim.Adam(list(self.flow.parameters()) + list(self.context_encoder.parameters()))

        with tqdm(range(n_epochs)) as pbar:
            for _ in pbar:
                self.optimizer.zero_grad()

                logpx = self._log_prob(X, context, params)
                loss = -logpx.mean()

                loss.backward()
                self.optimizer.step()

                pbar.set_description(str(loss.item()))

        return self

    def _log_prob(self, X: torch.Tensor, context: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        zero: torch.Tensor = torch.zeros(X.shape[0], 1, device=X.device)
        context_e: torch.Tensor = self.context_encoder(context)
        z, delta_logp = self.flow(x=X, context=context_e, logpx=zero)

        logpz: torch.Tensor = self.distribution.log_prob(z, params)
        logpz: torch.Tensor = logpz.reshape(-1, 1)
        logpx: torch.Tensor = logpz - delta_logp
        return logpx

    @torch.no_grad()
    def log_prob(self, X: np.ndarray, context: np.ndarray, params: np.ndarray) -> np.ndarray:
        X: torch.Tensor = torch.as_tensor(data=X, dtype=torch.float, device=self.DEVICE)
        context: torch.Tensor = torch.as_tensor(data=context, dtype=torch.float, device=self.DEVICE)
        params: torch.Tensor = torch.as_tensor(
            data=params,
            dtype=torch.float,
            device=self.DEVICE
        ) if params is not None else torch.zeros(X.shape[0], 2, device=self.DEVICE) # Assuming Normal(0, 1) prior

        logpx: torch.Tensor = self._log_prob(X=X, context=context, params=params)
        logpx: np.ndarray = logpx.detach().cpu().numpy()
        return logpx

    @torch.no_grad()
    def sample(self, context: np.ndarray, num_samples: int = 10, params: Union[np.ndarray, None] = None) -> np.ndarray:
        context: torch.Tensor = torch.as_tensor(data=context, dtype=torch.float, device=self.DEVICE)
        params: torch.Tensor = torch.as_tensor(
            data=params,
            dtype=torch.float,
            device=self.DEVICE
        ) if params is not None else torch.zeros(context.shape[0], 2, device=self.DEVICE)  # Assuming Normal(0, 1) prior

        samples: torch.Tensor = self.distribution.sample(num_samples=num_samples, context=params)
        context_e: torch.Tensor = self.context_encoder(context)
        samples: torch.Tensor = self.flow(x=samples, context=context_e, reverse=True)
        samples: np.ndarray = samples.detach().cpu().numpy()
        return samples

    @torch.no_grad()
    def embed(self, context: np.ndarray) -> np.ndarray:
        context: torch.Tensor = torch.as_tensor(data=context, dtype=torch.float, device=self.DEVICE)
        context_e: torch.Tensor = self.context_encoder(context)
        context_e: np.ndarray = context_e.detach().cpu().numpy()
        return context_e
