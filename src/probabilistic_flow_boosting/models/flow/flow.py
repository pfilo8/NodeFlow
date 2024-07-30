import torch
import torch.nn as nn

import numpy as np
import uuid
from typing import Union

from torch.utils.data import DataLoader, TensorDataset
from torch import optim

from nflows.distributions import StandardNormal
from nflows.utils.torchutils import repeat_rows, split_leading_dim

from .odefunc import ODEfunc, ODEnet, divergence_bf
from .normalization import MovingBatchNorm1d
from .cnf import CNF, SequentialFlow


class ContinuousNormalizingFlow(nn.Module):

    def __init__(
            self,
            input_dim,
            hidden_dims,
            context_dim=0,
            num_blocks=3,
            conditional=False,
            layer_type="concatsquash",
            nonlinearity="tanh",
            time_length=0.5,
            train_T=True,
            solver='dopri5',
            atol=1e-5,
            rtol=1e-5,
            use_adjoint=True,
            batch_norm=True,
            bn_lag=0.0,
            sync_bn=False,
            device=None
    ):
        super().__init__()
        self.device = device

        self.flow = self.build_model(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            context_dim=context_dim,
            num_blocks=num_blocks,
            conditional=conditional,
            layer_type=layer_type,
            nonlinearity=nonlinearity,
            time_length=time_length,
            train_T=train_T,
            solver=solver,
            atol=atol,
            rtol=rtol,
            use_adjoint=use_adjoint,
            batch_norm=batch_norm,
            bn_lag=bn_lag,
            sync_bn=sync_bn
        )
        self.flow = self.flow.to(self.device)
        self.distribution = StandardNormal(shape=[input_dim]).to(self.device)

    @staticmethod
    def build_model(
            input_dim,
            hidden_dims,
            context_dim=0,
            num_blocks=3,
            conditional=False,
            layer_type="concatsquash",
            nonlinearity="tanh",
            time_length=0.5,
            train_T=True,
            solver='dopri5',
            atol=1e-5,
            rtol=1e-5,
            use_adjoint=True,
            batch_norm=True,
            bn_lag=0.0,
            sync_bn=False
    ):
        def build_cnf():
            diffeq = ODEnet(
                hidden_dims=hidden_dims,
                input_shape=(input_dim,),
                context_dim=context_dim,
                layer_type=layer_type,
                nonlinearity=nonlinearity,
            )
            odefunc = ODEfunc(
                diffeq=diffeq,
                divergence_fn=divergence_bf
            )
            cnf = CNF(
                odefunc=odefunc,
                T=time_length,
                train_T=train_T,
                conditional=conditional,
                solver=solver,
                use_adjoint=use_adjoint,
                atol=atol,
                rtol=rtol,
            )
            return cnf

        chain = [build_cnf() for _ in range(num_blocks)]
        if batch_norm:
            bn_layers = [MovingBatchNorm1d(input_dim, bn_lag=bn_lag, sync=sync_bn) for _ in range(num_blocks)]
            bn_chain = [MovingBatchNorm1d(input_dim, bn_lag=bn_lag, sync=sync_bn)]
            for a, b in zip(chain, bn_layers):
                bn_chain.append(a)
                bn_chain.append(b)
            chain = bn_chain
        model = SequentialFlow(chain)
        return model
    
    def setup_context_encoder(self, context_encoder: nn.Module):
        self.context_encoder = context_encoder.to(self.device)

    def fit(self, X: np.ndarray, context: np.ndarray, params: np.ndarray, X_val: Union[np.ndarray, None] = None,
            context_val: Union[np.ndarray, None] = None, params_val: Union[np.ndarray, None] = None,
            n_epochs: int = 100, batch_size: int = 1000, verbose: bool = False):
        X_t: torch.Tensor = torch.as_tensor(data=X, dtype=torch.float, device=self.device)
        context_t: torch.Tensor = torch.as_tensor(data=context, dtype=torch.float, device=self.device)
        params_t: torch.Tensor = torch.as_tensor(data=params, dtype=torch.float, device=self.device)

        dataset: DataLoader = DataLoader(
            dataset=TensorDataset(X_t, context_t, params_t),
            shuffle=True,
            batch_size=batch_size
        )

        self.optimizer = optim.Adam(list(self.flow.parameters()) + list(self.context_encoder.parameters()))

        mid = uuid.uuid4()  # To be able to run multiple experiments in parallel.
        loss_best = np.inf

        for i in range(n_epochs):
            for x, c, p in dataset:
                self.optimizer.zero_grad()

                logpx = self.log_prob(x, c, p)
                loss = -logpx.mean()

                loss.backward()
                self.optimizer.step()

            self._log(X, context, params, mode="train", batch_size=batch_size, verbose=verbose)

            if X_val is not None and context_val is not None and params_val is not None:
                loss_val = self._log(X_val, context_val, params_val, mode="val", batch_size=batch_size,
                                     verbose=verbose)
                # Save model if better
                if loss_val < loss_best:
                    self.epoch_best = i
                    loss_best = loss_val
                    self._save_temp(i, mid)

        if X_val is not None and context_val is not None and params_val is not None:
            return self._load_temp(self.epoch_best, mid)
        return self

    def log_prob(self, X: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Calculate log probability on data (usually batch)."""
        zero: torch.Tensor = torch.zeros(X.shape[0], 1, device=X.device)
        z, delta_logp = self.flow(x=X, context=context, logpx=zero)

        logpz: torch.Tensor = self.distribution.log_prob(z)
        logpz: torch.Tensor = logpz.reshape(-1, 1)
        logpx: torch.Tensor = logpz - delta_logp
        return logpx

    @torch.no_grad()
    def sample(self, context: torch.Tensor, num_samples: int = 10) -> torch.Tensor:
        """Sample from the model (usually batch)."""
        context_shape = context.shape

        context: torch.Tensor = torch.as_tensor(data=context, dtype=torch.float, device=context.device)
        context: torch.Tensor = repeat_rows(context, num_reps=num_samples)
        base_dist_samples: torch.Tensor = self.distribution.sample(num_samples=context.shape[0])

        samples: torch.Tensor = self.flow(x=base_dist_samples, context=context, reverse=True)
        samples: torch.Tensor = split_leading_dim(samples, [context_shape[0], num_samples])
        return samples
