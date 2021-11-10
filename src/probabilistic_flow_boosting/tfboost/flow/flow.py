import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from nflows.distributions import ConditionalDiagonalNormal

from .odefunc import ODEfunc, ODEnet
from .normalization import MovingBatchNorm1d
from .cnf import CNF, SequentialFlow


class ContinuousNormalizingFlow:

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
            sync_bn=False
    ):
        self.DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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
        self.flow = self.flow.to(self.DEVICE)
        self.context_encoder = nn.Identity().to(self.DEVICE)
        self.distribution = ConditionalDiagonalNormal(shape=[input_dim], context_encoder=nn.Identity()).to(self.DEVICE)

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
        self.context_encoder = context_encoder.to(self.DEVICE)

    def fit(self, X: np.ndarray, context: np.ndarray, params: np.ndarray, n_epochs: int = 100):
        X: torch.Tensor = torch.as_tensor(data=X, dtype=torch.float, device=self.DEVICE)
        context: torch.Tensor = torch.as_tensor(data=context, dtype=torch.float, device=self.DEVICE)
        params: torch.Tensor = torch.as_tensor(data=params, dtype=torch.float, device=self.DEVICE)

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
        params: torch.Tensor = torch.as_tensor(data=params, dtype=torch.float, device=self.DEVICE)

        logpx: torch.Tensor = self._log_prob(X=X, context=context, params=params)
        logpx: np.ndarray = logpx.detach().cpu().numpy()
        return logpx

    @torch.no_grad()
    def sample(self, context: np.ndarray, params: np.ndarray, num_samples: int = 10) -> np.ndarray:
        context: torch.Tensor = torch.as_tensor(data=context, dtype=torch.float, device=self.DEVICE)
        params: torch.Tensor = torch.as_tensor(data=params, dtype=torch.float, device=self.DEVICE)

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
