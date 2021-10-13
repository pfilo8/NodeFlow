from .odefunc import ODEfunc, ODEnet
from .normalization import MovingBatchNorm1d
from .cnf import CNF, SequentialFlow


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
