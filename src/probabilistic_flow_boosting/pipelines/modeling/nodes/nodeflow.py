import sys
import logging
import optuna
import pandas as pd
import numpy as np

from ..utils import generate_params_for_grid_search, setup_random_seed, split_data
from ...utils import log_dataframe_artifact
from ...reporting.nodes import calculate_nll

from ....nodeflow import NodeFlow


def train_nodeflow(x_train, y_train, x_val, y_val, model_params, hyperparams,
                   n_epochs: int = 100, batch_size: int = 1000, random_seed: int = 42):
    """
    Train a TreeFlow model.
    :param x_train: Training data.
    :param y_train: Training labels.
    :param x_val: Validation data.
    :param y_val: Validation labels.
    :param flow_p: Flow parameters from grid search.
    :param flow_params: Flow parameters.
    :param tree_p: Tree parameters from grid search.
    :param tree_params: Tree parameters.
    :param tree_model_type: Type of the Tree model (see tfboost.tree package).
    :param n_epochs: Number of epochs.
    :param batch_size: Batch size for Flow model.
    :param random_seed: Random seed.
    :return:
    """
    setup_random_seed(random_seed)
    nodeflow = NodeFlow(
        input_dim=x_train.shape[1],
        output_dim=y_train.shape[1],
        **model_params,
        **hyperparams
    )
    if x_val is not None and y_val is not None:
        x_val, y_val = x_val.values, y_val.values

    m = nodeflow.fit(x_train.values, y_train.values, x_val, y_val, n_epochs=n_epochs, batch_size=batch_size)
    return m


def modeling_nodeflow(x_train: pd.DataFrame, y_train: pd.DataFrame, optuna_db: str, model_params, model_hyperparams,
                    split_size=0.8, n_epochs: int = 100, batch_size: int = 1000, random_seed: int = 42):
    setup_random_seed(random_seed)
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler())
    
    x_tr, x_val, y_tr, y_val = split_data(x_train=x_train, y_train=y_train, split_size=split_size)
    
    def objective(trial):
        hyperparams = {
            k: trial.suggest_categorical(k, v) for k, v in model_hyperparams.items()
        }
        m = train_nodeflow(x_tr, y_tr, x_val, y_val, model_params, hyperparams, n_epochs, batch_size, random_seed)

        result_train = calculate_nll(m, x_tr, y_tr, batch_size=batch_size)
        result_val = calculate_nll(m, x_val, y_val, batch_size=batch_size)

        # TODO: save best epoch
        logging.info(f"{hyperparams}, {result_train}, {result_val}")
#         print(hyperparams, result_train, result_val)#, best_epoch)
#         results.append([hyperparams, result_train, result_val])
        return result_val

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = optuna_db.split("/")[-1]  # Unique identifier of the study.
    storage_name = "sqlite:///{}".format(optuna_db)
    logging.info(f"{storage_name}")
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction='minimize',
        sampler=optuna.samplers.GridSampler(model_hyperparams),
        load_if_exists=True
    )
    num_trials = np.prod([len(p) for p in model_hyperparams.values()])
    study.optimize(objective, n_trials=num_trials)
    
    df = study.trials_dataframe()
    best_params = study.best_trial.params
    print(best_params)
    m = train_nodeflow(x_train, y_train, None, None, model_params, best_params, n_epochs, batch_size, random_seed)
    return df, m
