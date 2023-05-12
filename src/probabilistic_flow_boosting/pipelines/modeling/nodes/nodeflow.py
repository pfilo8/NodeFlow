import sys
import logging
# import optuna
import pandas as pd
import numpy as np
import torch
from joblib import Parallel, delayed
import multiprocessing
from functools import partial
import multiprocessing
from functools import partial

from ..utils import generate_params_for_grid_search, setup_random_seed, split_data
from ...utils import log_dataframe_artifact
from ...reporting.nodes import calculate_nll

from ....nodeflow import NodeFlow


def train_nodeflow(x_train, y_train, x_val, y_val, model_params, hyperparams,
                   n_epochs: int = 100, batch_size: int = 1000, random_seed: int = 42, show_tqdm=True, ):
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

    m = nodeflow.fit(x_train.values, y_train.values, x_val, y_val, n_epochs=n_epochs, batch_size=batch_size, verbose=show_tqdm)
    return m


def worker(hyperparams,
        x_tr, x_val, y_tr, y_val, model_params, n_epochs: int = 100, batch_size: int = 1000, random_seed: int = 42):
    gpu_id = multiprocessing.current_process()._identity[0] % 4
    torch.cuda.set_device(gpu_id)
    print(gpu_id)
    show_tqdm = False
    setup_random_seed(random_seed)
    m = train_nodeflow(x_tr, y_tr, x_val, y_val, model_params, hyperparams, n_epochs, batch_size, random_seed, show_tqdm)
    result_train = calculate_nll(m, x_tr, y_tr, batch_size=batch_size)
    result_val = calculate_nll(m, x_val, y_val, batch_size=batch_size)

    # TODO: save best epoch
    print(hyperparams, result_train, result_val)
    results={"hyperparams": hyperparams, "result_train": result_train, "result_val": result_val}
    return results


def modeling_nodeflow(x_train: pd.DataFrame, y_train: pd.DataFrame, model_params, model_hyperparams,
                    split_size=0.8, n_epochs: int = 100, batch_size: int = 1000, random_seed: int = 42):
    x_tr, x_val, y_tr, y_val = split_data(x_train=x_train, y_train=y_train, split_size=split_size)
    
    model_hyperparams = [params for params in generate_params_for_grid_search(model_hyperparams)]
    params = dict(
        x_tr=x_tr,
        x_val=x_val,
        y_tr=y_tr,
        y_val=y_val,
        model_params=model_params,
        n_epochs=n_epochs,
        batch_size=batch_size,
        random_seed=random_seed
    )
    partial_worker = partial(worker, **params)
    results = Parallel(n_jobs=4)(delayed(partial_worker)(params) for params in model_hyperparams)
    print(results)

    results = pd.DataFrame(results, columns=['hyperparams', 'log_prob_train', 'log_prob_val', 'best_epoch'])
    results = results.sort_values('log_prob_val', ascending=True)
    log_dataframe_artifact(results, 'grid_search_results')
    best_params = results.iloc[0].to_dict()['hyperparams']
    m = train_nodeflow(x_tr, y_tr, x_val, y_val, model_params, best_params, n_epochs, batch_size, random_seed)
    return m