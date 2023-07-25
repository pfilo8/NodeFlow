import os
import sys
import logging
import pandas as pd
import torch
import time

from functools import partial

from ..utils import generate_params_for_grid_search, setup_random_seed, split_data, run_multiple_gpu, as_worker
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

    m = nodeflow.fit(x_train.values, y_train.values, x_val, y_val,
                    n_epochs=n_epochs, batch_size=batch_size, verbose=show_tqdm, max_patience=30)
    return m

@as_worker
def train_nodeflow_worker(
    x_tr, x_val, y_tr, y_val,
    model_params,
    model_hyperparams,
    n_epochs: int = 100,
    batch_size: int = 1000,
    random_seed: int = 42
):
    show_tqdm = True
    setup_random_seed(random_seed)

    start_time = time.time()
    m = train_nodeflow(x_tr, y_tr, x_val, y_val, model_params, model_hyperparams, n_epochs, batch_size, random_seed, show_tqdm)
    elapsed_time = time.time() - start_time
    
    result_train = calculate_nll(m, x_tr, y_tr, batch_size=batch_size)
    result_val = calculate_nll(m, x_val, y_val, batch_size=batch_size)
    best_epoch = m._best_epoch
    os.remove(f"/tmp/model_{m.mid}.pt")

    print(model_hyperparams, result_train, result_val, best_epoch, elapsed_time)
    return {
        "hyperparams": model_hyperparams,
        "log_prob_train": result_train,
        "log_prob_val": result_val,
        "best_epoch": best_epoch,
        "train_time": elapsed_time
    }


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
    partial_worker = partial(train_nodeflow_worker, **params)

    results = run_multiple_gpu(partial_worker, model_hyperparams, n_gpu=4, n_jobs_per_gpu=2)

    results = pd.DataFrame(results, columns=['hyperparams', 'log_prob_train', 'log_prob_val', 'best_epoch', 'train_time'])
    results = results.sort_values('log_prob_val', ascending=True)
    print(results)
    log_dataframe_artifact(results, 'grid_search_results')
    best_params = results.iloc[0].to_dict()['hyperparams']
    torch.cuda.empty_cache()
    m = train_nodeflow(x_tr, y_tr, x_val, y_val, model_params, best_params, n_epochs, batch_size, random_seed)
    return m, results