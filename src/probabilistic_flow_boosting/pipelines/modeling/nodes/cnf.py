import sys
import logging
# import optuna
import pandas as pd
import numpy as np
import torch
from joblib import Parallel, delayed
import time
import multiprocessing
from functools import partial
import multiprocessing
from functools import partial

from ..utils import generate_params_for_grid_search, setup_random_seed, split_data
from ...utils import log_dataframe_artifact
from ...reporting.nodes import calculate_nll

from ....cnf import ContinuousNormalizingFlowRegressor


def train_cnf(x_train, y_train, x_val, y_val, model_params, hyperparams,
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
    model = ContinuousNormalizingFlowRegressor(
        # embedding_dim: int = 40,
        # hidden_dims: Iterable[int] = (80, 40),
        # num_blocks: int = 3,
        # layer_type: str = "concatsquash",
        # nonlinearity: str = "tanh",
        ###
        input_dim=x_train.shape[1],
        output_dim=y_train.shape[1],
        **model_params,
        **hyperparams
    )
    if x_val is not None and y_val is not None:
        x_val, y_val = x_val.values, y_val.values

    m = model.fit(x_train.values, y_train.values, x_val, y_val,
                    n_epochs=n_epochs, batch_size=batch_size, verbose=show_tqdm, max_patience=30)
    return m


def worker(stack, results, hyperparams,
        x_tr, x_val, y_tr, y_val, model_params, n_epochs: int = 100, batch_size: int = 1000, random_seed: int = 42):
    try:
        gpu_id = stack.get()
        # torch.cuda.set_device(gpu_id)
        # torch.cuda.empty_cache()
        print("Taking: ", gpu_id)
        show_tqdm = True
        setup_random_seed(random_seed)
        m = train_cnf(x_tr, y_tr, x_val, y_val, model_params, hyperparams, n_epochs, batch_size, random_seed, show_tqdm)
        result_train = calculate_nll(m, x_tr, y_tr, batch_size=batch_size)
        result_val = calculate_nll(m, x_val, y_val, batch_size=batch_size)
        best_epoch = m._best_epoch

        # TODO: save best epoch
        print(hyperparams, result_train, result_val, best_epoch)
        results.put({
            "hyperparams": hyperparams,
            "log_prob_train": result_train,
            "log_prob_val": result_val,
            "best_epoch": best_epoch,
        })
    except Exception as e:
        print(e)
    finally:
        stack.put(gpu_id)
        print("Free: ", gpu_id)




def modeling_cnf(x_train: pd.DataFrame, y_train: pd.DataFrame, model_params, model_hyperparams,
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
    # torch.multiprocessing.set_start_method('spawn', force=True)
    partial_worker = partial(worker, **params)
    manager = multiprocessing.Manager()
    stack = manager.Queue()
    result_q = manager.Queue()
    stack.put(0)
    stack.put(1)
    stack.put(0)
    stack.put(1)

    processes = []
    for params in model_hyperparams:
        while stack.empty():
            time.sleep(5)
        p = multiprocessing.Process(target=partial_worker, args=(stack,result_q,params,))
        p.start()
        processes.append(p)
        time.sleep(3)
    
    for p in processes:
        p.join()

    results = []
    while not result_q.empty():
        results.append(result_q.get())
    print(results)


    results = pd.DataFrame(results, columns=['hyperparams', 'log_prob_train', 'log_prob_val', 'best_epoch'])
    results = results.sort_values('log_prob_val', ascending=True)
    print(results)
    log_dataframe_artifact(results, 'grid_search_results')
    best_params = results.iloc[0].to_dict()['hyperparams']
    torch.cuda.empty_cache()
    m = train_cnf(x_tr, y_tr, x_val, y_val, model_params, best_params, n_epochs, batch_size, random_seed)
    return m, results