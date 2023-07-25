import time
import torch
import random
import functools
import itertools
import multiprocessing
import numpy as np

from typing import Callable, Dict, List, Any


def setup_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)


def split_data(x_train, y_train, split_size=0.8):
    num_training_examples = int(split_size * x_train.shape[0])
    x_train, x_val = x_train.iloc[:num_training_examples, :], x_train.iloc[num_training_examples:, :]
    y_train, y_val = y_train.iloc[:num_training_examples, :], y_train.iloc[num_training_examples:, :]
    return x_train, x_val, y_train, y_val


def generate_params_for_grid_search(param_grid):
    return [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]


def run_multiple_gpu(
        target_func: Callable,
        target_parms: List[Dict[str, Any]],
        n_gpu: int = 1,
        n_jobs_per_gpu: int = 1,
        _update_stack_sleep: int = 3
    ):
    torch.multiprocessing.set_start_method('spawn', force=True)
    manager = multiprocessing.Manager()
    stack = manager.Queue()
    result_q = manager.Queue()

    for gpu in range(n_gpu):
        for _ in range(n_jobs_per_gpu):
            stack.put(gpu)

    processes = []
    for params in target_parms:
        while stack.empty():
            time.sleep(_update_stack_sleep)
        p = multiprocessing.Process(
            target=target_func,
            kwargs=dict(stack=stack, results=result_q, model_hyperparams=params)
        )
        p.start()
        processes.append(p)
        time.sleep(_update_stack_sleep)

    for p in processes:
        p.join()
    
    results = []
    while not result_q.empty():
        results.append(result_q.get())
    return results


def as_worker(func):
    @functools.wraps(func)
    def wrapper_worker(stack, results, model_hyperparams, *args, **kwargs):
        try:
            gpu_id = stack.get()
            torch.cuda.set_device(gpu_id)
            torch.cuda.empty_cache()
            print("Taking: ", gpu_id)

            func_results = func(model_hyperparams=model_hyperparams, *args, **kwargs)

            if not isinstance(func_results, dict):
                raise TypeError(f"Function {func.__name__} has to return dict!")
            
            results.put(func_results)
            
        except Exception as e:
            print(e)
        finally:
            stack.put(gpu_id)
            print("Free: ", gpu_id)
    return wrapper_worker