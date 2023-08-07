import os
import sys
import logging
import pandas as pd
import torch
import time

from functools import partial

logging.basicConfig(level=logging.INFO)
import matplotlib.pyplot as plt
from lightning.pytorch import Trainer, seed_everything

from ..utils import generate_params_for_grid_search

from probabilistic_flow_boosting.models.nodeflow import NodeFlow, NodeFlowDataModule


def modeling_nodeflow(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    model_params,
    model_hyperparams,
    split_size=0.8,
    n_epochs: int = 100,
    batch_size: int = 1000,
    random_seed: int = 42,
):
    seed_everything(random_seed, workers=True) # sets seeds for numpy, torch and python.random.

    data_module = NodeFlowDataModule(x_train, y_train, split_size=split_size, batch_size=batch_size)

    model_hyperparams = [params for params in generate_params_for_grid_search(model_hyperparams)][0]
    model = NodeFlow(input_dim=x_train.shape[1], output_dim=y_train.shape[1], **model_params, **model_hyperparams)

    trainer = Trainer(max_epochs=n_epochs, check_val_every_n_epoch=5)
    trainer.fit(model, datamodule=data_module)

    results = pd.DataFrame()
    return results
