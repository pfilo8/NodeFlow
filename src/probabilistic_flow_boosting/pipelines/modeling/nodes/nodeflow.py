import os
from typing import Optional, List, Any
from dataclasses import dataclass
import uuid
import logging
import numpy as np
import pandas as pd
from multiprocessing import current_process
from concurrent.futures import ProcessPoolExecutor


from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from ..utils import generate_params_for_grid_search
from probabilistic_flow_boosting.models.nodeflow import NodeFlow, NodeFlowDataModule

logging.basicConfig(level=logging.INFO)

def train_nodeflow(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    model_hyperparams,
    split_size=0.8,
    n_epochs: int = 100,
    batch_size: int = 1000,
    ckpt_path: Optional[str] = None,
):
    if current_process().name == 'MainProcess':
        gpu_n = 0
    else:
        gpu_n = (int(current_process().name.split("-")[-1]) - 1) % 4

    data_module = NodeFlowDataModule(x_train, y_train, split_size=split_size, batch_size=batch_size)
    model = NodeFlow(input_dim=x_train.shape[1], output_dim=y_train.shape[1], **model_hyperparams)

    total_params = sum(p.numel() for p in model.parameters())
    if total_params > 10_000_000:
        return "TOO_BIG", float('inf'), n_epochs
    
    trainer = Trainer(
        max_epochs=n_epochs,
        devices=[gpu_n],
        check_val_every_n_epoch=4,
        accelerator="cuda",
        callbacks=[
            ModelCheckpoint(monitor="val_nll", dirpath=f"tmp/", filename=f"model-{uuid.uuid4()}"),
            EarlyStopping(monitor="val_nll", patience=7),
        ],
    )
    trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_model_score = trainer.checkpoint_callback.best_model_score.item()
    stopped_epoch = trainer.early_stopping_callback.stopped_epoch

    return best_model_path, best_model_score, stopped_epoch

@dataclass
class _Mid_Train_Result:
    hparams: dict
    model_score: float = float('inf')
    model_path: Optional[str] = None
    epoch_trained: int = 0
    pruned: bool = False
    _mid_proc: Optional[Any] = None

    def get_dict(self) -> dict:
        return {
            "model_path": self.model_path,
            "model_score": self.model_score,
            "epoch_trained": self.epoch_trained,
            "pruned": self.pruned,
            "hparams": self.hparams,
        }

    def __repr__(self) -> str:
        return str(self.get_dict())

def prune(mid_results: List[_Mid_Train_Result]):
    median_score = np.median([r.model_score for r in mid_results if not r.pruned])
    for r in mid_results:
        r.pruned = r.model_score > median_score


def clean_pruned(mid_results: List[_Mid_Train_Result]):
    for r in mid_results:
        if r.pruned or (r.model_path == None):
            try:
                os.remove(r.model_path)
            except FileNotFoundError:
                pass

# def modeling_nodeflow(
#     x_train: pd.DataFrame,
#     y_train: pd.DataFrame,
#     model_hyperparams,
#     split_size=0.8,
#     n_epochs: int = 100,
#     batch_size: int = 1000,
#     random_seed: int = 42,
# ):
#     seed_everything(random_seed, workers=True) # sets seeds for numpy, torch and python.random.
#     mid_train_results = [_Mid_Train_Result(hparams=hparams) for hparams in generate_params_for_grid_search(model_hyperparams)]

#     with ProcessPoolExecutor(max_workers=4) as executor:
#         for r in mid_train_results:
#             if not r.pruned:
#                 r._mid_proc = executor.submit(
#                     train_nodeflow,
#                     x_train=x_train,
#                     y_train=y_train,
#                     model_hyperparams=r.hparams,
#                     split_size=split_size,
#                     batch_size=batch_size,
#                     n_epochs=n_epochs,
#                     ckpt_path=r.model_path,
#                 )

#         for r in mid_train_results:
#             _, r.model_score, _ = r._mid_proc.result()
#         for r in mid_train_results:
#             print(r)
    
    
#     results = pd.DataFrame([result.get_dict() for result in mid_train_results])
#     print(results.sort_values("model_score"))
#     top_model_params = results.sort_values("model_score")["hparams"].iloc[0]

#     best_model_path, _, _ = train_nodeflow(
#         x_train=x_train,
#         y_train=y_train,
#         model_hyperparams=top_model_params,
#         split_size=split_size,
#         n_epochs=n_epochs,
#         batch_size=batch_size,
#         gpu_n=0,
#     )
#     model = NodeFlow.load_from_checkpoint(best_model_path, input_dim=x_train.shape[1], output_dim=y_train.shape[1], **top_model_params)
    
#     return model, results


def modeling_nodeflow(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    model_hyperparams,
    split_size=0.8,
    n_epochs: int = 100,
    batch_size: int = 1000,
    random_seed: int = 42,
):
    seed_everything(random_seed, workers=True) # sets seeds for numpy, torch and python.random.

    train_epoch = 32
    epoch_factor = 2
    mid_train_results = [_Mid_Train_Result(hparams=hparams) for hparams in generate_params_for_grid_search(model_hyperparams)]

    while train_epoch < n_epochs:
        with ProcessPoolExecutor(max_workers=4) as executor:
            for r in mid_train_results:
                if not r.pruned:
                    r._mid_proc = executor.submit(
                        train_nodeflow,
                        x_train=x_train,
                        y_train=y_train,
                        model_hyperparams=r.hparams,
                        split_size=split_size,
                        batch_size=batch_size,
                        n_epochs=train_epoch,
                        ckpt_path=r.model_path,
                    )

        for r in mid_train_results:
            if not r.pruned:
                try:
                    _, r.model_score, _ = r._mid_proc.result(timeout=30)
                    r.epoch_trained = train_epoch
                except TimeoutError:
                    r.pruned = True
        train_epoch *= epoch_factor
        
        prune(mid_train_results)
        clean_pruned(mid_train_results)
        for r in mid_train_results:
            print(r)
    
    
    results = pd.DataFrame([result.get_dict() for result in mid_train_results])
    print(results.sort_values("model_score"))
    top_model_params = results.sort_values("model_score")["hparams"].iloc[0]

    best_model_path, _, _ = train_nodeflow(
        x_train=x_train,
        y_train=y_train,
        model_hyperparams=top_model_params,
        split_size=split_size,
        n_epochs=n_epochs,
        batch_size=batch_size,
    )
    model = NodeFlow.load_from_checkpoint(best_model_path, input_dim=x_train.shape[1], output_dim=y_train.shape[1], **top_model_params)
    
    return model, results
