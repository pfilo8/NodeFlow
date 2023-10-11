import os
import uuid
import logging
import torch
import optuna
import pandas as pd

import warnings
from pytorch_lightning.utilities.warnings import PossibleUserWarning
warnings.filterwarnings("ignore", category=PossibleUserWarning)

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, StochasticWeightAveraging

from probabilistic_flow_boosting.pipelines.modeling.pytorch_lightning import PyTorchLightningPruningCallback
from probabilistic_flow_boosting.models.nodeflow import NodeFlow, NodeFlowDataModule

optuna.logging.enable_propagation()
logging.basicConfig(level=logging.INFO)

class CudaOutOfMemory(optuna.exceptions.OptunaError):
    def __init__(self, message):
        super().__init__(message)

def train_nodeflow(x_train, y_train, n_epochs, patience, split_size, batch_size, model_hyperparams):
    model = NodeFlow(input_dim=x_train.shape[1], output_dim=y_train.shape[1], **model_hyperparams)
    datamodule = NodeFlowDataModule(x_train, y_train, split_size=split_size, batch_size=batch_size)

    callbacks = [
        StochasticWeightAveraging(swa_lrs=1e-2),
        EarlyStopping(monitor="val_nll", patience=patience),
        ModelCheckpoint(monitor="val_nll", dirpath=f"tmp/", filename=f"model-{uuid.uuid4()}")
    ]
    trainer = Trainer(
        max_epochs=n_epochs,
        devices=1,
        check_val_every_n_epoch=1,
        accelerator="auto",
        callbacks=callbacks,
    )
    trainer.fit(model, datamodule=datamodule)
    best_model_path = trainer.checkpoint_callback.best_model_path
    return best_model_path

def objective(x_train, y_train, n_epochs, patience, split_size, batch_size, hparams, trial: optuna.trial.Trial) -> float:
    num_layers = trial.suggest_int("num_layers", *hparams["num_layers"])
    depth = trial.suggest_int("depth", *hparams["depth"])
    tree_output_dim = trial.suggest_int("tree_output_dim", *hparams["tree_output_dim"])
    num_trees = trial.suggest_int("num_trees", *hparams["num_trees"])

    flow_hidden_dims_size = trial.suggest_categorical("flow_hidden_dims_size", *hparams["flow_hidden_dims_size"])
    flow_hidden_dims_shape = trial.suggest_int("flow_hidden_dims_shape", *hparams["flow_hidden_dims_shape"])
    flow_hidden_dims = [flow_hidden_dims_size]*flow_hidden_dims_shape

    model_hyperparams = dict(
        num_layers=num_layers,
        depth=depth,
        tree_output_dim=tree_output_dim,
        num_trees=num_trees,
        flow_hidden_dims=flow_hidden_dims,
    )

    model = NodeFlow(input_dim=x_train.shape[1], output_dim=y_train.shape[1], **model_hyperparams)
    datamodule = NodeFlowDataModule(x_train, y_train, split_size=split_size, batch_size=batch_size)

    try:
        trainer = Trainer(
            logger=True,
            log_every_n_steps=1,
            enable_checkpointing=False,
            max_epochs=n_epochs,
            accelerator="auto",
            devices=1,
            callbacks=[
                StochasticWeightAveraging(swa_lrs=1e-2),
                EarlyStopping(monitor="val_nll", patience=patience),
                PyTorchLightningPruningCallback(trial, monitor="val_nll")
            ],
        )

        trainer.logger.log_hyperparams(model_hyperparams)
        trainer.fit(model, datamodule=datamodule)
        trial.set_user_attr("best_epoch", trainer.early_stopping_callback.stopped_epoch)
        trial.set_user_attr("total_epochs", trainer.current_epoch)
        return trainer.early_stopping_callback.best_score.item()
    except RuntimeError as exc:
        raise CudaOutOfMemory(str(exc))

def modeling_nodeflow(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    model_hyperparams,
    split_size=0.8,
    n_epochs: int = 100,
    patience: int = 10,
    batch_size: int = 1000,
    random_seed: int = 42,
):
    seed_everything(random_seed, workers=True) # sets seeds for numpy, torch and python.random.
    torch.cuda.set_per_process_memory_fraction(0.49)
    pruner = optuna.pruners.HyperbandPruner(min_resource=5, max_resource=n_epochs)
    # sampler = optuna.samplers.TPESampler(n_startup_trials=10)
    sampler = optuna.samplers.RandomSampler(seed=random_seed)
    study = optuna.create_study(direction="minimize", pruner=pruner, sampler=sampler)

    study.optimize(
        lambda trial: objective(
            x_train=x_train,
            y_train=y_train,
            n_epochs=n_epochs,
            patience=patience,
            split_size=split_size,
            batch_size=batch_size,
            hparams=model_hyperparams,
            trial=trial
        ),
        n_trials=500,
        timeout=7200,
        show_progress_bar=True,
        gc_after_trial=True,
        catch=(CudaOutOfMemory)
    )
    results = study.trials_dataframe()
    print(results)

    trial = study.best_trial
    print("Best trial: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
    model_params = trial.params
    model_params["flow_hidden_dims"] = [trial.params["flow_hidden_dims_size"]]*trial.params["flow_hidden_dims_shape"]
    del model_params["flow_hidden_dims_size"]
    del model_params["flow_hidden_dims_shape"]

    best_model_path = train_nodeflow(
        x_train=x_train,
        y_train=y_train,
        n_epochs=n_epochs,
        patience=patience,
        split_size=split_size,
        batch_size=batch_size,
        model_hyperparams=model_params
    )
    model = NodeFlow.load_from_checkpoint(best_model_path, input_dim=x_train.shape[1], output_dim=y_train.shape[1], **model_params)
    os.remove(best_model_path)
    
    return model, results, study
