# Copyright 2021 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
# or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This is a boilerplate pipeline 'reporting'
generated using Kedro 0.17.5
"""
from typing import Any, Dict, List, Tuple, Union

import time

import catboost
import matplotlib.pyplot as plt
# import mlflow
# import ngboost
import numpy as np
import pandas as pd
import properscoring as ps
import torch

from scipy import signal
from sklearn.metrics import mean_squared_error, mean_absolute_error
from nflows.distributions import ConditionalDiagonalNormal

from lightning.pytorch import Trainer, seed_everything

from .utils import batch, KDE
from ..utils import log_dataframe_artifact

# from ...pgbm import PGBM
from probabilistic_flow_boosting.models.tfboost.tfboost import TreeFlowBoost
from probabilistic_flow_boosting.models.nodeflow import NodeFlow, NodeFlowDataModule
from probabilistic_flow_boosting.models.cnf import ContinuousNormalizingFlowRegressor, CNFDataModule
from probabilistic_flow_boosting.models.independent_multivariate_boosting import IndependentNGBoost


def calculate_rmse(model: TreeFlowBoost, x: pd.DataFrame, y: pd.DataFrame, num_samples: int, batch_size: int):
    x: np.ndarray = x.values
    y: np.ndarray = y.values

    y_hat: np.ndarray = model.predict(x, num_samples=num_samples, batch_size=batch_size)
    return mean_squared_error(y, y_hat, squared=False)


def calculate_mae(model: TreeFlowBoost, x: pd.DataFrame, y: pd.DataFrame, num_samples: int, batch_size: int):
    x: np.ndarray = x.values
    y: np.ndarray = y.values

    y_hat: np.ndarray = model.predict(x, num_samples=num_samples, batch_size=batch_size)
    return mean_absolute_error(y, y_hat)


def calculate_nll(model: TreeFlowBoost, x: pd.DataFrame, y: pd.DataFrame, batch_size: int):
    x: np.ndarray = x.values
    y: np.ndarray = y.values
    return -model.log_prob(x, y, batch_size=batch_size).mean()

def calculate_peaks_from_sample(sample, find_peaks_parameters, n_peaks=2):
    kde = KDE()
    density, support = kde(sample)
    peaks_id, _ = signal.find_peaks(density, **find_peaks_parameters)
    peaks_order = np.argsort(-density[peaks_id])  # Sort in descending order.
    peaks = support[peaks_id]
    peaks = peaks[peaks_order]
    if len(peaks_id) == 0:
        peaks = np.repeat(np.mean(sample), n_peaks)
    peaks = np.resize(peaks, n_peaks)
    return peaks

def calculate_peaks(samples: np.ndarray, find_peaks_parameters: Dict[Any, Any] = None, n_peaks: int = 2):
    if find_peaks_parameters is None:
        find_peaks_parameters = {}
    results = [calculate_peaks_from_sample(sample, find_peaks_parameters, n_peaks) for sample in samples]
    results = np.array(results)
    return results

def rmse_at_k(y_true: np.ndarray, y_score: List[np.ndarray]):
    results = []

    for i in range(y_true.shape[0]):
        results.append(np.min((y_score[i] - y_true[i]) ** 2))  # Take closer prediction

    return np.sqrt(np.average(results))

def _calculate_rmse_at_k(model: TreeFlowBoost, x: pd.DataFrame, y: pd.DataFrame, num_samples: int, batch_size: int,
                         k: int = 2, find_peaks_parameters: Dict[Any, Any] = None):
    if find_peaks_parameters is None:
        find_peaks_parameters = {"height": 0.1}  # Proposed default

    x: np.ndarray = x.values
    y: np.ndarray = y.values

    # For TreeFlow
    samples = []
    for i in batch(range(num_samples), 100):
        samples.append(model.sample(x, num_samples=len(i), batch_size=batch_size).squeeze(-1))
    samples = np.concatenate(samples, axis=1)

    # For CNF
    # samples = model.sample(x, num_samples=num_samples, batch_size=batch_size)

    y_test_treeflow_peaks = calculate_peaks(samples, find_peaks_parameters)
    y_test_treeflow_peaks = [s[:k] for s in y_test_treeflow_peaks]

    return rmse_at_k(y, y_test_treeflow_peaks)

def calculate_metrics_nodeflow(
        model: NodeFlow,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        num_samples: int,
        batch_size: int,
        sample_batch_size: int,
        find_peaks_parameters: Dict[Any, Any] = None
    ):
    # torch.cuda.set_per_process_memory_fraction(0.49)
    y_test = y_test.to_numpy()
    torch.cuda.empty_cache()
    # NLL
    datamodule = NodeFlowDataModule(X_train, y_train, X_test, y_test, split_size=0.8, batch_size=sample_batch_size)
    trainer = Trainer(
        max_epochs=100,
        devices=1,
        check_val_every_n_epoch=4,
        accelerator="cuda",
        enable_checkpointing=False,
        inference_mode=False,
    )
    start_nll = time.time()
    test_results = trainer.test(model, datamodule=datamodule)
    nll = np.mean([val["test_nll"] for val in test_results])
    nll_time = time.time() - start_nll
    if y_test.shape[-1] != 1:
        return nll, -1, -1, -1, nll_time, -1, -1

    start_rmse = time.time()
    samples = trainer.predict(model, datamodule=datamodule, return_predictions=True)
    samples[-1] = np.atleast_2d(samples[-1])
    samples = np.concatenate(samples).reshape(-1, 1000)

    if find_peaks_parameters is None:
        find_peaks_parameters = {"height": 0.1}  # Proposed default
    # RMSE
    peaks_1 = calculate_peaks(samples, find_peaks_parameters, n_peaks=1)
    peaks_2 = calculate_peaks(samples, find_peaks_parameters, n_peaks=2)
    rmse_1 = np.sqrt(np.mean((y_test-peaks_1)**2))
    rmse_2 = np.sqrt(np.mean(np.min((np.tile(y_test, 2)-peaks_2)**2, axis=1)))
    rmse_time = time.time() - start_rmse

    # CRPS
    start_crps = time.time()
    y_test = y_test.reshape(-1)
    crpss = []
    for o, f in zip(batch(y_test, 100), batch(samples, 100)):
        crpss.append(ps.crps_ensemble(
            observations=o,
            forecasts=f
        ))
    crpss = np.concatenate(crpss)
    crps_time = time.time() - start_crps
    return nll, rmse_1, rmse_2, crpss.mean(), nll_time, rmse_time, crps_time

def calculate_metrics_cnf(
        model: ContinuousNormalizingFlowRegressor,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        num_samples: int,
        batch_size: int,
        sample_batch_size: int,
        find_peaks_parameters: Dict[Any, Any] = None
    ):
    torch.cuda.set_per_process_memory_fraction(0.49)
    y_test = y_test.to_numpy()
    torch.cuda.empty_cache()
    # NLL
    datamodule = CNFDataModule(X_train, y_train, X_test, y_test, split_size=0.8, batch_size=sample_batch_size)
    trainer = Trainer(
        max_epochs=100,
        devices=1,
        check_val_every_n_epoch=4,
        accelerator="cuda",
        enable_checkpointing=False,
        inference_mode=False,
    )
    start_nll = time.time()
    test_results = trainer.test(model, datamodule=datamodule)
    nll = np.mean([val["test_nll"] for val in test_results])
    nll_time = time.time() - start_nll

    start_rmse = time.time()
    samples = trainer.predict(model, datamodule=datamodule, return_predictions=True)
    samples[-1] = np.atleast_2d(samples[-1])
    samples = np.concatenate(samples).reshape(-1, 1000)

    if find_peaks_parameters is None:
        find_peaks_parameters = {"height": 0.1}  # Proposed default
    # RMSE
    peaks_1 = calculate_peaks(samples, find_peaks_parameters, n_peaks=1)
    peaks_2 = calculate_peaks(samples, find_peaks_parameters, n_peaks=2)
    rmse_1 = np.sqrt(np.mean((y_test-peaks_1)**2))
    rmse_2 = np.sqrt(np.mean(np.min((np.tile(y_test, 2)-peaks_2)**2, axis=1)))
    rmse_time = time.time() - start_rmse

    # CRPS
    start_crps = time.time()
    y_test = y_test.reshape(-1)
    crpss = []
    for o, f in zip(batch(y_test, 100), batch(samples, 100)):
        crpss.append(ps.crps_ensemble(
            observations=o,
            forecasts=f
        ))
    crpss = np.concatenate(crpss)
    crps_time = time.time() - start_crps
    return nll, rmse_1, rmse_2, crpss.mean(), nll_time, rmse_time, crps_time

def calculate_rmse_at_1(model: TreeFlowBoost, x: pd.DataFrame, y: pd.DataFrame, num_samples: int, batch_size: int):
    return _calculate_rmse_at_k(model=model, x=x, y=y, num_samples=num_samples, batch_size=batch_size, k=1,
                                find_peaks_parameters={"height": 0.1})


def calculate_rmse_at_2(model: TreeFlowBoost, x: pd.DataFrame, y: pd.DataFrame, num_samples: int, batch_size: int):
    return _calculate_rmse_at_k(model=model, x=x, y=y, num_samples=num_samples, batch_size=batch_size, k=2,
                                find_peaks_parameters={"height": 0.1})


def calculate_rmse_at_3(model: TreeFlowBoost, x: pd.DataFrame, y: pd.DataFrame, num_samples: int, batch_size: int):
    return _calculate_rmse_at_k(model=model, x=x, y=y, num_samples=num_samples, batch_size=batch_size, k=3,
                                find_peaks_parameters={"height": 0.1})


def calculate_crps(model: TreeFlowBoost, x: pd.DataFrame, y: pd.DataFrame, num_samples: int, batch_size: int):
    x: np.ndarray = x.values
    y: np.ndarray = y.values
    y = y.reshape(-1)

    samples = []
    for i in batch(range(num_samples), 100):
        samples.append(model.sample(x, num_samples=len(i), batch_size=batch_size).squeeze(-1))
    samples = np.concatenate(samples, axis=1)

    crpss = []
    for o, f in zip(batch(y, 100), batch(samples, 100)):
        crpss.append(ps.crps_ensemble(
            observations=o,
            forecasts=f
        ))
    crpss = np.concatenate(crpss)
    return crpss.mean()


def calculate_crps_treeflow(model: TreeFlowBoost, x: pd.DataFrame, y: pd.DataFrame, num_samples: int, batch_size: int):
    x: np.ndarray = x.values
    y: np.ndarray = y.values
    y = torch.Tensor(y.reshape(-1))

    samples = []
    for i in batch(range(num_samples), batch_size):
        samples.append(model.sample(x, num_samples=len(i), batch_size=batch_size).squeeze(-1))
    yhat_dist = torch.Tensor(np.concatenate(samples, axis=1))
    yhat_dist = yhat_dist.T

    n_forecasts = yhat_dist.shape[0]
    # Sort the forecasts in ascending order
    yhat_dist_sorted, _ = torch.sort(yhat_dist, 0)
    # Create temporary tensors
    y_cdf = torch.zeros_like(y)
    yhat_cdf = torch.zeros_like(y)
    yhat_prev = torch.zeros_like(y)
    crps = torch.zeros_like(y)
    # Loop over the samples generated per observation
    for yhat in yhat_dist_sorted:
        flag = (y_cdf == 0) * (y < yhat)
        crps += flag * ((y - yhat_prev) * yhat_cdf ** 2)
        crps += flag * ((yhat - y) * (yhat_cdf - 1) ** 2)
        y_cdf += flag
        crps += ~flag * ((yhat - yhat_prev) * (yhat_cdf - y_cdf) ** 2)
        yhat_cdf += 1 / n_forecasts
        yhat_prev = yhat

    # In case y_cdf == 0 after the loop
    flag = (y_cdf == 0)
    crps += flag * (y - yhat)
    return crps


def calculate_rmse_tree(model: TreeFlowBoost, x: pd.DataFrame, y: pd.DataFrame):
    x: np.ndarray = x.values
    y: np.ndarray = y.values

    y_hat: np.ndarray = model.predict_tree(x)

    if y.shape[1] == 1:
        y_hat: np.ndarray = y_hat[:, 0]  # Only get mean
    return mean_squared_error(y, y_hat, squared=False)


def calculate_mae_tree(model: TreeFlowBoost, x: pd.DataFrame, y: pd.DataFrame):
    x: np.ndarray = x.values
    y: np.ndarray = y.values

    y_hat: np.ndarray = model.predict_tree(x)
    if y.shape[1] == 1:
        y_hat: np.ndarray = y_hat[:, 0]  # Only get mean
    return mean_absolute_error(y, y_hat)


def calculate_nll_tree(model: TreeFlowBoost, x: pd.DataFrame, y: pd.DataFrame):
    x: np.ndarray = x.values
    y: np.ndarray = y.values

    if y.shape[1] > 1:
        return np.nan

    y_hat_tree = model.predict_tree(x)
    y_hat_tree[:, 1] = np.log(np.sqrt(y_hat_tree[:, 1]))  # Transform var to log std / CatBoost RMSEWithUncertainty

    distribution = ConditionalDiagonalNormal(shape=[1])  # Assume 1D distribution
    return -distribution.log_prob(y, y_hat_tree).numpy().mean()


def calculate_nll_catboost(model: catboost.CatBoostRegressor, x: pd.DataFrame, y: pd.DataFrame):
    x: np.ndarray = x.values
    y: np.ndarray = y.values

    if y.shape[1] > 1:
        return np.nan

    y_hat_tree = model.predict(x)
    y_hat_tree[:, 1] = np.log(np.sqrt(y_hat_tree[:, 1]))  # Transform var to log std / CatBoost RMSEWithUncertainty

    distribution = ConditionalDiagonalNormal(shape=[1])  # Assume 1D distribution
    return -distribution.log_prob(y, y_hat_tree).numpy().mean()


def calculate_rmse_catboost(model: catboost.CatBoostRegressor, x: pd.DataFrame, y: pd.DataFrame):
    x: np.ndarray = x.values
    y: np.ndarray = y.values

    y_hat: np.ndarray = model.predict(x)

    if y.shape[1] == 1:
        y_hat: np.ndarray = y_hat[:, 0]  # Only get mean
    return mean_squared_error(y, y_hat, squared=False)


def calculate_crps_catboost(model: catboost.CatBoostRegressor, x: pd.DataFrame, y: pd.DataFrame):
    x: np.ndarray = x.values
    y: np.ndarray = y.values

    if y.shape[1] > 1:
        return np.nan

    y_hat_tree = model.predict(x)
    y_hat_tree[:, 1] = np.sqrt(y_hat_tree[:, 1])  # Transform var to log std / CatBoost RMSEWithUncertainty

    return ps.crps_gaussian(
        y.reshape(-1),
        y_hat_tree[:, 0],
        y_hat_tree[:, 1]
    ).mean()


def plot_loss_function(model: TreeFlowBoost):
    losses = model.flow_model.losses
    x = range(len(losses["train"]))

    fig, ax = plt.subplots(1, 1)
    ax.plot(x, losses["train"])
    if len(losses["val"]) > 0:
        ax.plot(x, losses["val"])
    # mlflow.log_figure(fig, f'losses-{str(datetime.datetime.now()).replace(" ", "-")}.png')


def calculate_nll_ngboost(model, x: pd.DataFrame, y: pd.DataFrame,
                          independent=False) -> float:
    x: np.ndarray = x.values
    y: np.ndarray = y.values

    if independent:
        y_dists = model.scipy_distribution(x)
    else:
        y_dists = model.pred_dist(x).scipy_distribution()
    nlls = [-y_dists[i].logpdf(y[i, :]) for i in range(y.shape[0])]
    return np.mean(nlls)


def calculate_rmse_pgbm(model, x: torch.Tensor, y: torch.Tensor):
    y_hat = model.predict(x)
    return mean_squared_error(y, y_hat, squared=False)


def summary(
        train_results_rmse: float,
        train_results_mae: float,
        train_results_nll: float,
        train_results_rmse_at_1: float,
        train_results_rmse_at_2: float,
        train_results_rmse_at_3: float,
        train_results_rmse_tree: float,
        train_results_mae_tree: float,
        train_results_nll_tree: float,
        test_results_rmse: float,
        test_results_mae: float,
        test_results_nll: float,
        test_results_rmse_at_1: float,
        test_results_rmse_at_2: float,
        test_results_rmse_at_3: float,
        test_results_rmse_tree: float,
        test_results_mae_tree: float,
        test_results_nll_tree: float
):
    results = pd.DataFrame([
        ['train', 'rmse', train_results_rmse],
        ['train', 'mae', train_results_mae],
        ['train', 'nll', train_results_nll],
        ['train', 'rmse_at_1', train_results_rmse_at_1],
        ['train', 'rmse_at_2', train_results_rmse_at_2],
        ['train', 'rmse_at_3', train_results_rmse_at_3],
        ['train', 'rmse_tree', train_results_rmse_tree],
        ['train', 'mae_tree', train_results_mae_tree],
        ['train', 'nll_tree', train_results_nll_tree],
        ['test', 'rmse', test_results_rmse],
        ['test', 'mae', test_results_mae],
        ['test', 'nll', test_results_nll],
        ['test', 'rmse_at_1', test_results_rmse_at_1],
        ['test', 'rmse_at_2', test_results_rmse_at_2],
        ['test', 'rmse_at_3', test_results_rmse_at_3],
        ['test', 'rmse_tree', test_results_rmse_tree],
        ['test', 'mae_tree', test_results_mae_tree],
        ['test', 'nll_tree', test_results_nll_tree],
    ],
        columns=[
            'set', 'metric', 'value'
        ]
    )
    log_dataframe_artifact(results, "test_results")
    return results


def summary_ngboost(
        train_results_nll: float,
        test_results_nll: float,
):
    results = pd.DataFrame([
        ['train', 'nll', train_results_nll],
        ['test', 'nll', test_results_nll],
    ],
        columns=[
            'set', 'metric', 'value'
        ]
    )
    log_dataframe_artifact(results, "test_results")
    return results

def summary_nodeflow(
        train_results_nll: float,
        train_nll_time: float,
        test_results_nll: float,
        test_nll_time: float,
        train_results_rmse_1: float,
        test_results_rmse_1: float,
        train_results_rmse_2: float,
        train_rmse_2_time: float,
        test_results_rmse_2: float,
        test_rmse_2_time: float,
        train_results_crps: float,
        train_crps_time: float,
        test_results_crps: float,
        test_crps_time: float,
):
    results = pd.DataFrame([
        ['train', 'nll', train_results_nll],
        ['test', 'nll', test_results_nll],
        ['train', 'rmse_1', train_results_rmse_1],
        ['test', 'rmse_1', test_results_rmse_1],
        ['train', 'rmse_2', train_results_rmse_2],
        ['test', 'rmse_2', test_results_rmse_2],
        ['train', 'crps', train_results_crps],
        ['test', 'crps', test_results_crps],
        ['train', 'nll_time', train_nll_time],
        ['test', 'nll_time', test_nll_time],
        ['train', 'rmse_2_time', train_rmse_2_time],
        ['test', 'rmse_2_time', test_rmse_2_time],
        ['train', 'crps_time', train_crps_time],
        ['test', 'crps_time', test_crps_time],
    ],
        columns=[
            'set', 'metric', 'value'
        ]
    )
    # log_dataframe_artifact(results, "test_results")
    return results
    


def aggregated_report(*inputs: Tuple[pd.DataFrame]) -> pd.DataFrame:
    df = pd.concat(inputs)
    results = df.groupby(['set', 'metric']).agg([np.mean, np.std])
    results = results.reset_index()
    results.columns = ['set', 'metric', 'mean', 'std']

    # MLFlow
    results["set-metric"] = results["set"] + "_" + results["metric"]
    # mlflow.log_metrics({f"{k}_mean": v for k, v in zip(results["set-metric"].values, results["mean"].values)})
    # mlflow.log_metrics({f"{k}_std": v for k, v in zip(results["set-metric"].values, results["std"].values)})
    return results
