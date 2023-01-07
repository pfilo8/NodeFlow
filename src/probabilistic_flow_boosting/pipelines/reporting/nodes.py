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

import datetime

import catboost
import matplotlib.pyplot as plt
import mlflow
import ngboost
import numpy as np
import pandas as pd
import properscoring as ps

from scipy import signal
from sklearn.metrics import mean_squared_error, mean_absolute_error
from nflows.distributions import ConditionalDiagonalNormal

from .utils import batch, KDE
from ..utils import log_dataframe_artifact

from ...tfboost.tfboost import TreeFlowBoost
from ...independent_multivariate_boosting import IndependentNGBoost


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


def _calculate_rmse_at_k(model: TreeFlowBoost, x: pd.DataFrame, y: pd.DataFrame, num_samples: int, batch_size: int,
                         k: int = 2, find_peaks_parameters: Dict[Any, Any] = None):
    def calculate_treeflow_peaks(samples: np.ndarray, find_peaks_parameters: Dict[Any, Any] = None):
        if find_peaks_parameters is None:
            find_peaks_parameters = {}

        def calculate_peaks(sample, find_peaks_parameters):
            kde = KDE()
            density, support = kde(sample)

            peaks_id, _ = signal.find_peaks(density, **find_peaks_parameters)
            peaks = support[peaks_id]
            peaks_order = np.argsort(-density[peaks_id])  # Sort in descending order.
            return peaks[peaks_order]

        results = []

        for sample in samples:
            peaks = calculate_peaks(sample, find_peaks_parameters)
            results.append(peaks)

        return results

    def rmse_at_k(y_true: np.ndarray, y_score: List[np.ndarray]):
        results = []

        for i in range(y_true.shape[0]):
            results.append(np.min((y_score[i] - y_true[i]) ** 2))  # Take closer prediction

        return np.sqrt(np.average(results))

    if find_peaks_parameters is None:
        find_peaks_parameters = {"height": 0.1}  # Proposed default

    x: np.ndarray = x.values
    y: np.ndarray = y.values

    samples = []
    for i in batch(range(num_samples), 100):
        samples.append(model.sample(x, num_samples=len(i), batch_size=batch_size).squeeze(-1))
    samples = np.concatenate(samples, axis=1)

    y_test_treeflow_peaks = calculate_treeflow_peaks(samples, find_peaks_parameters)
    y_test_treeflow_peaks = [s[:k] for s in y_test_treeflow_peaks]

    return rmse_at_k(y, y_test_treeflow_peaks)


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
    mlflow.log_figure(fig, f'losses-{str(datetime.datetime.now()).replace(" ", "-")}.png')


def calculate_nll_ngboost(model: Union[ngboost.NGBoost, IndependentNGBoost], x: pd.DataFrame, y: pd.DataFrame,
                          independent=False) -> float:
    x: np.ndarray = x.values
    y: np.ndarray = y.values

    if independent:
        y_dists = model.scipy_distribution(x)
    else:
        y_dists = model.pred_dist(x).scipy_distribution()
    nlls = [-y_dists[i].logpdf(y[i, :]) for i in range(y.shape[0])]
    return np.mean(nlls)


def summary(
        train_results_rmse: float,
        train_results_mae: float,
        train_results_nll: float,
        train_results_rmse_tree: float,
        train_results_mae_tree: float,
        train_results_nll_tree: float,
        test_results_rmse: float,
        test_results_mae: float,
        test_results_nll: float,
        test_results_rmse_tree: float,
        test_results_mae_tree: float,
        test_results_nll_tree: float
):
    results = pd.DataFrame([
        ['train', 'rmse', train_results_rmse],
        ['train', 'mae', train_results_mae],
        ['train', 'nll', train_results_nll],
        ['train', 'rmse_tree', train_results_rmse_tree],
        ['train', 'mae_tree', train_results_mae_tree],
        ['train', 'nll_tree', train_results_nll_tree],
        ['test', 'rmse', test_results_rmse],
        ['test', 'mae', test_results_mae],
        ['test', 'nll', test_results_nll],
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


def aggregated_report(*inputs: Tuple[pd.DataFrame]) -> pd.DataFrame:
    df = pd.concat(inputs)
    results = df.groupby(['set', 'metric']).agg([np.mean, np.std])
    results = results.reset_index()
    results.columns = ['set', 'metric', 'mean', 'std']

    # MLFlow
    results["set-metric"] = results["set"] + "_" + results["metric"]
    mlflow.log_metrics({f"{k}_mean": v for k, v in zip(results["set-metric"].values, results["mean"].values)})
    mlflow.log_metrics({f"{k}_std": v for k, v in zip(results["set-metric"].values, results["std"].values)})
    return results
