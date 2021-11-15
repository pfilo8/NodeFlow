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
from typing import Tuple

import mlflow
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error

from ...tfboost.tfboost import TreeFlowBoost


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


def summary(
        train_results_rmse: float,
        train_results_mae: float,
        train_results_nll: float,
        test_results_rmse: float,
        test_results_mae: float,
        test_results_nll: float
):
    return pd.DataFrame([
        ['train', 'rmse', train_results_rmse],
        ['train', 'mae', train_results_mae],
        ['train', 'nll', train_results_nll],
        ['test', 'rmse', test_results_rmse],
        ['test', 'mae', test_results_mae],
        ['test', 'nll', test_results_nll],
    ],
        columns=[
            'set', 'metric', 'value'
        ]
    )


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
