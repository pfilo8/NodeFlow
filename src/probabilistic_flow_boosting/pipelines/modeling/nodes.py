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
This is a boilerplate pipeline 'modeling'
generated using Kedro 0.17.5
"""

import itertools
import random

from typing import Dict

import numpy as np
import pandas as pd
import torch

from ..utils import log_dataframe_artifact
from ..reporting.nodes import calculate_nll

from ...tfboost.flow import ContinuousNormalizingFlow
from ...tfboost.tree import MODELS
from ...tfboost.tfboost import TreeFlowBoost


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


def train_model(x_train, y_train, flow_p, flow_params, tree_p, tree_params, tree_model_type, n_epochs: int = 100,
                batch_size: int = 1000, random_seed: int = 42):
    """
    Train a TreeFlow model.
    :param x_train: Training data.
    :param y_train: Training labels.
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
    flow = ContinuousNormalizingFlow(conditional=True, **flow_p, **flow_params)
    tree = MODELS[tree_model_type](**tree_params, **tree_p, random_seed=random_seed)

    m = TreeFlowBoost(flow_model=flow, tree_model=tree, embedding_size=flow_p['context_dim'])
    m = m.fit(x_train.values, y_train.values, n_epochs=n_epochs, batch_size=batch_size)
    return m


def modeling(x_train: pd.DataFrame, y_train: pd.DataFrame, tree_model_type, flow_params, tree_params,
             flow_hyperparams, tree_hyperparams, split_size=0.8, n_epochs: int = 100, batch_size: int = 1000,
             random_seed: int = 42):
    setup_random_seed(random_seed)

    x_tr, x_val, y_tr, y_val = split_data(x_train=x_train, y_train=y_train, split_size=split_size)

    flow_hyperparams['hidden_dims'] = map(tuple, flow_hyperparams['hidden_dims'])
    results = []

    for flow_p in generate_params_for_grid_search(flow_hyperparams):
        for tree_p in generate_params_for_grid_search(tree_hyperparams):
            m = train_model(x_tr, y_tr, flow_p, flow_params, tree_p, tree_params, tree_model_type,
                            n_epochs, batch_size, random_seed)

            result_train = calculate_nll(m, x_tr, y_tr, batch_size=batch_size)
            result_val = calculate_nll(m, x_val, y_val, batch_size=batch_size)

            results.append([flow_p, tree_p, result_train, result_val])

    results = pd.DataFrame(results, columns=['flow_p', 'tree_p', 'log_prob_train', 'log_prob_val'])
    results = results.sort_values('log_prob_val', ascending=True)
    log_dataframe_artifact(results, 'grid_search_results')

    best_params = results.iloc[0].to_dict()
    best_flow_p = best_params['flow_p']
    best_tree_p = best_params['tree_p']

    m = train_model(x_train, y_train, best_flow_p, flow_params, best_tree_p, tree_params, tree_model_type,
                    n_epochs, batch_size, random_seed)
    return m
