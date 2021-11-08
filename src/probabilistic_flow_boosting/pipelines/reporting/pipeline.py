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

from kedro.pipeline import Pipeline, pipeline, node

from .nodes import calculate_rmse, calculate_mae, calculate_nll


def create_pipeline_report_train():
    return pipeline(
        create_pipeline_calculate_metrics(),
        inputs={
            "x": "x_train",
            "y": "y_train"
        },
        outputs={
            "results_rmse": "train_results_rmse",
            "results_mae": "train_results_mae",
            "results_nll": "train_results_nll"
        }
    )


def create_pipeline_report_test():
    return pipeline(
        create_pipeline_calculate_metrics(),
        inputs={
            "x": "x_test",
            "y": "y_test"
        },
        outputs={
            "results_rmse": "test_results_rmse",
            "results_mae": "test_results_mae",
            "results_nll": "test_results_nll"
        }
    )


def create_pipeline_calculate_metrics(**kwargs):
    return Pipeline([
        node(
            func=calculate_rmse,
            inputs=["model", "x", "y"],
            outputs=["results_rmse"]
        ),
        node(
            func=calculate_mae,
            inputs=["model", "x", "y"],
            outputs=["results_mae"]
        ),
        node(
            func=calculate_nll,
            inputs=["model", "x", "y"],
            outputs=["results_nll"]
        )
    ])
