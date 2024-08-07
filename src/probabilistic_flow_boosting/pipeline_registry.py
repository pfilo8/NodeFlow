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

"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from .pipelines import (
    create_general_pipeline_treeflow,
    create_general_pipeline_ngboost,
    create_pipeline_aggregated_report,
    create_general_pipeline_nodeflow,
    create_general_pipeline_cnf,
    create_general_pipeline_nodegmm,
) 


def create_general_uci_pipeline(namespace, n):
    return Pipeline([
        *[create_general_pipeline_treeflow(f"{namespace}_{i}") for i in range(n)],
        create_pipeline_aggregated_report(
            inputs=[f"{namespace}_{i}.summary" for i in range(n)],
            outputs=f"{namespace}.aggregated_summary"
        )
    ])


def create_general_momogp_pipeline(namespace):
    return Pipeline([
        create_general_pipeline_treeflow(namespace),
        create_pipeline_aggregated_report(
            inputs=[f"{namespace}.summary"],
            outputs=f"{namespace}.aggregated_summary"
        )
    ])

def create_general_momogp_nodeflow_pipeline(namespace):
    return Pipeline([
        create_general_pipeline_nodeflow(namespace),
        create_pipeline_aggregated_report(
            inputs=[f"{namespace}.summary"],
            outputs=f"{namespace}.aggregated_summary"
        )
    ])

def create_general_momogp_ngboost_pipeline(namespace):
    return Pipeline([
        create_general_pipeline_ngboost(namespace),
        create_pipeline_aggregated_report(
            inputs=[f"{namespace}.summary"],
            outputs=f"{namespace}.aggregated_summary"
        )
    ])

def create_general_uci_nodeflow_pipeline(namespace, n):
    return Pipeline([
        *[create_general_pipeline_nodeflow(f"{namespace}_{i}") for i in range(n)],
        create_pipeline_aggregated_report(
            inputs=[f"{namespace}_{i}.summary" for i in range(n)],
            outputs=f"{namespace}.aggregated_summary"
        )
    ])

def create_general_uci_cnf_pipeline(namespace, n):
    return Pipeline([
        *[create_general_pipeline_cnf(f"{namespace}_{i}") for i in range(n)],
        create_pipeline_aggregated_report(
            inputs=[f"{namespace}_{i}.summary" for i in range(n)],
            outputs=f"{namespace}.aggregated_summary"
        )
    ])

def create_general_uci_nodegmm_pipeline(namespace, n):
    return Pipeline([
        *[create_general_pipeline_nodegmm(f"{namespace}_{i}") for i in range(n)],
        create_pipeline_aggregated_report(
            inputs=[f"{namespace}_{i}.summary" for i in range(n)],
            outputs=f"{namespace}.aggregated_summary"
        )
    ])

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    momogp_datasets = [
        "momogp_energy",
        "momogp_parkinsons",
        "momogp_scm20d",
        "momogp_usflight",
        "momogp_wind"
    ]

    momogp_pipelines = {
        d: create_general_momogp_pipeline(d) for d in momogp_datasets
    }

    momogp_pipelines_nodeflow = {
        f"{d}_nodeflow": create_general_momogp_nodeflow_pipeline(f"{d}_nodeflow") for d in momogp_datasets
    }

    momogp_ngboost_pipelines = {
        f"{d}_ngboost": create_general_momogp_ngboost_pipeline(f"{d}_ngboost") for d in momogp_datasets
    }

    oceanographic_pipelines = {
        "oceanographic": create_general_uci_pipeline("oceanographic", 20)
    }

    oceanographic_pipelines = {
        "oceanographic_nodeflow": create_general_uci_nodeflow_pipeline("oceanographic", 20)
    }

    uci_datasets = [
        ("uci_boston", 20),
        ("uci_concrete", 20),
        ("uci_energy", 20),
        ("uci_kin8nm", 20),
        ("uci_naval_propulsion_plant", 20),
        ("uci_power_plant", 20),
        ("uci_protein_tertiary_structure", 5),
        ("uci_wine_quality_red", 20),
        ("uci_yacht", 20),
        ("uci_year_prediction_msd", 1)
    ]

    uci_pipelines = {
        d: create_general_uci_pipeline(d, n) for d, n in uci_datasets
    }

    uci_nodeflow_pipelines = {
        f"{d}_nodeflow": create_general_uci_nodeflow_pipeline(f"{d}_nodeflow", n) for d, n in uci_datasets
    }

    uci_cnf_pipelines = {
        f"{d}_cnf": create_general_uci_cnf_pipeline(f"{d}_cnf", n) for d, n in uci_datasets
    }

    uci_nodegmm_pipelines = {
        f"{d}_nodegmm": create_general_uci_nodegmm_pipeline(f"{d}_nodegmm", n) for d, n in uci_datasets
    }

    return {
        "__default__": Pipeline([]),
        **momogp_pipelines,
        **momogp_ngboost_pipelines,
        **uci_pipelines,
        **uci_nodeflow_pipelines,
        **oceanographic_pipelines,
        **uci_cnf_pipelines,
        **uci_nodegmm_pipelines,
        **momogp_pipelines_nodeflow,
    }
