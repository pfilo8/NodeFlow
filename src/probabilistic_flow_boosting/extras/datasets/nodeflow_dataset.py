from typing import Any, Dict

from kedro.io import AbstractDataSet
from ...nodeflow.nodeflow import NodeFlow
from ...cnf.cnf import ContinuousNormalizingFlowRegressor


class CNFDataSet(AbstractDataSet):

    def __init__(self, filepath):
        self._filepath = filepath

    def _load(self) -> NodeFlow:
        return ContinuousNormalizingFlowRegressor.load(self._filepath)

    def _save(self, model: NodeFlow) -> None:
        return model.save(self._filepath)

    def _describe(self) -> Dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset"""
        return dict(
            filepath=self._filepath
        )


class NodeFlowDataSet(AbstractDataSet):

    def __init__(self, filepath):
        self._filepath = filepath

    def _load(self) -> NodeFlow:
        return NodeFlow.load(self._filepath)

    def _save(self, model: NodeFlow) -> None:
        return model.save(self._filepath)

    def _describe(self) -> Dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset"""
        return dict(
            filepath=self._filepath
        )

class OptunaDbDataSet(AbstractDataSet):
    def __init__(self, filepath):
        self._filepath = filepath

    def _load(self) -> NodeFlow:
        return self._filepath

    def _save(self) -> None:
        return self._filepath
    
    def _describe(self) -> Dict[str, Any]:
        """Returns a filepath to optuna experiment db"""
        return dict(
            filepath=self._filepath
        )