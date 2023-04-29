from typing import Any, Dict

from kedro.io import AbstractDataSet
from ...nodeflow.nodeflow import NodeFlow


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