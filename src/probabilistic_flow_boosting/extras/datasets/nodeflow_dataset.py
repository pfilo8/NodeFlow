from typing import Any, Dict
import joblib
from pathlib import Path
import pickle
import gcsfs
from google.oauth2 import service_account

import optuna
from kedro.io import AbstractDataSet
from probabilistic_flow_boosting.models.cnf.cnf import ContinuousNormalizingFlowRegressor
from probabilistic_flow_boosting.models.nodeflow import NodeFlow
from probabilistic_flow_boosting.models.node_gmm import NodeGMM




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
    
class NodeGMMDataSet(AbstractDataSet):

    def __init__(self, filepath):
        self._filepath = Path(filepath)
        self._filepath.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> NodeFlow:
        return NodeGMM.load(self._filepath)

    def _save(self, model: NodeFlow) -> None:
        return model.save(self._filepath)

    def _describe(self) -> Dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset"""
        return dict(
            filepath=self._filepath
        )
    
class OptunaStudyDataSet(AbstractDataSet):
    def __init__(self, filepath, fs_args, credentials):
        self._filepath = filepath
        credentials = service_account.Credentials.from_service_account_file(credentials["token"], scopes=["https://www.googleapis.com/auth/cloud-platform"])
        self.fs = gcsfs.GCSFileSystem(token=credentials, **fs_args)

    def _load(self) -> optuna.Study:
        with self.fs.open(f"{self._filepath}", "rb") as f:
            pickle_data = f.read()
            loaded_data = pickle.loads(pickle_data)
        return loaded_data

    def _save(self, study: optuna.Study) -> None:
        pickle_data = pickle.dumps(study) # "study.pkl"
        with self.fs.open(f"{self._filepath}", "wb") as f:
            f.write(pickle_data)


    def _describe(self) -> Dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset"""
        return dict(
            filepath=self._filepath
        )