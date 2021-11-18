from .ecatboost import EmbeddableCatBoostPriorNormal, EmbeddableCatBoostPriorPredicted, EmbeddableCatBoostPriorAveraged
from .engboost import EmbeddableNGBoost

MODELS = {
    "CatBoostPriorNormal": EmbeddableCatBoostPriorNormal,
    "CatBoostPriorPredicted": EmbeddableCatBoostPriorPredicted,
    "CatBoostPriorAveraged": EmbeddableCatBoostPriorAveraged
}
