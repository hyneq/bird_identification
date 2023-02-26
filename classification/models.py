import numpy as np

from prediction.models import PredictionModelConfig, AImagePredictionModel, get_prediction_model_factory

class ClassificationModelConfig(PredictionModelConfig):
    pass

ClassificationModelOutput = np.ndarray

class ClassificationModel(AImagePredictionModel[ClassificationModelConfig, ClassificationModelOutput]):
    __slots__: tuple

from defaults.classification import DEFAULT_MODEL_CLS, DEFAULT_MODEL_CONFIG

get_classification_model = get_prediction_model_factory(
    name="get_classification_model",
    model_cls=ClassificationModel,
    model_config_cls=ClassificationModelConfig,
    DEFAULT_MODEL_CLS=DEFAULT_MODEL_CLS,
    DEFAULT_MODEL_CONFIG=DEFAULT_MODEL_CONFIG
)