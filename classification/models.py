from typing import TypeVar

import numpy as np

from prediction.models import PredictionModelConfig, IImagePredictionModel, PredictionModelFactory, get_prediction_model_factory

class ClassificationModelConfig(PredictionModelConfig):
    pass

TClassificationModelConfig = TypeVar("TClassificationModelConfig", bound=ClassificationModelConfig)

ClassificationModelOutput = np.ndarray

class ClassificationModel(IImagePredictionModel[ClassificationModelConfig, ClassificationModelOutput]):
    __slots__: tuple

TClassificationModel = TypeVar("TClassificationModel", bound=ClassificationModel)

class ClassificationModelFactory(PredictionModelFactory[TClassificationModel, TClassificationModelConfig]):
    pass

from defaults.classification import DEFAULT_MODEL_CLS, DEFAULT_MODEL_CONFIG

get_classification_model = get_prediction_model_factory(
    name="get_classification_model",
    model_cls=ClassificationModel,
    model_config_cls=ClassificationModelConfig,
    model_type_cls=ClassificationModelFactory,
    DEFAULT_MODEL_CLS=DEFAULT_MODEL_CLS,
    DEFAULT_MODEL_CONFIG=DEFAULT_MODEL_CONFIG
)