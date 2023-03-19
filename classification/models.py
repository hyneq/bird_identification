from typing import TypeVar

import numpy as np

from prediction.models import PredictionModelConfig, PathPredictionModelConfig, IImagePredictionModel, PredictionModelFactory, PathPredictionModelFactory, MultiPathPredictionModelFactory

class ClassificationModelConfig(PredictionModelConfig):
    pass

TClassificationModelConfig = TypeVar("TClassificationModelConfig", bound=ClassificationModelConfig)

class PathClassificationModelConfig(ClassificationModelConfig, PathPredictionModelConfig):
    pass

TPathClassificationModelConfig = TypeVar("TPathClassificationModelConfig", bound=PathClassificationModelConfig)

ClassificationModelOutput = np.ndarray

class ClassificationModel(IImagePredictionModel[ClassificationModelConfig, ClassificationModelOutput]):
    __slots__: tuple

TClassificationModel = TypeVar("TClassificationModel", bound=ClassificationModel)

class ClassificationModelFactory(PredictionModelFactory[TClassificationModel, TClassificationModelConfig]):
    pass

class PathClassificationModelFactory(PathPredictionModelFactory[TClassificationModel, TPathClassificationModelConfig]):
    pass

from defaults.classification import MODEL_FACTORIES, DEFAULT_MODEL_FACTORY

model_factory = MultiPathPredictionModelFactory[ClassificationModel, PathClassificationModelConfig](
        factories=MODEL_FACTORIES,
        default_factory=DEFAULT_MODEL_FACTORY
    )

get_classification_model = model_factory.get_model