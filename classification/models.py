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

ClassificationModel = IImagePredictionModel[ClassificationModelConfig, ClassificationModelOutput]

TClassificationModel = TypeVar("TClassificationModel", bound=ClassificationModel)

ClassificationModelFactory = PredictionModelFactory[TClassificationModel, TClassificationModelConfig]

PathClassificationModelFactory = PathPredictionModelFactory[TClassificationModel, TPathClassificationModelConfig]

from defaults.classification import MODEL_FACTORIES, DEFAULT_MODEL_FACTORY

classification_model_factory = MultiPathPredictionModelFactory[ClassificationModel, PathClassificationModelConfig](
        factories=MODEL_FACTORIES,
        default_factory=DEFAULT_MODEL_FACTORY
    )

get_classification_model = classification_model_factory.get_model