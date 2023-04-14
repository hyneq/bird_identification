from typing import TypeVar

import numpy as np

from prediction.models import PredictionModelConfig, ModelConfigLoaderInputT, IImagePredictionModel, PredictionModelFactory, MultiPathPredictionModelFactory

class ClassificationModelConfig(PredictionModelConfig):
    pass

ClassificationModelConfigT = TypeVar("ClassificationModelConfigT", bound=ClassificationModelConfig)

ClassificationModelOutput = np.ndarray

ClassificationModel = IImagePredictionModel[ClassificationModelConfig, ClassificationModelOutput]

ClassificationModelFactory = PredictionModelFactory[ClassificationModel, ModelConfigLoaderInputT, ClassificationModelConfigT]

from defaults.classification import MODEL_FACTORIES, DEFAULT_MODEL_FACTORY

classification_model_factory = MultiPathPredictionModelFactory[ClassificationModel, ClassificationModelConfig](
        factories=MODEL_FACTORIES,
        default_factory=DEFAULT_MODEL_FACTORY
    )

get_classification_model = classification_model_factory.get_model