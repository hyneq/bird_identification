from typing import TypeVar

import numpy as np

from prediction.image_utils import Image
from prediction.models import ModelConfigLoaderInputT_cls, PredictionModelConfig, ModelConfigLoaderInputT, IImagePredictionModel, PredictionModelFactory, MultiPathPredictionModelFactory

class ClassificationModelConfig(PredictionModelConfig):
    pass

ClassificationModelConfigT = TypeVar("ClassificationModelConfigT", bound=ClassificationModelConfig)

ClassificationModelOutput = np.ndarray

ClassificationModel = IImagePredictionModel[ClassificationModelConfig, ClassificationModelOutput]

ClassificationModelFactory = PredictionModelFactory[ModelConfigLoaderInputT_cls, ClassificationModelConfigT, Image, ClassificationModelOutput]

from defaults.classification import MODEL_FACTORIES, DEFAULT_MODEL_FACTORY

classification_model_factory = MultiPathPredictionModelFactory[ClassificationModelConfig, Image, ClassificationModelOutput](
        factories=MODEL_FACTORIES,
        default_factory=DEFAULT_MODEL_FACTORY
    )

get_classification_model = classification_model_factory.get_model