from typing import TypeVar

import numpy as np

from ..prediction.classes import Scores
from ..image_utils import Image
from ..prediction.models import (
    ModelConfigLoaderInputT_cls,
    PredictionModelWithClassesConfig,
    IPredictionModelWithClasses,
    PredictionModelFactory,
    MultiPathPredictionModelFactory,
)

from ..factories import search_factories

class ClassificationModelConfig(PredictionModelWithClassesConfig):
    pass


ClassificationModelConfigT = TypeVar(
    "ClassificationModelConfigT", bound=ClassificationModelConfig
)

ClassificationModelOutput = Scores

ClassificationModel = IPredictionModelWithClasses[
    ClassificationModelConfig, Image, ClassificationModelOutput
]

ClassificationModelFactory = PredictionModelFactory[
    ModelConfigLoaderInputT_cls,
    ClassificationModelConfigT,
    Image,
    ClassificationModelOutput,
]

from ..defaults.classification import DEFAULT_MODEL_FACTORY

classification_model_factory = MultiPathPredictionModelFactory[
    ClassificationModelConfig, Image, ClassificationModelOutput
](factories=search_factories(prefix='classification_models_'), default_factory=DEFAULT_MODEL_FACTORY)

get_classification_model = classification_model_factory
