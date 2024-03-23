from abc import ABC, abstractmethod
from typing import TypeVar, Generic
from collections.abc import Sequence

import numpy as np

from ..image_utils import Image
from ..prediction.models import (
    PredictionModelWithClassesConfig,
    IPredictionModelWithClasses,
    ModelConfigLoaderInputT_cls,
    PredictionModelFactory,
    MultiPathPredictionModelFactory,
)

from ..factories import search_factories

class DetectionModelConfig(PredictionModelWithClassesConfig):
    pass


DetectionModelConfigT = TypeVar("DetectionModelConfigT", bound=DetectionModelConfig)

DetectionObjT = TypeVar("DetectionObjT")

DetectionModelRawOutputT = TypeVar("DetectionModelRawOutputT")


class DetectionModelOutputIter(ABC, Generic[DetectionObjT, DetectionModelRawOutputT]):
    raw_output: DetectionModelRawOutputT

    def __init__(self, raw_output: DetectionModelRawOutputT):
        self.raw_output = raw_output

    @abstractmethod
    def __next__(self) -> DetectionObjT:
        pass


DetectionModelOutputIterT = TypeVar(
    "DetectionModelOutputIterT", bound=DetectionModelOutputIter
)


class DetectionModelOutput(
    ABC, Generic[DetectionObjT, DetectionModelRawOutputT, DetectionModelOutputIterT]
):
    raw_output: DetectionModelRawOutputT

    iter_cls: type[DetectionModelOutputIterT]

    def __init__(self, raw_output: DetectionModelRawOutputT):
        self.raw_output = raw_output

    @abstractmethod
    def get_box(self, obj: DetectionObjT) -> Sequence:
        pass

    @abstractmethod
    def get_scores(self, obj: DetectionObjT) -> np.ndarray:
        pass

    def __iter__(self) -> DetectionModelOutputIterT:
        return self.iter_cls(self.raw_output)


class DetectionModel(
    IPredictionModelWithClasses[DetectionModelConfig, Image, DetectionModelOutput]
):
    external_NMS: bool = True


DetectionModelFactory = PredictionModelFactory[
    ModelConfigLoaderInputT_cls, DetectionModelConfigT, Image, DetectionModelOutput
]

from ..defaults.detection import DEFAULT_MODEL_FACTORY

model_factory = MultiPathPredictionModelFactory[
    DetectionModelConfig, Image, DetectionModelOutput
](factories=search_factories(prefix='detection_models_'), default_factory=DEFAULT_MODEL_FACTORY)

get_detection_model = model_factory
