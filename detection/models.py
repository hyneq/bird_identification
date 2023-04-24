from abc import ABC, abstractmethod
from typing import TypeVar, Generic
from collections.abc import Sequence

import numpy as np

from image_utils import Image
from prediction.models import PredictionModelConfig, IPredictionModel, ModelConfigLoaderInputT_cls, PredictionModelFactory, MultiPathPredictionModelFactory

class DetectionModelConfig(PredictionModelConfig):
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

DetectionModelOutputIterT = TypeVar("DetectionModelOutputIterT", bound=DetectionModelOutputIter)

class DetectionModelOutput(ABC, Generic[DetectionObjT, DetectionModelRawOutputT, DetectionModelOutputIterT]):

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

DetectionModel = IPredictionModel[DetectionModelConfig, Image, DetectionModelOutput]

DetectionModelFactory = PredictionModelFactory[ModelConfigLoaderInputT_cls, DetectionModelConfigT, Image, DetectionModelOutput]

from defaults.detection import MODEL_FACTORIES, DEFAULT_MODEL_FACTORY

model_factory = MultiPathPredictionModelFactory[DetectionModelConfig, Image, DetectionModelOutput](
        factories=MODEL_FACTORIES,
        default_factory=DEFAULT_MODEL_FACTORY
    )

get_detection_model = model_factory.get_model