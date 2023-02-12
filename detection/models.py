from abc import ABC, abstractmethod
from typing import TypeVar, Generic
from collections.abc import Sequence

import numpy as np

from prediction.models import PredictionModelConfig, AImagePredictionModel

class DetectionModelConfig(PredictionModelConfig):
    pass

TDetectionObj = TypeVar("TDetectionObj")

TDetectionModelRawOutput = TypeVar("TDetectionModelRawOutput")

class DetectionModelOutputIter(ABC, Generic[TDetectionObj, TDetectionModelRawOutput]):

    raw_output: TDetectionModelRawOutput

    def __init__(self, raw_output: TDetectionModelRawOutput):
        self.raw_output = raw_output

    @abstractmethod
    def __next__(self) -> TDetectionObj:
        pass

TDetectionModelOutputIter = TypeVar("TDetectionModelOutputIter", bound=DetectionModelOutputIter)

class DetectionModelOutput(ABC, Generic[TDetectionObj, TDetectionModelRawOutput, TDetectionModelOutputIter]):

    raw_output: TDetectionModelRawOutput

    iter_cls: type[TDetectionModelOutputIter]

    def __init__(self, raw_output: TDetectionModelRawOutput):
        self.raw_output = raw_output

    @abstractmethod
    def get_box(self, obj: TDetectionObj) -> Sequence:
        pass
    
    @abstractmethod
    def get_scores(self, obj: TDetectionObj) -> np.ndarray:
        pass
    
    def __iter__(self) -> TDetectionModelOutputIter:
        return self.iter_cls(self.raw_output)

class DetectionModel(AImagePredictionModel[DetectionModelConfig, DetectionModelOutput]):
    pass