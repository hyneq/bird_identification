from abc import ABC, abstractmethod
from typing import TypeVar, Generic
from collections.abc import Sequence

import numpy as np

from prediction.models import PredictionModelConfig, PathPredictionModelConfig, IImagePredictionModel, PredictionModelFactory, get_prediction_model_factory

class DetectionModelConfig(PredictionModelConfig):
    pass

TDetectionModelConfig = TypeVar("TDetectionModelConfig", bound=DetectionModelConfig)

class PathDetectionModelConfig(PathPredictionModelConfig):
    pass

TPathDetectionModelConfig = TypeVar("TPathDetectionModelConfig", bound=PathDetectionModelConfig)

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

class DetectionModel(IImagePredictionModel[DetectionModelConfig, DetectionModelOutput]):
    pass

TDetectionModel = TypeVar("TDetectionModel", bound=DetectionModel)

class DetectionModelFactory(PredictionModelFactory[TDetectionModel, TDetectionModelConfig]):
    pass

from defaults.detection import DEFAULT_MODEL_CLS, DEFAULT_MODEL_CONFIG

get_detection_model = get_prediction_model_factory(
    name="get_classification_model",
    model_cls=DetectionModel,
    model_config_cls=DetectionModelConfig,
    model_type_cls=DetectionModelFactory,
    DEFAULT_MODEL_CLS=DEFAULT_MODEL_CLS,
    DEFAULT_MODEL_CONFIG=DEFAULT_MODEL_CONFIG
)