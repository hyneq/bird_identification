from typing import Generic, TypeVar
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from classes import ClassNames

Image = np.ndarray

TPredictionModelInput = TypeVar("TPredictionModelInput")
TPredictionModelOutput = TypeVar("TPredictionModelOutput")

@dataclass()
class PredictionModelConfig:
    classes_path: str

TPredictionModelConfig = TypeVar("TPredictionModelConfig", bound=PredictionModelConfig)

class APredictionModel(ABC, Generic[TPredictionModelConfig, TPredictionModelInput, TPredictionModelOutput]):
    __slots__: tuple

    @abstractmethod
    def __init__(self, cfg: TPredictionModelConfig):
        pass

    @abstractmethod
    def predict(self, input: TPredictionModelInput) -> TPredictionModelOutput:
        pass

TPredictionModel = TypeVar("TPredictionModel", bound=APredictionModel)

class PredictionModelConfigWithCls(Generic[TPredictionModel]):
    model_cls: type[TPredictionModel]

class PredictionModel(APredictionModel[TPredictionModelConfig, TPredictionModelInput, TPredictionModelOutput]):
    __slots__: tuple

    class_names: ClassNames

    def __init__(self, cfg: TPredictionModelConfig):
        self.class_names = self.load_classes(cfg.classes_path)

    @staticmethod
    def load_classes(classes_path: str):
        return ClassNames.load_from_file(classes_path)

class AImagePredictionModel(APredictionModel[TPredictionModelConfig, Image, TPredictionModelOutput]):
    pass

class ImagePredictionModel(PredictionModel[TPredictionModelConfig, Image, TPredictionModelOutput], AImagePredictionModel):
    pass