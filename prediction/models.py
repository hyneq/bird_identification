from typing import Generic, TypeVar, Optional
from abc import ABC, abstractmethod, abstractclassmethod
from dataclasses import dataclass

import numpy as np

from .classes import ClassNames

Image = np.ndarray

TPredictionModelInput = TypeVar("TPredictionModelInput")
TPredictionModelOutput = TypeVar("TPredictionModelOutput")

@dataclass()
class PredictionModelConfig:
    classes_path: str

TPredictionModelConfig = TypeVar("TPredictionModelConfig", bound=PredictionModelConfig)

class PathPredictionModelConfig(PredictionModelConfig, ABC):
    
    @abstractclassmethod
    def from_path(cls):
        pass

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

TPredictionModelConfigWithCls = TypeVar("TPredictionModelConfigWithCls", bound=PredictionModelConfigWithCls)

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

class ImagePredictionModel(PredictionModel[TPredictionModelConfig, Image, TPredictionModelOutput], AImagePredictionModel[TPredictionModelConfig, TPredictionModelOutput]):
    pass

@dataclass
class PredictionModelType(Generic[TPredictionModel, TPredictionModelConfig]):
    name: str
    model_cls: type[TPredictionModel]
    model_config_cls: type[TPredictionModelConfig]

def get_prediction_model_factory(
        name: str,
        model_cls: type[TPredictionModel],
        model_config_cls: type[TPredictionModelConfig],
        model_type_cls: type[PredictionModelType[TPredictionModel, TPredictionModelConfig]],
        DEFAULT_MODEL_CLS: type[TPredictionModel],
        DEFAULT_MODEL_CONFIG: TPredictionModelConfig
    ):

    def get_prediction_model(
            model_config: Optional[model_config_cls]=None,
            model_cls: Optional[type[model_cls]]=None,
            model_type: Optional[model_type_cls]=None
        ):

        if not model_config:
            model_config = DEFAULT_MODEL_CONFIG

        if not model_cls:
            if hasattr(model_config, "model_cls"):
                model_cls = model_config.model_cls
            elif model_type:
                model_cls = model_type.model_cls
            else:
                model_cls = DEFAULT_MODEL_CLS

        return model_cls(model_config)
    
    get_prediction_model.__name__ = name
    
    return get_prediction_model