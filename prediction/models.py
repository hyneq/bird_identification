from typing import Generic, TypeVar, Optional, Union
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

TPathPredictionModelConfig = TypeVar("TPathPredictionModelConfig", bound=PathPredictionModelConfig)

class IPredictionModel(ABC, Generic[TPredictionModelConfig, TPredictionModelInput, TPredictionModelOutput]):
    __slots__: tuple

    @abstractmethod
    def __init__(self, cfg: TPredictionModelConfig):
        pass

    @abstractmethod
    def predict(self, input: TPredictionModelInput) -> TPredictionModelOutput:
        pass

TPredictionModel = TypeVar("TPredictionModel", bound=IPredictionModel)

class PredictionModelConfigWithCls(Generic[TPredictionModel]):
    model_cls: type[TPredictionModel]

TPredictionModelConfigWithCls = TypeVar("TPredictionModelConfigWithCls", bound=PredictionModelConfigWithCls)

class PredictionModel(IPredictionModel[TPredictionModelConfig, TPredictionModelInput, TPredictionModelOutput]):
    __slots__: tuple

    class_names: ClassNames

    def __init__(self, cfg: TPredictionModelConfig):
        self.class_names = self.load_classes(cfg.classes_path)

    @staticmethod
    def load_classes(classes_path: str):
        return ClassNames.load_from_file(classes_path)

class IImagePredictionModel(IPredictionModel[TPredictionModelConfig, Image, TPredictionModelOutput]):
    pass

class ImagePredictionModel(PredictionModel[TPredictionModelConfig, Image, TPredictionModelOutput], IImagePredictionModel[TPredictionModelConfig, TPredictionModelOutput]):
    pass

class IPredictionModelFactory(ABC, Generic[TPredictionModel, TPredictionModelConfig]):
    name: str

    @abstractmethod
    def get_model(self, *args, **kwargs):
        pass

class IPathPredictionModelFactory(IPredictionModelFactory[TPredictionModel, TPathPredictionModelConfig]):

    @abstractmethod
    def get_model(self, *args, path: Optional[str]=None, **kwargs):
        pass

@dataclass
class PredictionModelFactory(IPredictionModelFactory[TPredictionModel, TPredictionModelConfig]):
    name: str
    model_cls: type[TPredictionModel]
    model_config_cls: type[TPredictionModelConfig]

    def get_model(self, cfg: Optional[TPredictionModelConfig]=None):
        if not cfg:
            raise RuntimeError("No config supplied, cannot instantiate model")
        
        return self.model_cls(cfg)

class PathPredictionModelFactory(PredictionModelFactory[TPredictionModel, TPathPredictionModelConfig], IPathPredictionModelFactory[TPredictionModel, TPathPredictionModelConfig]):

    def get_model(self, path: Optional[str]=None, cfg: Optional[TPredictionModelConfig]=None):
        if not cfg and path:
            cfg = self.model_config_cls.from_path(path)
        
        return super().get_model(cfg=cfg)

class MultiPredictionModelFactory(IPredictionModelFactory[TPredictionModel, TPredictionModelConfig]):
    name: str

    factories: dict[str, IPredictionModelFactory[TPredictionModel, TPredictionModelConfig]]
    default_factory: str

    def __init__(self,
                factories: Union[list[IPredictionModelFactory[TPredictionModel, TPredictionModelConfig]],dict[str,IPredictionModelFactory[TPredictionModel, TPredictionModelConfig]]],
                default_factory: str,
                name="multi"
        ):

        if isinstance(factories, list):
            factories = {f.name: f for f in factories}

        self.name = name
        self.factories = factories
        self.default_factory = default_factory
    
    def get_model(self, *args, factory: Optional[str]=None, **kwargs):
        if not factory:
            factory = self.default_factory
        
        return self.factories[factory].get_model(*args, **kwargs)

def get_prediction_model_factory(
        name: str,
        model_cls: type[TPredictionModel],
        model_config_cls: type[TPredictionModelConfig],
        model_type_cls: type[PredictionModelFactory[TPredictionModel, TPredictionModelConfig]],
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