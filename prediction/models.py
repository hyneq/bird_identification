from typing import Any, Generic, TypeVar, Optional, Union, Callable
from abc import ABC, abstractmethod, abstractclassmethod
from dataclasses import dataclass

from .classes import ClassNames
from .image_utils import Image

TPredictionModelInput = TypeVar("TPredictionModelInput")
TPredictionModelOutput = TypeVar("TPredictionModelOutput")

@dataclass()
class PredictionModelConfig:
    classes_path: str

TPredictionModelConfig = TypeVar("TPredictionModelConfig", bound=PredictionModelConfig)

ModelConfigLoaderInputT = TypeVar("ModelConfigLoaderInputT")

ModelConfigLoaderInputT_cls = TypeVar("ModelConfigLoaderInputT_cls")
ModelConfigLoaderInputT_fun = TypeVar("ModelConfigLoaderInputT_fun")

ModelConfigLoader = Callable[[ModelConfigLoaderInputT], TPredictionModelConfig]

def default_model_config_loader(input: TPredictionModelConfig) -> TPredictionModelConfig:
    return input

class IPredictionModel(ABC, Generic[TPredictionModelConfig, TPredictionModelInput, TPredictionModelOutput]):
    __slots__: tuple

    class_names: ClassNames

    @abstractmethod
    def __init__(self, cfg: TPredictionModelConfig):
        pass

    @abstractmethod
    def predict(self, input: TPredictionModelInput) -> TPredictionModelOutput:
        pass

TPredictionModel = TypeVar("TPredictionModel", bound=IPredictionModel)

class PredictionModel(IPredictionModel[TPredictionModelConfig, TPredictionModelInput, TPredictionModelOutput]):
    __slots__: tuple

    def __init__(self, cfg: TPredictionModelConfig):
        self.class_names = self.load_classes(cfg.classes_path)

    @staticmethod
    def load_classes(classes_path: str):
        return ClassNames.load_from_file(classes_path)

class IImagePredictionModel(IPredictionModel[TPredictionModelConfig, Image, TPredictionModelOutput]):
    pass

class ImagePredictionModel(PredictionModel[TPredictionModelConfig, Image, TPredictionModelOutput], IImagePredictionModel[TPredictionModelConfig, TPredictionModelOutput]):
    pass

class IPredictionModelFactory(ABC, Generic[TPredictionModel, ModelConfigLoaderInputT_cls, TPredictionModelConfig]):
    name: str

    @abstractmethod
    def get_model(self, cfg_input: Optional[Union[ModelConfigLoaderInputT_cls, ModelConfigLoaderInputT_fun]]=None, loader: Optional[ModelConfigLoader[ModelConfigLoaderInputT_fun, TPredictionModelConfig]]=None) -> TPredictionModel:
        pass

@dataclass(frozen=True)
class PredictionModelFactory(IPredictionModelFactory[TPredictionModel, ModelConfigLoaderInputT_cls, TPredictionModelConfig]):
    name: str
    model_cls: type[TPredictionModel]
    model_config_cls: type[TPredictionModelConfig]
    model_config_loader: ModelConfigLoader[Any, TPredictionModelConfig] = default_model_config_loader
    default_model_config_input: Optional[Any] = None

    def get_model_cfg(self, cfg_input: Optional[Union[ModelConfigLoaderInputT_cls, ModelConfigLoaderInputT_fun]]=None, cfg_loader: Optional[ModelConfigLoader[ModelConfigLoaderInputT_fun, TPredictionModelConfig]]=None) -> TPredictionModelConfig:
        if cfg_input:
            if not cfg_loader:
                cfg_loader = self.model_config_loader
            
            cfg = cfg_loader(cfg_input)
        else:
            cfg = self.model_config_loader(self.default_model_config_input)
        
        return cfg
    
    def get_model(self, *args, cfg: Optional[TPredictionModelConfig]=None, **kwargs) -> TPredictionModel:
        if not cfg:
            cfg = self.get_model_cfg(*args, **kwargs)
        
        return self.model_cls(cfg)

class MultiPredictionModelFactory(IPredictionModelFactory[TPredictionModel, ModelConfigLoaderInputT_cls, TPredictionModelConfig]):
    name: str

    factories: dict[str, IPredictionModelFactory[TPredictionModel, ModelConfigLoaderInputT_cls, TPredictionModelConfig]]
    default_factory: str

    def __init__(self,
                factories: Union[list[IPredictionModelFactory[TPredictionModel, ModelConfigLoaderInputT_cls, TPredictionModelConfig]],dict[str,IPredictionModelFactory[TPredictionModel, ModelConfigLoaderInputT_cls, TPredictionModelConfig]]],
                default_factory: str,
                name="multi"
        ):

        if isinstance(factories, list):
            factories = {f.name: f for f in factories}

        self.name = name
        self.factories = factories
        self.default_factory = default_factory
    
    def get_model(self, *args, factory: Optional[str]=None, **kwargs) -> TPredictionModel:
        if not factory:
            factory = self.default_factory
        
        return self.factories[factory].get_model(*args, **kwargs)
    
    def get_factory_names(self):
        return list(self.factories.keys())

MultiPathPredictionModelFactory = MultiPredictionModelFactory[TPredictionModel, str, TPredictionModelConfig]