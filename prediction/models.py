from typing import Any, Generic, TypeVar, Optional, Union, Callable
from abc import ABC, abstractmethod, abstractclassmethod
from dataclasses import dataclass

from .classes import ClassNames
from .image_utils import Image

PredictionModelInputT = TypeVar("PredictionModelInputT")
PredictionModelOutputT = TypeVar("PredictionModelOutputT")

@dataclass()
class PredictionModelConfig:
    classes_path: str

PredictionModelConfigT = TypeVar("PredictionModelConfigT", bound=PredictionModelConfig)

ModelConfigLoaderInputT = TypeVar("ModelConfigLoaderInputT")

ModelConfigLoaderInputT_cls = TypeVar("ModelConfigLoaderInputT_cls")
ModelConfigLoaderInputT_fun = TypeVar("ModelConfigLoaderInputT_fun")

ModelConfigLoader = Callable[[ModelConfigLoaderInputT], PredictionModelConfigT]

def default_model_config_loader(input: PredictionModelConfigT) -> PredictionModelConfigT:
    return input

class IPredictionModel(ABC, Generic[PredictionModelConfigT, PredictionModelInputT, PredictionModelOutputT]):
    __slots__: tuple

    class_names: ClassNames

    @abstractmethod
    def __init__(self, cfg: PredictionModelConfigT):
        pass

    @abstractmethod
    def predict(self, input: PredictionModelInputT) -> PredictionModelOutputT:
        pass

PredictionModelT = TypeVar("PredictionModelT", bound=IPredictionModel)

class PredictionModel(IPredictionModel[PredictionModelConfigT, PredictionModelInputT, PredictionModelOutputT]):
    __slots__: tuple

    def __init__(self, cfg: PredictionModelConfigT):
        self.class_names = self.load_classes(cfg.classes_path)

    @staticmethod
    def load_classes(classes_path: str):
        return ClassNames.load_from_file(classes_path)

class IImagePredictionModel(IPredictionModel[PredictionModelConfigT, Image, PredictionModelOutputT]):
    pass

class ImagePredictionModel(PredictionModel[PredictionModelConfigT, Image, PredictionModelOutputT], IImagePredictionModel[PredictionModelConfigT, PredictionModelOutputT]):
    pass

class IPredictionModelFactory(ABC, Generic[PredictionModelT, ModelConfigLoaderInputT_cls, PredictionModelConfigT]):
    name: str

    @abstractmethod
    def get_model(self, cfg_input: Optional[Union[ModelConfigLoaderInputT_cls, ModelConfigLoaderInputT_fun]]=None, loader: Optional[ModelConfigLoader[ModelConfigLoaderInputT_fun, PredictionModelConfigT]]=None) -> PredictionModelT:
        pass

@dataclass(frozen=True)
class PredictionModelFactory(IPredictionModelFactory[PredictionModelT, ModelConfigLoaderInputT_cls, PredictionModelConfigT]):
    name: str
    model_cls: type[PredictionModelT]
    model_config_cls: type[PredictionModelConfigT]
    model_config_loader: ModelConfigLoader[Any, PredictionModelConfigT] = default_model_config_loader
    default_model_config_input: Optional[Any] = None

    def get_model_cfg(self, cfg_input: Optional[Union[ModelConfigLoaderInputT_cls, ModelConfigLoaderInputT_fun]]=None, cfg_loader: Optional[ModelConfigLoader[ModelConfigLoaderInputT_fun, PredictionModelConfigT]]=None) -> PredictionModelConfigT:
        if cfg_input:
            if not cfg_loader:
                cfg_loader = self.model_config_loader
            
            cfg = cfg_loader(cfg_input)
        else:
            cfg = self.model_config_loader(self.default_model_config_input)
        
        return cfg
    
    def get_model(self, *args, cfg: Optional[PredictionModelConfigT]=None, **kwargs) -> PredictionModelT:
        if not cfg:
            cfg = self.get_model_cfg(*args, **kwargs)
        
        return self.model_cls(cfg)

class MultiPredictionModelFactory(IPredictionModelFactory[PredictionModelT, ModelConfigLoaderInputT_cls, PredictionModelConfigT]):
    name: str

    factories: dict[str, IPredictionModelFactory[PredictionModelT, ModelConfigLoaderInputT_cls, PredictionModelConfigT]]
    default_factory: str

    def __init__(self,
                factories: Union[list[IPredictionModelFactory[PredictionModelT, ModelConfigLoaderInputT_cls, PredictionModelConfigT]],dict[str,IPredictionModelFactory[PredictionModelT, ModelConfigLoaderInputT_cls, PredictionModelConfigT]]],
                default_factory: str,
                name="multi"
        ):

        if isinstance(factories, list):
            factories = {f.name: f for f in factories}

        self.name = name
        self.factories = factories
        self.default_factory = default_factory
    
    def get_model(self, *args, factory: Optional[str]=None, **kwargs) -> PredictionModelT:
        if not factory:
            factory = self.default_factory
        
        return self.factories[factory].get_model(*args, **kwargs)
    
    def get_factory_names(self):
        return list(self.factories.keys())

MultiPathPredictionModelFactory = MultiPredictionModelFactory[PredictionModelT, str, PredictionModelConfigT]