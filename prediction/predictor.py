from typing import Generic, TypeVar, Optional, Union, Callable
from typing_extensions import Self
from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2

from config import merge_conf
from .classes import ClassList, ClassSelectorConfig, ClassSelector, ClassificationMode, DEFAULT_CLASS_SELECTOR, ClassSelectorFactory, DEFAULT_CLASS_SELECTOR_FACTORY, get_class_selector
from .models import TPredictionModel, TPredictionModelConfig, TPathPredictionModelConfig, TPredictionModelInput, TPredictionModelOutput, MultiPathPredictionModelFactory

TPredictionInput = TypeVar("TPredictionInput")
TPredictionResult = TypeVar("TPredictionResult")

@dataclass
class PredictorConfig(Generic[TPredictionModelConfig]):
    model_config: TPredictionModelConfig
    model_path: str
    min_confidence: float = None
    classification_mode: ClassificationMode = None
    classes: Union[list[int],list[str]] = None

TPredictorConfig = TypeVar("TPredictorConfig", bound=PredictorConfig)

class PredictionProcessor(ABC, Generic[TPredictionModel, TPredictionModelOutput, TPredictionResult]):
    __slots__: tuple

    model: TPredictionModel

    output: TPredictionModelOutput

    def __init__(self, output: TPredictionModelOutput):
        self.output = output
    
    @classmethod
    def get_subclass(cls) -> Self:
        class cls_copy(cls):
            pass
        
        cls_copy.__name__ = cls.__name__

        return cls_copy
    
    @classmethod
    def with_args(cls, model_: TPredictionModel) -> Self:
        cls = cls.get_subclass()

        cls.model = model_

        return cls

    @classmethod
    def with_model(cls, model_: TPredictionModel) -> Self:
        cls = cls.get_subclass()

        cls.model = model_

        return cls
    
    @abstractmethod
    def process(self) -> TPredictionResult:
        pass

class PredictionProcessorWithCS(PredictionProcessor[TPredictionModel, TPredictionModelOutput, TPredictionResult]):
    __slots__: tuple

    cs: ClassSelector

    @classmethod
    def with_args(cls, *args, cs_: ClassSelector, **kwargs):
        cls = super().with_args(*args, **kwargs)

        cls.cs = cs_

        return cls 

    @classmethod
    def with_cs(cls, cs_: ClassSelector) -> Self:
        cls = cls.get_subclass()

        cls.cs = cs_

        return cls

TPredictionProcessor = TypeVar("TPredictionProcessor", bound=PredictionProcessor)
TPredictionProcessorWithCS = TypeVar("TPredictionProcessorWithCS", bound=PredictionProcessorWithCS)

class IPredictor(ABC, Generic[TPredictionInput, TPredictionResult]):
    __slots__: tuple

    @abstractmethod
    def predict(self, input: TPredictionInput) -> TPredictionResult:
        pass

class Predictor(IPredictor[TPredictionModelInput, TPredictionResult], Generic[TPredictionModel, TPredictionModelConfig, TPredictionProcessor, TPredictionModelInput, TPredictionModelOutput, TPredictionResult]):
    __slots__: tuple
    
    model_cls: type[TPredictionModel]
    
    model: TPredictionModel

    prediction_processor: type[TPredictionProcessor]

    def __init__(self,
            *processor_args,
            model: Optional[TPredictionModel]=None,
            model_cfg: Optional[TPredictionModelConfig]=None,
            **processor_kwargs
        ):

        if not model:
            self.model = self.model_cls(model_cfg)
        else:
            self.model = model
        
        self.prediction_processor = self.prediction_processor.with_args(self.model, *processor_args, **processor_kwargs)

    def predict(self, input: TPredictionModelInput) -> TPredictionResult:
        output: TPredictionModelOutput = self.model.predict(input)

        return self.prediction_processor(output).process()

TPredictor = TypeVar("TPredictor", bound=IPredictor)

class PredictorWithCS(Predictor[TPredictionModel, TPredictionModelConfig, TPredictionProcessorWithCS, TPredictionModelInput, TPredictionModelOutput, TPredictionResult]):
    __slots__: tuple
    
    def __init__(self,
            cs: Optional[ClassSelector]=None,
            *args, **kwargs
        ):

        if not cs:
            cs = DEFAULT_CLASS_SELECTOR()

        super().__init__(*args, **kwargs, cs_=cs)

class FileImagePredictor(IPredictor[str, TPredictionResult], Generic[TPredictor, TPredictionResult]):
    __slots__: tuple

    predictor_cls: type[TPredictor]

    predictor: TPredictor

    def __init__(self, *args, predictor: Optional[TPredictor] = None, **kwargs):

        if not predictor:
            predictor = self.predictor_cls(*args, **kwargs)
        
        self.predictor = predictor
    
    def predict(self, path: str) -> TPredictionResult:
        image = cv2.imread(path)

        return self.predictor.predict(image)

class IPredictorFactory(Generic[TPredictor, TPredictionModel, TPathPredictionModelConfig, TPredictionProcessorWithCS, TPredictionModelInput, TPredictionModelOutput, TPredictionResult]):

    @abstractmethod
    def get_predictor(self,
            model_config: Optional[TPredictionModelConfig]=None,
            model_path: Optional[str]=None,
            model: Optional[TPredictionModel]=None,
            predictor: Optional[type[TPredictor]]=None,
            cs_config: Optional[ClassSelectorConfig]=None,
            cs: Optional[ClassSelector]= None,
            mode: Optional[ClassificationMode]=None, 
            min_confidence: Optional[float]=None,
            classes: Optional[ClassList]=None,
        ) -> TPredictor:
        pass


@dataclass
class PredictorFactory(IPredictorFactory[TPredictor, TPredictionModel, TPathPredictionModelConfig, TPredictionProcessorWithCS, TPredictionModelInput, TPredictionModelOutput, TPredictionResult], ABC):

    predictor: type[TPredictor]
    model_factory: MultiPathPredictionModelFactory[TPredictionModel, TPredictionModelConfig]
    cs_factory: ClassSelectorFactory=DEFAULT_CLASS_SELECTOR_FACTORY

    def get_predictor(self,
            model_config: Optional[TPredictionModelConfig]=None,
            model_path: Optional[str]=None,
            model: Optional[TPredictionModel]=None,
            predictor: Optional[type[TPredictor]]=None,
            cs_config: Optional[ClassSelectorConfig]=None,
            cs: Optional[ClassSelector]= None,
            mode: Optional[ClassificationMode]=None, 
            min_confidence: Optional[float]=None,
            classes: Optional[ClassList]=None,
        ) -> TPredictor:

        if not predictor:
            predictor = self.predictor

        if not model:
            model = self.model_factory.get_model(
                path=model_path,
                cfg=model_config
            )

        if not cs:
            cs = get_class_selector(
                cfg=cs_config,
                mode=mode,
                min_confidence=min_confidence,
                classes=classes,
                model_class_names=model.class_names,
            )

        return predictor(model=model, cs=cs)


def get_predictor_factory(
        name: str,
        predictor: type[PredictorWithCS[TPredictionModel, TPredictionModelConfig, TPredictionProcessorWithCS, TPredictionModelInput, TPredictionModelOutput, TPredictionResult]],
        predictor_config_cls: type[TPredictorConfig],
        get_model: Callable,
        cs_cls: type[ClassSelector] = ClassSelector,
    ):

    @merge_conf(predictor_config_cls)
    def get_predictor(
            model_config: Optional[TPredictionModelConfig]=None,
            model_path: Optional[str]=None,
            model: Optional[TPredictionModel]=None,
            predictor: type[predictor]=predictor,
            cs_config: Optional[ClassSelectorConfig]=None,
            cs: Optional[cs_cls]= None,
            mode: Optional[ClassificationMode]=None, 
            min_confidence: Optional[float]=None,
            classes: Optional[ClassList]=None,
        ):

        if not model:
            model = get_model(
                path=model_path,
                cfg=model_config
            )

        if not cs:
            cs = get_class_selector(
                cfg=cs_config,
                mode=mode,
                min_confidence=min_confidence,
                classes=classes,
                model_class_names=model.class_names,
            )

        return predictor(model=model, cs=cs)

    get_predictor.__name__ = name

    return get_predictor