from typing import Generic, TypeVar, Optional, Callable
from typing_extensions import Self
from abc import ABC, abstractmethod
from dataclasses import dataclass

from config import merge_conf
from .classes import ClassList, ClassSelectorConfig, ClassSelector, ClassificationMode, ClassSelectorFactory, DEFAULT_CLASS_SELECTOR_FACTORY, get_class_selector
from .models import TPredictionModel, TPredictionModelConfig, TPathPredictionModelConfig, TPredictionModelInput, TPredictionModelOutput, MultiPathPredictionModelFactory

TPredictionInput = TypeVar("TPredictionInput")
TPredictionResult = TypeVar("TPredictionResult")

@dataclass
class PredictorConfig(Generic[TPredictionModelConfig]):
    model_config: Optional[TPredictionModelConfig] = None
    model_path: Optional[str] = None
    min_confidence: Optional[float] = None
    classification_mode: Optional[ClassificationMode] = None
    classes: Optional[ClassList] = None

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
    
    model: TPredictionModel

    prediction_processor: type[TPredictionProcessor]

    def __init__(self,
            *processor_args,
            model: TPredictionModel,
            **processor_kwargs
        ):

        self.model = model
        
        self.prediction_processor = self.prediction_processor.with_args(self.model, *processor_args, **processor_kwargs)

    def predict(self, input: TPredictionModelInput) -> TPredictionResult:
        output: TPredictionModelOutput = self.model.predict(input)

        return self.prediction_processor(output).process()

TPredictor = TypeVar("TPredictor", bound=IPredictor)

class PredictorWithCS(Predictor[TPredictionModel, TPredictionModelConfig, TPredictionProcessorWithCS, TPredictionModelInput, TPredictionModelOutput, TPredictionResult]):
    __slots__: tuple
    
    def __init__(self,
            cs: ClassSelector,
            *args, **kwargs
        ):

        super().__init__(*args, **kwargs, cs_=cs)

class IPredictorFactory(Generic[TPredictor, TPredictionModel, TPredictorConfig, TPathPredictionModelConfig]):

    @abstractmethod
    def get_predictor(self,
            model_config: Optional[TPathPredictionModelConfig]=None,
            model_path: Optional[str]=None,
            model_type: Optional[str]=None,
            model: Optional[TPredictionModel]=None,
            predictor: Optional[type[TPredictor]]=None,
            cs_config: Optional[ClassSelectorConfig]=None,
            cs: Optional[ClassSelector]= None,
            mode: Optional[ClassificationMode]=None, 
            min_confidence: Optional[float]=None,
            min_confidence_pc: Optional[int]=None,
            classes: Optional[ClassList]=None,
        ) -> TPredictor:
        pass

    @abstractmethod
    def get_model_factory(self) -> MultiPathPredictionModelFactory[TPredictionModel, TPathPredictionModelConfig]:
        pass


@dataclass
class PredictorFactory(IPredictorFactory[TPredictor, TPredictionModel, TPredictorConfig, TPathPredictionModelConfig], ABC):

    predictor: type[TPredictor]
    predictor_config: type[TPredictorConfig]
    model_factory: MultiPathPredictionModelFactory[TPredictionModel, TPathPredictionModelConfig]
    cs_factory: ClassSelectorFactory=DEFAULT_CLASS_SELECTOR_FACTORY

    def get_predictor(self,
            model_config: Optional[TPredictionModelConfig]=None,
            model_path: Optional[str]=None,
            model_type: Optional[str]=None,
            model: Optional[TPredictionModel]=None,
            predictor: Optional[type[TPredictor]]=None,
            cs_config: Optional[ClassSelectorConfig]=None,
            cs: Optional[ClassSelector]= None,
            mode: Optional[ClassificationMode]=None, 
            min_confidence: Optional[float]=None,
            min_confidence_pc: Optional[int]=None,
            classes: Optional[ClassList]=None,
        ) -> TPredictor:

        if not predictor:
            predictor = self.predictor

        if not model:
            model = self.model_factory.get_model(
                factory=model_type,
                path=model_path,
                cfg=model_config
            )

        if not cs:
            cs = self.cs_factory.get_class_selector(
                cfg=cs_config,
                mode=mode,
                min_confidence=min_confidence,
                min_confidence_pc=min_confidence_pc,
                classes=classes,
                model_class_names=model.class_names,
            )

        return predictor(model=model, cs=cs)
    
    def get_model_factory(self) -> MultiPathPredictionModelFactory[TPredictionModel, TPathPredictionModelConfig]:
        return self.model_factory

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