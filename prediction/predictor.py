from typing import Any, Generic, TypeVar, Optional, Callable, Union
from typing_extensions import Self
from abc import ABC, abstractmethod
from dataclasses import dataclass

from config import merge_conf
from .classes import ClassList, ClassSelectorConfig, ClassSelector, ClassificationMode, ClassSelectorFactory, DEFAULT_CLASS_SELECTOR_FACTORY
from .models import PredictionModelT, PredictionModelConfigT, PredictionModelInputT, PredictionModelOutputT, MultiPathPredictionModelFactory

PredictionInputT = TypeVar("PredictionInputT")
PredictionResultT = TypeVar("PredictionResultT")

@dataclass
class PredictorConfig(Generic[PredictionModelConfigT]):
    model_config: Optional[PredictionModelConfigT] = None
    model_path: Optional[str] = None
    min_confidence: Optional[float] = None
    classification_mode: Optional[ClassificationMode] = None
    classes: Optional[ClassList] = None

PredictorConfigT = TypeVar("PredictorConfigT", bound=PredictorConfig)

InputStrategy = Callable[[PredictionInputT], PredictionModelInputT]

def default_input_strategy(input: PredictionModelInputT) -> PredictionModelInputT:
    return input

class PredictionProcessor(ABC, Generic[PredictionModelT, PredictionModelOutputT, PredictionResultT]):
    __slots__: tuple

    model: PredictionModelT

    output: PredictionModelOutputT

    def __init__(self, output: PredictionModelOutputT):
        self.output = output
    
    @classmethod
    def get_subclass(cls) -> Self:
        class cls_copy(cls):
            pass
        
        cls_copy.__name__ = cls.__name__

        return cls_copy
    
    @classmethod
    def with_args(cls, model_: PredictionModelT) -> Self:
        cls = cls.get_subclass()

        cls.model = model_

        return cls

    @classmethod
    def with_model(cls, model_: PredictionModelT) -> Self:
        cls = cls.get_subclass()

        cls.model = model_

        return cls
    
    @abstractmethod
    def process(self) -> PredictionResultT:
        pass

class PredictionProcessorWithCS(PredictionProcessor[PredictionModelT, PredictionModelOutputT, PredictionResultT]):
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

PredictionProcessorT = TypeVar("PredictionProcessorT", bound=PredictionProcessor)
PredictionProcessorWithCST = TypeVar("PredictionProcessorWithCST", bound=PredictionProcessorWithCS)

class IPredictor(ABC, Generic[PredictionModelInputT, PredictionResultT]):
    __slots__: tuple

    @abstractmethod
    def predict(self, input: PredictionInputT, input_strategy: Optional[InputStrategy[PredictionInputT, PredictionModelInputT]]=None) -> PredictionResultT:
        pass

class Predictor(IPredictor[PredictionModelInputT, PredictionResultT], Generic[PredictionModelT, PredictionModelConfigT, PredictionProcessorT, PredictionModelInputT, PredictionModelOutputT, PredictionResultT]):
    __slots__: tuple
    
    model: PredictionModelT

    input_strategy: InputStrategy[Any, PredictionModelInputT]

    prediction_processor: type[PredictionProcessorT]

    def __init__(self,
            *processor_args,
            model: PredictionModelT,
            input_strategy: InputStrategy[Any, PredictionModelInputT],
            **processor_kwargs
        ):

        self.model = model

        self.input_strategy = input_strategy
        
        self.prediction_processor = self.prediction_processor.with_args(self.model, *processor_args, **processor_kwargs)

    def predict(self, input: PredictionInputT, input_strategy: Optional[InputStrategy[PredictionInputT, PredictionModelInputT]]=None) -> PredictionResultT:
        if not input_strategy:
            input_strategy = self.input_strategy

        model_input = input_strategy(input)

        model_output: PredictionModelOutputT = self.model.predict(model_input)

        return self.prediction_processor(model_output).process()

PredictorT = TypeVar("PredictorT", bound=IPredictor)

class PredictorWithCS(Predictor[PredictionModelT, PredictionModelConfigT, PredictionProcessorWithCST, PredictionModelInputT, PredictionModelOutputT, PredictionResultT]):
    __slots__: tuple
    
    def __init__(self,
            cs: ClassSelector,
            *args, **kwargs
        ):

        super().__init__(*args, **kwargs, cs_=cs)

class IPredictorFactory(Generic[PredictorT, PredictionModelT, PredictionModelInputT, PredictorConfigT, PredictionModelConfigT]):

    @abstractmethod
    def get_predictor(self,
            model_config: Optional[PredictionModelConfigT]=None,
            model_path: Optional[str]=None,
            model_type: Optional[str]=None,
            model: Optional[PredictionModelT]=None,
            predictor: Optional[type[PredictorT]]=None,
            input_strategy: Optional[InputStrategy[PredictionInputT, PredictionModelInputT]]=None,
            cs_config: Optional[ClassSelectorConfig]=None,
            cs: Optional[ClassSelector]= None,
            mode: Optional[ClassificationMode]=None, 
            min_confidence: Optional[float]=None,
            min_confidence_pc: Optional[int]=None,
            classes: Optional[ClassList]=None,
        ) -> PredictorT:
        pass

    @abstractmethod
    def get_model_factory(self) -> MultiPathPredictionModelFactory[PredictionModelT, PredictionModelConfigT]:
        pass


@dataclass(frozen=True)
class PredictorFactory(IPredictorFactory[PredictorT, PredictionModelT, PredictionModelInputT, PredictorConfigT, PredictionModelConfigT], ABC):

    predictor: type[PredictorT]
    predictor_config: type[PredictorConfigT]
    model_factory: MultiPathPredictionModelFactory[PredictionModelT, PredictionModelConfigT]
    input_strategy: InputStrategy[Any, PredictionModelInputT]=default_input_strategy
    cs_factory: ClassSelectorFactory=DEFAULT_CLASS_SELECTOR_FACTORY

    def get_predictor(self,
            model_config: Optional[PredictionModelConfigT]=None,
            model_path: Optional[str]=None,
            model_type: Optional[str]=None,
            model: Optional[PredictionModelT]=None,
            predictor: Optional[type[PredictorT]]=None,
            input_strategy: Optional[InputStrategy[PredictionInputT, PredictionModelInputT]]=None,
            cs_config: Optional[ClassSelectorConfig]=None,
            cs: Optional[ClassSelector]= None,
            mode: Optional[ClassificationMode]=None, 
            min_confidence: Optional[float]=None,
            min_confidence_pc: Optional[int]=None,
            classes: Optional[ClassList]=None,
        ) -> PredictorT:

        if not predictor:
            predictor = self.predictor
        
        if not input_strategy:
            input_strategy = self.input_strategy

        if not model:
            model = self.model_factory.get_model(
                factory=model_type,
                cfg_input=model_path,
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

        return predictor(
            model=model,
            cs=cs,
            input_strategy=input_strategy
        )
    
    def get_model_factory(self) -> MultiPathPredictionModelFactory[PredictionModelT, PredictionModelConfigT]:
        return self.model_factory