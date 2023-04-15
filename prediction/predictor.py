from typing import Any, Generic, TypeVar, Optional, Callable, Union
from typing_extensions import Self
from abc import ABC, abstractmethod
from dataclasses import dataclass

from config import merge_conf
from .classes import ClassList, ClassSelectorConfig, ClassSelector, ClassificationMode, ClassSelectorFactory, DEFAULT_CLASS_SELECTOR_FACTORY
from .models import IPredictionModel, PredictionModelT, PredictionModelConfigT, PredictionModelInputT, PredictionModelOutputT, MultiPathPredictionModelFactory

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

class Predictor(IPredictor[PredictionModelInputT, PredictionResultT], Generic[PredictionModelInputT, PredictionModelOutputT, PredictionResultT]):
    __slots__: tuple
    
    model: IPredictionModel[Any, PredictionModelInputT, PredictionModelOutputT]

    input_strategy: InputStrategy[Any, PredictionModelInputT]

    prediction_processor: type[PredictionProcessor[IPredictionModel[Any, PredictionModelInputT, PredictionModelOutputT], PredictionModelOutputT, PredictionResultT]]

    def __init__(self,
            *processor_args,
            model: IPredictionModel[Any, PredictionModelInputT, PredictionModelOutputT],
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

class PredictorWithCS(Predictor[PredictionModelInputT, PredictionModelOutputT, PredictionResultT]):
    __slots__: tuple

    prediction_processor: type[PredictionProcessorWithCS[IPredictionModel[Any, PredictionModelInputT, PredictionModelOutputT], PredictionModelOutputT, PredictionResultT]]
    
    def __init__(self,
            cs: ClassSelector,
            *args, **kwargs
        ):

        super().__init__(*args, **kwargs, cs_=cs)

class IPredictorFactory(Generic[PredictorConfigT, PredictionModelConfigT, PredictionModelInputT, PredictionModelOutputT, PredictionResultT]):

    @abstractmethod
    def get_predictor(self,
            model_config: Optional[PredictionModelConfigT]=None,
            model_path: Optional[str]=None,
            model_type: Optional[str]=None,
            model: Optional[IPredictionModel[PredictionModelConfigT, PredictionModelInputT, PredictionModelOutputT]]=None,
            predictor: Optional[type[IPredictor[PredictionModelInputT, PredictionResultT]]]=None,
            input_strategy: Optional[InputStrategy[PredictionInputT, PredictionModelInputT]]=None,
            cs_config: Optional[ClassSelectorConfig]=None,
            cs: Optional[ClassSelector]= None,
            mode: Optional[ClassificationMode]=None, 
            min_confidence: Optional[float]=None,
            min_confidence_pc: Optional[int]=None,
            classes: Optional[ClassList]=None,
        ) -> IPredictor[PredictionModelInputT, PredictionResultT]:
        pass

    @abstractmethod
    def get_model_factory(self) -> MultiPathPredictionModelFactory[PredictionModelConfigT, PredictionModelInputT, PredictionModelOutputT]:
        pass


@dataclass(frozen=True)
class PredictorFactory(IPredictorFactory[PredictorConfigT, PredictionModelConfigT, PredictionModelInputT, PredictionModelOutputT, PredictionResultT], ABC):

    predictor: type[Predictor[PredictionModelInputT, PredictionModelOutputT, PredictionResultT]]
    predictor_config: type[PredictorConfigT]
    model_factory: MultiPathPredictionModelFactory[PredictionModelConfigT, PredictionModelInputT, PredictionModelOutputT]
    input_strategy: InputStrategy[Any, PredictionModelInputT]=default_input_strategy
    cs_factory: ClassSelectorFactory=DEFAULT_CLASS_SELECTOR_FACTORY

    def get_predictor(self,
            model_config: Optional[PredictionModelConfigT]=None,
            model_path: Optional[str]=None,
            model_type: Optional[str]=None,
            model: Optional[IPredictionModel[PredictionModelConfigT, PredictionModelInputT, PredictionModelOutputT]]=None,
            predictor: Optional[type[Predictor[PredictionModelInputT, PredictionModelOutputT, PredictionResultT]]]=None,
            input_strategy: Optional[InputStrategy[PredictionInputT, PredictionModelInputT]]=None,
            cs_config: Optional[ClassSelectorConfig]=None,
            cs: Optional[ClassSelector]= None,
            mode: Optional[ClassificationMode]=None, 
            min_confidence: Optional[float]=None,
            min_confidence_pc: Optional[int]=None,
            classes: Optional[ClassList]=None,
        ) -> Predictor[PredictionModelInputT, PredictionModelOutputT, PredictionResultT]:

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
    
    def get_model_factory(self) -> MultiPathPredictionModelFactory[PredictionModelConfigT, PredictionModelInputT, PredictionModelOutputT]:
        return self.model_factory