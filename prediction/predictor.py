from typing import Any, Generic, TypeVar, Optional, Callable, Union, overload
from typing_extensions import Self
from abc import ABC, abstractmethod
from dataclasses import dataclass

from image_utils import BoundingBox
from config import merge_conf
from .classes import ClassList, ClassNames, ClassSelectorConfig, ClassSelector, ClassificationMode, ClassSelectorFactory, DEFAULT_CLASS_SELECTOR_FACTORY
from .models import IPredictionModel, IPredictionModelWithClasses, PredictionModelT, PredictionModelConfigT, PredictionModelWithClassesConfigT, PredictionModelInputT, PredictionModelOutputT, MultiPathPredictionModelFactory

PredictionInputT = TypeVar("PredictionInputT")
PredictionInputT_cls = TypeVar("PredictionInputT_cls")
PredictionInputT_fun = TypeVar("PredictionInputT_fun")

PredictionResultT = TypeVar("PredictionResultT")

class IPredictionResultWithClasses(ABC):

    class_names: list[str]
    confidences: list[float]

    @property
    def class_name(self) -> Optional[str]:
        return self.class_names[0] if self.class_names else None
    
    @property
    def confidence(self) -> Optional[float]:
        return self.confidences[0] if self.confidences else None

class IPredictionResultWithBoundingBoxes(ABC):

   bounding_box: BoundingBox

class IPredictionResultWithClassesAndBoundingBoxes(IPredictionResultWithClasses, IPredictionResultWithBoundingBoxes):
    pass

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

class PredictionProcessor(ABC, Generic[PredictionModelOutputT, PredictionResultT]):
    __slots__: tuple
    
    @abstractmethod
    def process(self, output: PredictionModelOutputT) -> PredictionResultT:
        pass

class PredictionProcessorWithClasses(PredictionProcessor[PredictionModelOutputT, PredictionResultT]):
    __slots__: tuple

    class_names: ClassNames

    cs: ClassSelector

    def __init__(self, class_names: ClassNames, cs: ClassSelector):
        self.class_names = class_names
        self.cs = cs

    @abstractmethod
    def process(self, output: PredictionModelOutputT) -> PredictionResultT:
        pass

PredictionProcessorT = TypeVar("PredictionProcessorT", bound=PredictionProcessor)
PredictionProcessorWithClassesT = TypeVar("PredictionProcessorWithClassesT", bound=PredictionProcessorWithClasses)

@dataclass()
class PredictionProcessorFactory(ABC, Generic[PredictionModelOutputT, PredictionResultT]):
    prediction_processor_cls: type[PredictionProcessor[PredictionModelOutputT, PredictionResultT]]

    def get_prediction_processor(self) -> PredictionProcessor[PredictionModelOutputT, PredictionResultT]:
        return self.prediction_processor_cls()

@dataclass()
class PredictionProcessorWithClassesFactory(PredictionProcessorFactory[PredictionModelOutputT, PredictionResultT]):
    prediction_processor_cls: type[PredictionProcessorWithClasses[PredictionModelOutputT, PredictionResultT]]

    cs_factory: ClassSelectorFactory=DEFAULT_CLASS_SELECTOR_FACTORY

    def get_prediction_processor(self,
            model_class_names: ClassNames,
            *args,
            cs_config: Optional[ClassSelectorConfig]=None,
            cs: Optional[ClassSelector]=None,
            mode: Optional[ClassificationMode]=None, 
            min_confidence: Optional[float]=None,
            min_confidence_pc: Optional[int]=None,
            classes: Optional[ClassList]=None,
            **kwargs
        ) -> PredictionProcessorWithClasses[PredictionModelOutputT, PredictionResultT]:
        
        if not cs:
            cs = self.cs_factory.get_class_selector(
                cfg=cs_config,
                mode=mode,
                min_confidence=min_confidence,
                min_confidence_pc=min_confidence_pc,
                classes=classes,
                model_class_names=model_class_names,
            )
        
        return self.prediction_processor_cls(model_class_names, cs, *args, **kwargs)


class APredictor(ABC, Generic[PredictionInputT_cls, PredictionModelInputT, PredictionResultT]):
    __slots__: tuple

    input_strategy: InputStrategy[Any, PredictionModelInputT]

    def __init__(self, input_strategy: InputStrategy[Any, PredictionModelInputT]):
        self.input_strategy = input_strategy

    @overload
    def predict(self, input: PredictionInputT_cls) -> PredictionResultT:
        pass

    @overload
    def predict(self, input: PredictionInputT_fun, input_strategy: InputStrategy[PredictionInputT_fun, PredictionModelInputT]) -> PredictionResultT:
        pass

    def predict(self, input: Union[PredictionInputT_cls, PredictionInputT_fun], input_strategy: Optional[InputStrategy[PredictionInputT_fun, PredictionModelInputT]]=None) -> PredictionResultT:
        if not input_strategy:
            input_strategy = self.input_strategy

        model_input = input_strategy(input)

        return self._predict(model_input)
    
    @abstractmethod
    def _predict(self, input: PredictionModelInputT) -> PredictionResultT:
        pass

class Predictor(APredictor[PredictionInputT_cls, PredictionModelInputT, PredictionResultT], Generic[PredictionInputT_cls, PredictionModelInputT, PredictionModelOutputT, PredictionResultT]):
    __slots__: tuple
    
    model: IPredictionModel[Any, PredictionModelInputT, PredictionModelOutputT]

    input_strategy: InputStrategy[Any, PredictionModelInputT]

    prediction_processor: PredictionProcessor[PredictionModelOutputT, PredictionResultT]

    def __init__(self,
            model: IPredictionModel[Any, PredictionModelInputT, PredictionModelOutputT],
            prediction_processor: PredictionProcessor[PredictionModelOutputT, PredictionResultT],
            input_strategy: InputStrategy[Any, PredictionModelInputT],
        ):

        super().__init__(input_strategy)

        self.model = model
        
        self.prediction_processor = prediction_processor

    def _predict(self, input: PredictionModelInputT) -> PredictionResultT:

        model_output: PredictionModelOutputT = self.model.predict(input)

        return self.prediction_processor.process(model_output)

PredictorT = TypeVar("PredictorT", bound=APredictor)

class PredictorWithClasses(Predictor[PredictionInputT_cls, PredictionModelInputT, PredictionModelOutputT, PredictionResultT]):
    __slots__: tuple

    prediction_processor: PredictionProcessorWithClasses[PredictionModelOutputT, PredictionResultT]

class IPredictorFactory(Generic[PredictorConfigT, PredictionModelConfigT, PredictionInputT_cls, PredictionModelInputT, PredictionModelOutputT, PredictionResultT]):

    @abstractmethod
    def get_predictor(self,
            model_config: Optional[PredictionModelConfigT]=None,
            model_path: Optional[str]=None,
            model_type: Optional[str]=None,
            model: Optional[IPredictionModel[PredictionModelConfigT, PredictionModelInputT, PredictionModelOutputT]]=None,
            predictor: Optional[type[APredictor[PredictionInputT_cls, PredictionModelInputT, PredictionResultT]]]=None,
            input_strategy: Optional[InputStrategy[PredictionInputT_cls, PredictionModelInputT]]=None,
            cs_config: Optional[ClassSelectorConfig]=None,
            cs: Optional[ClassSelector]= None,
            mode: Optional[ClassificationMode]=None, 
            min_confidence: Optional[float]=None,
            min_confidence_pc: Optional[int]=None,
            classes: Optional[ClassList]=None,
        ) -> APredictor[PredictionInputT_cls, PredictionModelInputT, PredictionResultT]:
        pass

    @abstractmethod
    def get_model_factory(self) -> MultiPathPredictionModelFactory[PredictionModelConfigT, PredictionModelInputT, PredictionModelOutputT]:
        pass


@dataclass(frozen=True)
class PredictorWithClassesFactory(IPredictorFactory[PredictorConfigT, PredictionModelConfigT, PredictionInputT_cls, PredictionModelInputT, PredictionModelOutputT, PredictionResultT], ABC):

    predictor: type[Predictor[PredictionInputT_cls, PredictionModelInputT, PredictionModelOutputT, PredictionResultT]]
    predictor_config: type[PredictorConfigT]
    model_factory: MultiPathPredictionModelFactory[PredictionModelConfigT, PredictionModelInputT, PredictionModelOutputT]
    prediction_processor_factory: PredictionProcessorWithClassesFactory[PredictionModelOutputT, PredictionResultT]
    input_strategy: InputStrategy[Any, PredictionModelInputT]=default_input_strategy

    def get_predictor(self,
            model_config: Optional[PredictionModelWithClassesConfigT]=None,
            model_path: Optional[str]=None,
            model_type: Optional[str]=None,
            model: Optional[IPredictionModelWithClasses[PredictionModelWithClassesConfigT, PredictionModelInputT, PredictionModelOutputT]]=None,
            predictor: Optional[type[Predictor[PredictionInputT_cls, PredictionModelInputT, PredictionModelOutputT, PredictionResultT]]]=None,
            prediction_processor: Optional[PredictionProcessorWithClasses[PredictionModelOutputT, PredictionResultT]]=None,
            input_strategy: Optional[InputStrategy[PredictionInputT, PredictionModelInputT]]=None,
            cs_config: Optional[ClassSelectorConfig]=None,
            cs: Optional[ClassSelector]= None,
            mode: Optional[ClassificationMode]=None, 
            min_confidence: Optional[float]=None,
            min_confidence_pc: Optional[int]=None,
            classes: Optional[ClassList]=None,
        ) -> Predictor[PredictionInputT_cls, PredictionModelInputT, PredictionModelOutputT, PredictionResultT]:

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
        
        if not prediction_processor:
            prediction_processor = self.prediction_processor_factory.get_prediction_processor(
                model_class_names=model.class_names,
                cs_config=cs_config,
                cs=cs,
                mode=mode,
                min_confidence=min_confidence,
                min_confidence_pc=min_confidence_pc,
                classes=classes
            )

        return predictor(
            model=model,
            prediction_processor=prediction_processor,
            input_strategy=input_strategy,
        )
    
    def get_model_factory(self) -> MultiPathPredictionModelFactory[PredictionModelConfigT, PredictionModelInputT, PredictionModelOutputT]:
        return self.model_factory