from typing import Generic, TypeVar, Optional, Union, Callable
from typing_extensions import Self
from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2

from config import merge_conf
from .classes import ClassList, ClassSelectorConfig, ClassSelector, ClassificationMode, DEFAULT_CLASS_SELECTOR, get_class_selector
from .models import TPredictionModel, TPredictionModelConfig, TPredictionModelInput, TPredictionModelOutput

TPredictionInput = TypeVar("TPredictionInput")
TPredictionResult = TypeVar("TPredictionResult")

@dataclass
class PredictorConfig(Generic[TPredictionModelConfig]):
    model_config: TPredictionModelConfig
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
            model: Optional[TPredictionModel]=None,
            model_cfg: Optional[TPredictionModelConfig]=None,
        ):

        if not model:
            self.model = self.model_cls(model_cfg)
        else:
            self.model = model
        
        self.prediction_processor = self.prediction_processor.with_model(self.model)

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
        
        self.prediction_processor = self.prediction_processor.with_cs(cs)

        super().__init__(*args, **kwargs)

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
            model: Optional[TPredictionModel]=None,
            predictor: type[predictor]=predictor,
            model_cls: type[TPredictionModel]=None,
            cs_config: Optional[ClassSelectorConfig]=None,
            cs: Optional[cs_cls]= None,
            mode: Optional[ClassificationMode]=None, 
            min_confidence: Optional[float]=None,
            classes: Optional[ClassList]=None,
        ):

        if not model:
            model = get_model(
                model_config=model_config,
                model_cls=model_cls,
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