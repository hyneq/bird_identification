from typing import Optional, Generic, TypeVar, get_type_hints
from abc import ABC, abstractmethod
from dataclasses import dataclass

import csv
import numpy as np
from PIL.Image import Image, open as open_image

from classes import ClassSelector, get_class_selector, DEFAULT_CLASS_SELECTOR, ClassNames

TPredictionModelInput = TypeVar("TPredictionModelInput")
TPredictionModelOutput = TypeVar("TPredictionModelOutput")
TPredictionResult = TypeVar("TPredictionResult")

@dataclass()
class PredictionModelConfig:
    classes_path: str

TPredictionModelConfig = TypeVar("TPredictionModelConfig", bound=PredictionModelConfig)

class PredictionModel(ABC, Generic[TPredictionModelConfig, TPredictionModelInput, TPredictionModelOutput]):
    __slots__: tuple

    class_names: ClassNames

    @abstractmethod
    def __init__(self, cfg: TPredictionModelConfig):
        self.class_names = self.load_classes(cfg.classes_path)

    @abstractmethod
    def predict(self, input: TPredictionModelInput) -> TPredictionModelOutput:
        pass

    @staticmethod
    def load_classes(classes_path: str):
        return ClassNames.load_from_file(classes_path)

TPredictionModel = TypeVar("TPredictionModel", bound=PredictionModel)

class PredictionProcessor(ABC, Generic[TPredictionModelOutput, TPredictionResult]):
    __slots__: tuple

    output: TPredictionModelOutput

    def __init__(self, output: TPredictionModelOutput):
        self.output = output
    
    @abstractmethod
    def process(self) -> TPredictionResult:
        pass

class PredictionProcessorWithCS(PredictionProcessor[TPredictionModelOutput, TPredictionResult]):
    __slots__: tuple

    cs: ClassSelector

    @classmethod
    def with_cs(cls, cs_: ClassSelector):
        class cls_copy(cls):
            __name__ = cls.__name__
            cs = cs_
        
        return cls_copy

TPredictionProcessor = TypeVar("TPredictionProcessor", bound=PredictionProcessor)
TPredictionProcessorWithCS = TypeVar("TPredictionProcessorWithCS", bound=PredictionProcessorWithCS)

class Predictor(Generic[TPredictionModel, TPredictionModelConfig, TPredictionProcessor, TPredictionModelInput, TPredictionModelOutput]):
    __slots__: tuple
    
    model_cls: type[TPredictionModel]
    
    model: TPredictionModel

    prediction_processor: type[TPredictionProcessor]

    def __init__(self,
            model: Optional[TPredictionModel]=None,
            model_cfg: Optional[TPredictionModelConfig]=None,
        ):

        if not model:
            self.model = self.load_model(model_cfg)
        else:
            self.model = model

    def predict(self, input: TPredictionModelInput) -> TPredictionResult:
        output: TPredictionModelOutput = self.model.predict(input)

        return self.prediction_processor(output).process()

TPredictor = TypeVar("TPredictor", bound=Predictor)

class PredictorWithCS(Predictor[TPredictionModel, TPredictionModelConfig, TPredictionProcessorWithCS, TPredictionModelInput, TPredictionModelOutput]):
    __slots__: tuple
    
    def __init__(self,
            cs: Optional[ClassSelector]=None,
            *args, **kwargs
        ):

        if not cs:
            cs = DEFAULT_CLASS_SELECTOR
            
        
        self.prediction_processor = self.prediction_processor.with_cs(cs)

        super().__init__(*args, **kwargs)

class FileImagePredictor(Generic[TPredictor]):
    __slots__: tuple

    predictor_cls: type[TPredictor]

    predictor: TPredictor

    def __init__(self, *args, predictor: Optional[TPredictor] = None, **kwargs):

        if not predictor:
            predictor = self.predictor_cls(*args, **kwargs)
        
        self.predictor = predictor
    
    def predict(self, path: str):
        image = open_image(path)

        return self.predictor.predict(image)

    
def get_predictor_factory(
        name: str,
        predictor: type[PredictorWithCS[TPredictionModel, TPredictionModelConfig, TPredictionProcessorWithCS, TPredictionModelInput, TPredictionModelOutput]],
        model_cls: type[TPredictionModel],
        model_cfg_cls: type[TPredictionModelConfig],
        DEFAULT_MODEL_CONFIG: TPredictionModelConfig,
        cs_cls: type[ClassSelector] = ClassSelector
    ):

    def get_predictor(
            model_config: model_cfg_cls=DEFAULT_MODEL_CONFIG,
            model: Optional[model_cls]=None,
            predictor: type[predictor]=predictor,
            cs: Optional[cs_cls]= None,
            **cs_kwargs
        ):

        if not model:
            model = model_cls(model_config)

        if not cs:
            cs = get_class_selector(model_class_names=model.class_names, **cs_kwargs)

        return predictor(model=model, cs=cs)

    get_predictor.__name__ = name

    return get_predictor