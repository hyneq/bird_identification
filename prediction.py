from typing import Optional, Generic, TypeVar, get_type_hints
from abc import ABC, abstractmethod
from dataclasses import dataclass

import csv
import numpy as np

from classes import ClassSelector, get_class_selector, DEFAULT_CLASS_SELECTOR

TPredictionModelInput = TypeVar("TPredictionModelInput")
TPredictionModelOutput = TypeVar("TPredictionModelOutput")
TPredictionResult = TypeVar("TPredictionResult")

@dataclass()
class PredictionModelConfig:
    classes_path: str

TPredictionModelConfig = TypeVar("TPredictionModelConfig", bound=PredictionModelConfig)

class PredictionModel(ABC, Generic[TPredictionModelConfig, TPredictionModelInput, TPredictionModelOutput]):

    class_names: list[str]

    @abstractmethod
    def __init__(self, cfg: TPredictionModelConfig):
        self.class_names = self.load_classes(cfg.classes_path)

    @abstractmethod
    def predict(self, input: TPredictionModelInput) -> TPredictionModelOutput:
        pass

    @staticmethod
    def load_classes(classes_path: str):
        with open(classes_path, newline='') as f:
            reader = csv.reader(f, delimiter=',')
            return {int(row[0]): row[1] for row in reader}

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
TPredictionProcessorWithACP = TypeVar("TPredictionProcessorWithACP", bound=PredictionProcessorWithCS)

class Predictor(Generic[TPredictionModel, TPredictionModelConfig, TPredictionProcessor, TPredictionModelInput, TPredictionModelOutput]):

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

    @classmethod
    def load_model(cls, cfg: TPredictionModelConfig) -> TPredictionModel:
        return cls.model_cls(cfg)

    def predict(self, input: TPredictionModelInput) -> TPredictionResult:
        output: TPredictionModelOutput = self.model.predict(input)

        return self.prediction_processor(output).process()

class PredictorWithCS(Predictor[TPredictionModel, TPredictionModelConfig, TPredictionProcessorWithACP, TPredictionModelInput, TPredictionModelOutput]):

    def __init__(self,
            cs: Optional[ClassSelector]=None,
            *args, **kwargs
        ):

        if not cs:
            cs = DEFAULT_CLASS_SELECTOR
            
        
        self.prediction_processor = self.prediction_processor.with_cs(cs)

        super().__init__(*args, **kwargs)
            
    
def get_predictor_factory(
        name: str,
        predictor: type[PredictorWithCS[TPredictionModel, TPredictionModelConfig, TPredictionProcessorWithACP, TPredictionModelInput, TPredictionModelOutput]],
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
            model = predictor.load_model(model_config)

        if not cs:
            cs = get_class_selector(model_class_names=model.class_names, **cs_kwargs)

        return predictor(model=model, cs=cs)

    get_predictor.__name__ = name

    return get_predictor