from typing import Optional, Generic, TypeVar, get_type_hints
from abc import ABC, abstractmethod
from dataclasses import dataclass

import csv
import numpy as np

from abstract_classification import AbstractClassificationProcessor as ACP, get_acp

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

class PredictionProcessorWithACP(PredictionProcessor[TPredictionModelOutput, TPredictionResult]):
    __slots__: tuple

    acp: ACP

    @classmethod
    def with_acp(cls, acp_: ACP):
        class cls_copy(cls):
            __name__ = cls.__name__
            acp = acp_
        
        return cls_copy

TPredictionProcessor = TypeVar("TPredictionProcessor", bound=PredictionProcessor)

class Predictor(Generic[TPredictionModel, TPredictionModelConfig, TPredictionProcessor, TPredictionModelInput, TPredictionModelOutput]):

    model_cls: type[TPredictionModel]
    
    model: TPredictionModel

    predciction_processor: type[TPredictionProcessor]

    def __init__(self,
            model: Optional[TPredictionModel]=None,
            model_cfg: Optional[TPredictionModelConfig]=None
        ):

        if not model:
            self.model = self.load_model(model_cfg)
        else:
            self.model = model

    @classmethod
    def load_model(cls, cfg: TPredictionModelConfig) -> TPredictionModel:
        return cls.model_cls(cfg)

    def predict(self, input: TPredictionModelInput) -> TPredictionModelOutput:
        output: TPredictionModelOutput = self.model.predict(input)

        return self.predciction_processor(output).process()