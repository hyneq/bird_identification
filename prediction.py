from typing import Optional, Generic, TypeVar, get_type_hints
from abc import ABC, abstractmethod
from dataclasses import dataclass

import csv
import numpy as np

from abstract_classification import AbstractClassificationProcessor as ACP, get_acp, DEFAULT_ACP

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
TPredictionProcessorWithACP = TypeVar("TPredictionProcessorWithACP", bound=PredictionProcessorWithACP)

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

class PredictorWithACP(Predictor[TPredictionModel, TPredictionModelConfig, TPredictionProcessorWithACP, TPredictionModelInput, TPredictionModelOutput]):

    def __init__(self,
            acp: Optional[ACP]=None,
            *args, **kwargs
        ):

        if not acp:
            acp = DEFAULT_ACP
            
        
        self.prediction_processor = self.prediction_processor.with_acp(acp)

        super().__init__(*args, **kwargs)
            
    
def get_predictor_factory(
        name: str,
        predictor: type[PredictorWithACP[TPredictionModel, TPredictionModelConfig, TPredictionProcessorWithACP, TPredictionModelInput, TPredictionModelOutput]],
        model_cls: type[TPredictionModel],
        model_cfg_cls: type[TPredictionModelConfig],
        DEFAULT_MODEL_CONFIG: TPredictionModelConfig,
        acp_cls: type[ACP] = ACP
    ):

    def get_predictor(
            model_config: model_cfg_cls=DEFAULT_MODEL_CONFIG,
            model: Optional[model_cls]=None,
            predictor: type[predictor]=predictor,
            acp: Optional[acp_cls]= None,
            **acp_kwargs
        ):

        if not model:
            model = predictor.load_model(model_config)

        if not acp:
            acp = get_acp(class_names=model.class_names, **acp_kwargs)

        return predictor(model=model, acp=acp)

    get_predictor.__name__ = name

    return get_predictor