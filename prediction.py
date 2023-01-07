from typing import Optional, Generic, TypeVar, get_type_hints
from abc import ABC, abstractmethod
from dataclasses import dataclass

import csv
import numpy as np

from abstract_classification import AbstractClassificationProcessor as ACP, get_acp

@dataclass()
class PredictionModelConfig:
    classes_path: str

TPredictionModelConfig = TypeVar("TPredictionModelConfig", bound=PredictionModelConfig)
TPredictionModelInput = TypeVar("TPredictionModelInput")
TPredictionModelOutput = TypeVar("TPredictionModelOutput")

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


class PredictionProcessor(ABC):
    __slots__: tuple

    output: np.ndarray

    def __init__(self, output):
        self.output = output
    
    @abstractmethod
    def process(self):
        pass

class PredictionProcessorWithACP(PredictionProcessor):
    __slots__: tuple

    acp: ACP

    @classmethod
    def with_acp(cls, acp_: ACP):
        class cls_copy(cls):
            __name__ = cls.__name__
            acp = acp_
        
        return cls_copy

TPredictionProcessor = TypeVar("TPredictionProcessor", bound=PredictionProcessor)

class Predictor(Generic[TPredictionModel, TPredictionModelConfig, TPredictionProcessor, TPredictionModelInput]):

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

    def predict(self, input: TPredictionModelInput):
        output = self.model.predict(input)

        return self.prediction_processor(output).process()


class PredictorWithACP(Predictor[TPredictionModel, TPredictionModelConfig, TPredictionProcessor, TPredictionModelInput]):

    acp: ACP

    def __init__(self,
            acp: ACP,
            *args,
            **kwargs
        ):

        super().__init__(*args, **kwargs)

        self.prediction_processor = self.prediction_processor.with_acp(acp)


def get_predictor_factory(
        name: str,
        predictor: type[PredictorWithACP],
        model_cls: type[PredictionModel],
        model_cfg_cls: type[PredictionModelConfig],
        DEFAULT_MODEL_CONFIG: PredictionModelConfig,
        acp_cls: type[ACP] = ACP
    ):

    def get_predictor(
            model_config: model_cfg_cls=DEFAULT_MODEL_CONFIG,
            model: Optional[model_cls]=None,
            predictor: type[predictor]=predictor,
            acp: Optional[acp_cls]= None,
            **acp_kwargs,
        ):

        if not model:
            model = predictor.load_model(model_config)

        if not acp:
            acp: acp_cls = get_acp(class_names=model.class_names, **acp_kwargs)
        
        return predictor(model=model, acp=acp)
    
    get_predictor.__name__ = name

    return get_predictor
