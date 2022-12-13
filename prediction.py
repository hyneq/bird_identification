from typing import Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from abstract_classification import AbstractClassificationProcessor as ACP, get_acp

@dataclass()
class PredictionModel:
    pass

@dataclass()
class PredictionModelConfig:
    pass

class PredictionProcessor(ABC):
    __slots__: tuple

    output: np.array

    def __init__(self, output):
        self.output = output
    
    @abstractmethod
    def process(self):
        pass

class PredictionProcessorWithACP(PredictionProcessor):
    __slots__: tuple

    acp: ACP

    @classmethod
    def with_acp(cls, acp: ACP):
        class cls_copy(cls):
            pass

        cls_copy.__name__ = cls.__name__
        cls_copy.acp = acp
        
        return cls_copy


class Predictor(ABC):

    classification_processor: type[PredictionProcessor]

    def __init__(self,
            model: PredictionModel
        ):
        
        self.model = model

    @abstractmethod
    def predict(self):
        pass

class PredictorWithACP(ABC):
    def __init__(self,
            model: PredictionModel,
            acp: ACP
        ):

        super().__init__(model)

        self.classification_processor = self.classification_processor.with_acp(acp)


def get_predictor_factory(
        name: str,
        predictor: type[Predictor],
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
            acp = get_acp(class_names=model.class_names, **acp_kwargs)
        
        return predictor(model=model, acp=acp)
    
    get_predictor.__name__ = name

    return get_predictor
