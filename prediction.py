from typing import Optional, Generic, TypeVar
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import cv2

from classes import ClassSelector, get_class_selector, DEFAULT_CLASS_SELECTOR, ClassNames

Image = np.ndarray

TPredictionModelInput = TypeVar("TPredictionModelInput")
TPredictionModelOutput = TypeVar("TPredictionModelOutput")
TPredictionResult = TypeVar("TPredictionResult")

@dataclass()
class PredictionModelConfig:
    classes_path: str

TPredictionModelConfig = TypeVar("TPredictionModelConfig", bound=PredictionModelConfig)

class APredictionModel(ABC, Generic[TPredictionModelConfig, TPredictionModelInput, TPredictionModelOutput]):
    __slots__: tuple

    @abstractmethod
    def __init__(self, cfg: TPredictionModelConfig):
        pass

    @abstractmethod
    def predict(self, input: TPredictionModelInput) -> TPredictionModelOutput:
        pass

class PredictionModel(Generic[TPredictionModelConfig, TPredictionModelInput, TPredictionModelOutput]):
    __slots__: tuple

    class_names: ClassNames

    def __init__(self, cfg: TPredictionModelConfig):
        self.class_names = self.load_classes(cfg.classes_path)

    @staticmethod
    def load_classes(classes_path: str):
        return ClassNames.load_from_file(classes_path)

TPredictionModel = TypeVar("TPredictionModel", bound=APredictionModel)

class ImagePredictionModel(PredictionModel[TPredictionModelConfig, Image, TPredictionModelOutput]):
    pass

class PredictionProcessor(ABC, Generic[TPredictionModel, TPredictionModelOutput, TPredictionResult]):
    __slots__: tuple

    model: TPredictionModel

    output: TPredictionModelOutput

    def __init__(self, output: TPredictionModelOutput):
        self.output = output
    
    @classmethod
    def with_model(cls, model_: TPredictionModel):
        class cls_copy(cls):
            model = model_
        
        cls_copy.__name__ = cls.__name__

        return cls_copy
    
    @abstractmethod
    def process(self) -> TPredictionResult:
        pass

class PredictionProcessorWithCS(PredictionProcessor[TPredictionModel, TPredictionModelOutput, TPredictionResult]):
    __slots__: tuple

    cs: ClassSelector

    @classmethod
    def with_cs(cls, cs_: ClassSelector):
        class cls_copy(cls):
            cs = cs_
        
        cls_copy.__name__ = cls.__name__
        
        return cls_copy

TPredictionProcessor = TypeVar("TPredictionProcessor", bound=PredictionProcessor)
TPredictionProcessorWithCS = TypeVar("TPredictionProcessorWithCS", bound=PredictionProcessorWithCS)

class APredictor(ABC, Generic[TPredictionModel, TPredictionModelConfig, TPredictionProcessor, TPredictionModelInput, TPredictionModelOutput, TPredictionResult]):
    __slots__: tuple

    @abstractmethod
    def predict(TPredictionModelInput) -> TPredictionResult:
        pass

class Predictor(APredictor[TPredictionModel, TPredictionModelConfig, TPredictionProcessor, TPredictionModelInput, TPredictionModelOutput, TPredictionResult]):
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

TPredictor = TypeVar("TPredictor", bound=APredictor)

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

class FileImagePredictor(APredictor[TPredictionModel, TPredictionModelConfig, TPredictionProcessorWithCS, str, TPredictionModelOutput, TPredictionResult], Generic[TPredictor]):
    __slots__: tuple

    predictor_cls: type[TPredictor]

    predictor: TPredictor

    def __init__(self, *args, predictor: Optional[TPredictor] = None, **kwargs):

        if not predictor:
            predictor = self.predictor_cls(*args, **kwargs)
        
        self.predictor = predictor
    
    def predict(self, path: str) -> TPredictionModelOutput:
        image = cv2.imread(path)

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