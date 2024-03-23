import os
from dataclasses import dataclass
from abc import ABC, abstractmethod
from threading import Lock

import numpy as np
from tensorflow import keras
import cv2

from ..prediction.models import (
    PredictionModel,
    PredictionModelWithClasses,
    PredictionModelConfig,
    PredictionModelWithClassesConfig,
    PredictionModelOutputT,
)
from ..image_utils import Image


@dataclass
class KerasModelConfig(PredictionModelConfig):
    model_path: str

    @classmethod
    def from_path(cls, path: str):
        return cls(model_path=os.path.join(path, "model.h5"))


@dataclass
class KerasModelWithClassesConfig(KerasModelConfig, PredictionModelWithClassesConfig):
    @classmethod
    def from_path(cls, path: str):
        return cls(
            model_path=os.path.join(path, "model.h5"),
            classes_path=os.path.join(path, "classes.txt"),
        )


class KerasPredictionModel(
    PredictionModel[KerasModelConfig, Image, PredictionModelOutputT], ABC
):
    __slots__: tuple

    model: keras.Model
    model_lock: Lock

    def __init__(self, cfg: KerasModelConfig):
        self.model = keras.models.load_model(cfg.model_path)
        self.model_lock = Lock()
        super().__init__(cfg)

    def predict(self, input: Image) -> PredictionModelOutputT:
        input = self.get_input(input)

        with self.model_lock:
            predictions = self.model.predict(input)

        return self.get_output(predictions)

    @abstractmethod
    def get_input(self, input: Image) -> np.ndarray:
        pass

    @abstractmethod
    def get_output(self, predictions: np.ndarray) -> PredictionModelOutputT:
        pass


class KerasPredictionModelWithClasses(
    KerasPredictionModel,
    PredictionModelWithClasses[
        KerasModelWithClassesConfig, Image, PredictionModelOutputT
    ],
):
    pass
