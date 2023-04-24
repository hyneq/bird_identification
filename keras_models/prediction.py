import os
from dataclasses import dataclass
from abc import ABC, abstractmethod
from threading import Lock

import numpy as np
from tensorflow import keras
import cv2

from prediction.models import PredictionModel, PredictionModelWithClasses, PredictionModelConfig, PredictionModelWithClassesConfig, PredictionModelOutputT
from image_utils import Image

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
        return cls(model_path=os.path.join(path, "model.h5"), classes_path=os.path.join(path, "classes.csv"))

class KerasPredictionModel(PredictionModel[KerasModelConfig, Image, PredictionModelOutputT], ABC):
    __slots__: tuple

    model: keras.Model
    model_lock: Lock

    def __init__(self, cfg: KerasModelConfig):
        self.model = keras.models.load_model(cfg.model_path)
        self.model_lock = Lock()
        super().__init__(cfg)
    
    def predict(self, input: Image) -> PredictionModelOutputT:
        blob: np.ndarray = cv2.dnn.blobFromImage(input, size=(224,224), swapRB=True)

        blob = np.moveaxis(blob, (1, 2, 3), (3, 1, 2)) # Making the color channel the last dimension instead of the first, in order to match model input shape

        with self.model_lock:
            predictions = self.model.predict(blob)
        
        return self.get_output(predictions)
    
    @abstractmethod
    def get_output(self, predictions: np.ndarray) -> PredictionModelOutputT:
        pass

class KerasPredictionModelWithClasses(KerasPredictionModel, PredictionModelWithClasses[KerasModelWithClassesConfig, Image, PredictionModelOutputT]):
    pass