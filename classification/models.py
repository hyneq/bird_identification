import os
from threading import Lock
from dataclasses import dataclass

import numpy as np
from tensorflow import keras
import cv2

from prediction.models import PredictionModelConfig, APredictionModel, Image

class ClassificationModelConfig(PredictionModelConfig):
    pass

class ClassificationModel(APredictionModel[ClassificationModelConfig, Image, np.ndarray]):
    __slots__: tuple