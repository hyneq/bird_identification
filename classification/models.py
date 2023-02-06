import os
from threading import Lock
from dataclasses import dataclass

import numpy as np
from tensorflow import keras
import cv2

from prediction.models import PredictionModelConfig, AImagePredictionModel, Image

class ClassificationModelConfig(PredictionModelConfig):
    pass

class ClassificationModel(AImagePredictionModel[ClassificationModelConfig, np.ndarray]):
    __slots__: tuple