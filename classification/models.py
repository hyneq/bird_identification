import numpy as np

from prediction.models import PredictionModelConfig, AImagePredictionModel

class ClassificationModelConfig(PredictionModelConfig):
    pass

class ClassificationModel(AImagePredictionModel[ClassificationModelConfig, np.ndarray]):
    __slots__: tuple