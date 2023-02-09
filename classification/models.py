import numpy as np

from prediction.models import PredictionModelConfig, AImagePredictionModel

class ClassificationModelConfig(PredictionModelConfig):
    pass

ClassificationModelOutput = np.ndarray

class ClassificationModel(AImagePredictionModel[ClassificationModelConfig, ClassificationModelOutput]):
    __slots__: tuple