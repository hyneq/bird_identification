import numpy as np

from prediction.models import PredictionModelConfigWithCls
from .prediction import KerasPredictionModel, KerasModelConfig
from classification.models import ClassificationModel, ClassificationModelConfig

class KerasClassificationModel(KerasPredictionModel[np.ndarray], ClassificationModel):
    __slots__: tuple

    def get_output(self, predictions: np.ndarray) -> np.ndarray:
        return predictions[0]

class KerasClassificationModelConfig(KerasModelConfig, ClassificationModelConfig, PredictionModelConfigWithCls[KerasClassificationModel]):
    pass