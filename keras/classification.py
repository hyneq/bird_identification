import numpy as np

from .prediction import KerasPredictionModel
from classification.models import ClassificationModel

class KerasClassificationModel(KerasPredictionModel[np.ndarray], ClassificationModel):
    __slots__: tuple

    def get_output(self, predictions: np.ndarray) -> np.ndarray:
        return predictions[0]