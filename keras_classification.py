import numpy as np
from keras_prediction import KerasPredictionModel

class KerasClassificationModel(KerasPredictionModel[np.ndarray]):
    __slots__: tuple

    def get_output(self, predictions: np.ndarray) -> np.ndarray:
        return predictions[0]