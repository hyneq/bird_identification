import numpy as np

from prediction.models import PredictionModelConfigWithCls
from .prediction import KerasPredictionModel, KerasModelConfig
from classification.models import ClassificationModel, ClassificationModelConfig, ClassificationModelOutput

class KerasClassificationModel(KerasPredictionModel[ClassificationModelOutput], ClassificationModel):
    __slots__: tuple

    def get_output(self, predictions: np.ndarray) -> ClassificationModelOutput:
        return predictions[0]

class KerasClassificationModelConfig(KerasModelConfig, ClassificationModelConfig, PredictionModelConfigWithCls[KerasClassificationModel]):
    model_cls = KerasClassificationModel