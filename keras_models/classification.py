import numpy as np

from .prediction import KerasPredictionModel, KerasModelConfig
from classification.models import ClassificationModel, PathClassificationModelConfig, ClassificationModelOutput, PathClassificationModelFactory

class KerasClassificationModel(KerasPredictionModel[ClassificationModelOutput], ClassificationModel):
    __slots__: tuple

    def get_output(self, predictions: np.ndarray) -> ClassificationModelOutput:
        return predictions[0]

class KerasClassificationModelConfig(KerasModelConfig, PathClassificationModelConfig):
    pass

KERAS_CLASSIFICATION_MODEL_FACTORY = PathClassificationModelFactory[KerasClassificationModel, KerasClassificationModelConfig](
    name="keras",
    model_cls=KerasClassificationModel,
    model_config_cls=KerasClassificationModelConfig,
    default_path="models/classification"
)