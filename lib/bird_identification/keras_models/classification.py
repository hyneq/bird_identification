import numpy as np

from .prediction import KerasPredictionModelWithClasses, KerasModelWithClassesConfig
from ..classification.models import ClassificationModel, ClassificationModelConfig, ClassificationModelOutput, ClassificationModelFactory

class KerasClassificationModel(KerasPredictionModelWithClasses[ClassificationModelOutput], ClassificationModel):
    __slots__: tuple

    def get_output(self, predictions: np.ndarray) -> ClassificationModelOutput:
        return predictions[0]

class KerasClassificationModelConfig(KerasModelWithClassesConfig, ClassificationModelConfig):
    pass

KERAS_CLASSIFICATION_MODEL_FACTORY = ClassificationModelFactory[str, KerasClassificationModelConfig](
    name="keras",
    model_cls=KerasClassificationModel,
    model_config_cls=KerasClassificationModelConfig,
    model_config_loader=KerasClassificationModelConfig.from_path,
    default_model_config_input="models/classification"
)