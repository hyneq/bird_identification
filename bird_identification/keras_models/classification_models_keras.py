import numpy as np
import cv2

from ..image_utils import Image

from ..prediction.classes import VectorScores

from .prediction import KerasPredictionModelWithClasses, KerasModelWithClassesConfig
from ..classification.models import (
    ClassificationModel,
    ClassificationModelConfig,
    ClassificationModelOutput,
    ClassificationModelFactory,
)


class KerasClassificationModel(
    KerasPredictionModelWithClasses[ClassificationModelOutput], ClassificationModel
):
    __slots__: tuple

    def get_input(self, input: Image) -> np.ndarray:
        blob: np.ndarray = cv2.dnn.blobFromImage(input, size=(224, 224), swapRB=True)

        blob = np.moveaxis(
            blob, (1, 2, 3), (3, 1, 2)
        )  # Making the color channel the last dimension instead of the first, in order to match model input shape

        return blob

    def get_output(self, predictions: np.ndarray) -> ClassificationModelOutput:
        return VectorScores(predictions[0])


class KerasClassificationModelConfig(
    KerasModelWithClassesConfig, ClassificationModelConfig
):
    pass


factory = ClassificationModelFactory[
    str, KerasClassificationModelConfig
](
    name="keras",
    model_cls=KerasClassificationModel,
    model_config_cls=KerasClassificationModelConfig,
    model_config_loader=KerasClassificationModelConfig.from_path,
    default_model_config_input="models/classification",
)
