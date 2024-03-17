import cv2
import numpy as np

from ..image_utils import Image

from ..classification.models import (
    ClassificationModel,
    ClassificationModelOutput,
    ClassificationModelConfig,
    ClassificationModelFactory,
)

from .prediction import TFLiteModelWithClassesConfig, TFLitePredictionModelWithClasses


class TFLiteClassificationModel(
    TFLitePredictionModelWithClasses[Image, ClassificationModelOutput],
    ClassificationModel
):

    def set_input(self, input: Image):
        input_shape = self.input_details[0]['shape'][1:3]
        blob = cv2.dnn.blobFromImage(input, size=input_shape, swapRB=True)
        blob = np.moveaxis(blob, 1, 3)
        self.interpreter.set_tensor(self.input_details[0]['index'], blob)


    def get_output(self, _: Image) -> ClassificationModelOutput:
        return self.interpreter.get_tensor(self.output_details[0]["index"])[0]


class TFLiteClassificationModelConfig(
    TFLiteModelWithClassesConfig, ClassificationModelConfig
):
    pass


factory = ClassificationModelFactory[
    str, TFLiteClassificationModelConfig
](
    name="tflite",
    model_cls=TFLiteClassificationModel,
    model_config_cls=TFLiteClassificationModelConfig,
    model_config_loader=TFLiteClassificationModelConfig.from_path,
    default_model_config_input="models/classification",
)
