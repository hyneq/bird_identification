import cv2
import numpy as np

from ..image_utils import Image

from ..detection.models import DetectionModel, DetectionModelConfig, DetectionModelFactory

from .prediction import TFLiteModelWithClassesConfig, TFLitePredictionModelWithClasses

from .yolo import YOLODetectionModelOutput


class TFLiteYOLODetectionModel(
    TFLitePredictionModelWithClasses[Image, YOLODetectionModelOutput],
    DetectionModel
):

    def set_input(self, input: Image):
        input_shape = self.input_details[0]['shape'][1:3]
        input_dtype = self.input_details[0]['dtype']
        blob = cv2.dnn.blobFromImage(
            input, 1 / 255.0, size=input_shape, swapRB=True, crop=False
        ).astype(input_dtype)
        blob = np.moveaxis(blob, 1, 3)
        self.interpreter.set_tensor(self.input_details[0]['index'], blob)


    def get_output(self, input: Image) -> YOLODetectionModelOutput:
        raw_output = self.interpreter.get_tensor(self.output_details[0]["index"])[0].transpose()
        height, width = input.shape[0:2]


        return YOLODetectionModelOutput(raw_output, width, height)


class TFLiteYOLODetectionModelConfig(
    TFLiteModelWithClassesConfig, DetectionModelConfig
):
    pass


factory = DetectionModelFactory[
    str, TFLiteYOLODetectionModelConfig
](
    name="tflite-YOLO",
    model_cls=TFLiteYOLODetectionModel,
    model_config_cls=TFLiteYOLODetectionModelConfig,
    model_config_loader=TFLiteYOLODetectionModelConfig.from_path,
    default_model_config_input="models/YOLO-COCO",
)
