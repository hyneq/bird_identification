import cv2
import numpy as np

from ..image_utils import Image

from ..detection.models import DetectionModel, DetectionModelConfig, DetectionModelFactory

from .prediction import TFLiteModelWithClassesConfig, TFLitePredictionModelWithClasses

from .yolo import YOLOv3DetectionModelOutput


class TFLiteYOLOv3DetectionModel(
    TFLitePredictionModelWithClasses[Image, YOLOv3DetectionModelOutput],
    DetectionModel
):

    def set_input(self, input: Image):
        input_shape = self.input_details[1]['shape'][1:3]
        blob = cv2.dnn.blobFromImage(
            input, 1 / 255.0, size=input_shape, swapRB=True, crop=False
        )
        blob = np.moveaxis(blob, 1, 3)
        self.interpreter.set_tensor(self.input_details[1]['index'], blob)
        self.interpreter.set_tensor(self.input_details[0]["index"], [[input.shape[1], input.shape[0]]])


    def get_output(self, _: Image) -> YOLOv3DetectionModelOutput:
        boxes = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        scores = self.interpreter.get_tensor(self.output_details[1]["index"])[0]

        return YOLOv3DetectionModelOutput(boxes, scores)


class TFLiteYOLOv3DetectionModelConfig(
    TFLiteModelWithClassesConfig, DetectionModelConfig
):
    pass


factory = DetectionModelFactory[
    str, TFLiteYOLOv3DetectionModelConfig
](
    name="tflite-YOLOv3",
    model_cls=TFLiteYOLOv3DetectionModel,
    model_config_cls=TFLiteYOLOv3DetectionModelConfig,
    model_config_loader=TFLiteYOLOv3DetectionModelConfig.from_path,
    default_model_config_input="models/YOLOv3-COCO",
)
