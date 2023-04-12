import numpy as np
import os

from detection.models import DetectionModelConfig, DetectionModelFactory
from .prediction import DarknetPredictionModel, DarknetPredictionModelConfig
from YOLOv3_models.detection import YOLOv3DetectionModel, YOLOv3DetectionModelConfig, YOLOv3DetectionModelOutput, YOLOv3DetectionModelRawOutput

class DarknetYOLOv3DetectionModel(DarknetPredictionModel[YOLOv3DetectionModelOutput], YOLOv3DetectionModel):
    
    def get_output(self, raw_output: YOLOv3DetectionModelRawOutput, width: int, height: int) -> YOLOv3DetectionModelOutput:
        return YOLOv3DetectionModelOutput(raw_output, width, height)

class DarknetYOLOv3DetectionModelConfig(DarknetPredictionModelConfig, YOLOv3DetectionModelConfig, DetectionModelConfig):

    @classmethod
    def from_path(cls, path: str):
        return cls(
            classes_path=os.path.join(path, "coco.names"),
            config_path=os.path.join(path, "yolov3.cfg"),
            weights_path=os.path.join(path, "yolov3.weights")
        )

DARKNET_YOLOV3_DETECTION_MODEL_FACTORY = DetectionModelFactory[DarknetYOLOv3DetectionModel, DarknetYOLOv3DetectionModelConfig](
    name="darknet-YOLOv3",
    model_cls=DarknetYOLOv3DetectionModel,
    model_config_cls=DarknetYOLOv3DetectionModelConfig,
    model_config_loader=DarknetYOLOv3DetectionModelConfig.from_path,
    default_model_config_input="models/YOLOv3-COCO"
)