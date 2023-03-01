import numpy as np

from prediction.models import PredictionModelConfigWithCls
from detection.models import DetectionModelType
from .prediction import DarknetPredictionModel, DarknetPredictionModelConfig
from YOLOv3_models.detection import YOLOv3DetectionModel, YOLOv3DetectionModelConfig, YOLOv3DetectionModelOutput, YOLOv3DetectionModelRawOutput

class DarknetYOLOv3DetectionModel(DarknetPredictionModel[YOLOv3DetectionModelOutput], YOLOv3DetectionModel):
    
    def get_output(self, raw_output: YOLOv3DetectionModelRawOutput, width: int, height: int) -> YOLOv3DetectionModelOutput:
        return YOLOv3DetectionModelOutput(raw_output, width, height)

class DarknetYOLOv3DetectionModelConfig(DarknetPredictionModelConfig, YOLOv3DetectionModelConfig, PredictionModelConfigWithCls[DarknetYOLOv3DetectionModel]):
    model_cls = DarknetYOLOv3DetectionModel

DARKNET_YOLOV3_DETECTION_MODEL_TYPE = DetectionModelType(
    name="darknet-YOLOv3",
    model_cls=DarknetYOLOv3DetectionModel,
    model_config_cls=DarknetPredictionModelConfig
)