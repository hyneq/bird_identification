import numpy as np
import os

from ..detection.models import DetectionModelFactory

from .detection_models_darknet_YOLOv3 import DarknetYOLOv3DetectionModel, DarknetYOLOv3DetectionModelConfig

class DarknetYOLOv4DetectionModelConfig(DarknetYOLOv3DetectionModelConfig):
    @classmethod
    def from_path(cls, path: str):
        return cls(
            classes_path=os.path.join(path, "coco.names"),
            config_path=os.path.join(path, "yolov4.cfg"),
            weights_path=os.path.join(path, "yolov4.weights"),
        )

DARKNET_YOLOV4_DETECTION_MODEL_FACTORY = DetectionModelFactory[
    DarknetYOLOv3DetectionModel, DarknetYOLOv4DetectionModelConfig
](
    name="darknet-YOLOv4",
    model_cls=DarknetYOLOv3DetectionModel,
    model_config_cls=DarknetYOLOv4DetectionModelConfig,
    model_config_loader=DarknetYOLOv4DetectionModelConfig.from_path,
    default_model_config_input="models/YOLOv4-COCO",
)
