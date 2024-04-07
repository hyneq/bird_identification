from dataclasses import dataclass

from ..detection.models import DetectionModelFactory

from .edgetpu import EdgeTPUTFLiteModelWithClassesConfig
from .detection_models_tflite_YOLO import TFLiteYOLODetectionModel, TFLiteYOLODetectionModelConfig

@dataclass
class EdgeTPUTFLiteYOLODetectionModelConfig(
    EdgeTPUTFLiteModelWithClassesConfig,
    TFLiteYOLODetectionModelConfig,
):
    pass


factory = DetectionModelFactory[
    str, EdgeTPUTFLiteYOLODetectionModelConfig
](
    name="tflite_edgetpu",
    model_cls=TFLiteYOLODetectionModel,
    model_config_cls=EdgeTPUTFLiteYOLODetectionModelConfig,
    model_config_loader=EdgeTPUTFLiteYOLODetectionModelConfig.from_path,
    default_model_config_input="models/YOLO-COCO",
)
