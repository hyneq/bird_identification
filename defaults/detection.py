import os

from detection.models import DetectionModelConfig, DetectionModel

from darknet_models.detection import DARKNET_YOLOV3_DETECTION_MODEL_FACTORY

DEFAULT_MODEL_CLS: type[DetectionModel]

DEFAULT_MODEL_CONFIG: DetectionModelConfig

MODEL_FACTORIES = [
    DARKNET_YOLOV3_DETECTION_MODEL_FACTORY
]

DEFAULT_MODEL_FACTORY = DARKNET_YOLOV3_DETECTION_MODEL_FACTORY.name

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__),os.path.pardir,"models","YOLOv3-COCO")

DEFAULT_PROBABILITY_MINIMUM = 0.5
DEFAULT_NMS_THRESHOLD = 0.5