import os

from detection.models import DetectionModelConfig, DetectionModel

from darknet_models.detection import DarknetYOLOv3DetectionModel, DarknetYOLOv3DetectionModelConfig

DEFAULT_MODEL_CLS: type[DetectionModel]

DEFAULT_MODEL_CONFIG: DetectionModelConfig

DEFAULT_MODEL_CLS = DarknetYOLOv3DetectionModel

YOLO_COCO_PATH = os.path.join(os.path.dirname(__file__),os.path.pardir,"models","YOLOv3-COCO")

DEFAULT_MODEL_CONFIG = DarknetYOLOv3DetectionModelConfig(
    classes_path=os.path.join(YOLO_COCO_PATH, "coco.names"),
    config_path=os.path.join(YOLO_COCO_PATH, "yolov3.cfg"),
    weights_path=os.path.join(YOLO_COCO_PATH, "yolov3.weights")
)

DEFAULT_PROBABILITY_MINIMUM = 0.5
DEFAULT_NMS_THRESHOLD = 0.5