from ..darknet_models.detection import DARKNET_YOLOV3_DETECTION_MODEL_FACTORY, DARKNET_YOLOV4_DETECTION_MODEL_FACTORY

MODEL_FACTORIES = [
    DARKNET_YOLOV3_DETECTION_MODEL_FACTORY,
    DARKNET_YOLOV4_DETECTION_MODEL_FACTORY
]

DEFAULT_MODEL_FACTORY = DARKNET_YOLOV3_DETECTION_MODEL_FACTORY.name

DEFAULT_NMS_THRESHOLD = 0.5