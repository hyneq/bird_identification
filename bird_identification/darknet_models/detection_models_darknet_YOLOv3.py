import numpy as np
import os

from dataclasses import dataclass
from threading import Lock

import cv2
import numpy as np

from ..image_utils import Image

from ..prediction.models import PredictionModelWithClasses

from ..detection.models import DetectionModel, DetectionModelFactory

from ..YOLOv3_models.detection import (
    YOLOv3DetectionModel,
    YOLOv3DetectionModelConfig,
    YOLOv3DetectionModelOutput,
    YOLOv3DetectionModelRawOutput,
)

@dataclass()
class DarknetYOLOv3DetectionModelConfig(YOLOv3DetectionModelConfig):
    config_path: str
    weights_path: str

    @classmethod
    def from_path(cls, path: str):
        return cls(
            classes_path=os.path.join(path, "coco.names"),
            config_path=os.path.join(path, "yolov3.cfg"),
            weights_path=os.path.join(path, "yolov3.weights"),
        )


class DarknetYOLOv3DetectionModel(PredictionModelWithClasses, YOLOv3DetectionModel):
    __slots__: tuple

    network: any
    layer_names_output: any
    lock: Lock

    def __init__(self, cfg: DarknetYOLOv3DetectionModelConfig):
        self.lock = Lock()

        self.network = network = cv2.dnn.readNetFromDarknet(
            cfg.config_path, cfg.weights_path
        )

        # Getting list with names of all layers from YOLO v3 network
        layers_names_all = network.getLayerNames()

        # Getting only output layers' names that we need from YOLO v3 algorithm
        # with function that returns indexes of layers with unconnected outputs
        self.layers_names_output = [
            layers_names_all[i - 1] for i in network.getUnconnectedOutLayers()
        ]

        super().__init__(cfg)

    def predict(self, input: Image) -> YOLOv3DetectionModelOutput:
        height, width = input.shape[0:2]

        # blob from image
        blob = cv2.dnn.blobFromImage(
            input, 1 / 255.0, (416, 416), swapRB=True, crop=False
        )

        with self.lock:
            self.network.setInput(blob)
            raw_output = self.network.forward(self.layers_names_output)

        return self.get_output(
            raw_output, width, height
        )  # TODO: width, height generalize

    def get_output(
        self, raw_output: YOLOv3DetectionModelRawOutput, width: int, height: int
    ) -> YOLOv3DetectionModelOutput:
        return YOLOv3DetectionModelOutput(raw_output, width, height)


factory = DetectionModelFactory[
    DarknetYOLOv3DetectionModel, DarknetYOLOv3DetectionModelConfig
](
    name="darknet-YOLOv3",
    model_cls=DarknetYOLOv3DetectionModel,
    model_config_cls=DarknetYOLOv3DetectionModelConfig,
    model_config_loader=DarknetYOLOv3DetectionModelConfig.from_path,
    default_model_config_input="models/YOLOv3-COCO",
)
