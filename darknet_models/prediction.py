from abc import ABC, abstractmethod
from dataclasses import dataclass
from threading import Lock
import os

import cv2
import numpy as np

from prediction.models import PathPredictionModelConfig, ImagePredictionModel, TPredictionModelOutput, Image

@dataclass()
class DarknetPredictionModelConfig(PathPredictionModelConfig):
    config_path: str
    weights_path: str

    @classmethod
    def from_path(cls, path: str):
        return cls(
            classes_path=os.path.join(path, "classes.names"),
            config_path=os.path.join(path, "model.cfg"),
            weights_path=os.path.join(path, "model.weights")
        )

class DarknetPredictionModel(ImagePredictionModel[DarknetPredictionModelConfig, TPredictionModelOutput], ABC):
    __slots__: tuple

    network: any
    layer_names_output: any
    lock: Lock

    def __init__(self, cfg: DarknetPredictionModelConfig):
        self.lock = Lock()

        self.network = network = cv2.dnn.readNetFromDarknet(cfg.config_path, cfg.weights_path)

        # Getting list with names of all layers from YOLO v3 network
        layers_names_all = network.getLayerNames()

        # Getting only output layers' names that we need from YOLO v3 algorithm
        # with function that returns indexes of layers with unconnected outputs
        self.layers_names_output = \
            [layers_names_all[i - 1] for i in network.getUnconnectedOutLayers()]

        super().__init__(cfg)
    
    def predict(self, input: Image) -> TPredictionModelOutput:
        height, width = input.shape[0:2]

        # blob from image
        blob = cv2.dnn.blobFromImage(input, 1 / 255.0, (416, 416),
                                    swapRB=True, crop=False)
        
        with self.lock:
            self.network.setInput(blob)
            raw_output = self.network.forward(self.layers_names_output)

        return self.get_output(raw_output, width, height) # TODO: width, height generalize

    @abstractmethod
    def get_output(self, raw_output: np.ndarray, width: int, height: int) -> TPredictionModelOutput:
        pass