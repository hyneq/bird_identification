import os
from dataclasses import dataclass
from typing import Sequence, Iterator

import numpy as np

import keras_cv

from ..image_utils import Image

from ..prediction.classes import DictScores

from .prediction import KerasPredictionModelWithClasses, KerasModelWithClassesConfig
from ..detection.models import (
    DetectionModel,
    DetectionModelConfig,
    DetectionModelOutput,
    DetectionModelOutputIter,
    DetectionModelFactory,
)

KerasYOLOv8DetectionObj = tuple[np.ndarray, np.ndarray, np.ndarray]
KerasYOLOv8DetectionModelRawOutput = dict[str, np.ndarray]


class KerasYOLOv8DetectionModelOutputIter(
    DetectionModelOutputIter[KerasYOLOv8DetectionObj, KerasYOLOv8DetectionModelRawOutput]
):
    def __init__(self, raw_output: KerasYOLOv8DetectionModelRawOutput):
        super().__init__(raw_output)

        self.boxes = raw_output["boxes"][0]
        self.confidence = raw_output["confidence"][0]
        self.classes = raw_output["classes"][0]
        self.num_detections = raw_output["num_detections"][0]

        self._i = 0

    def __next__(self):
        if self._i >= self.num_detections:
            raise StopIteration

        i = self._i

        obj = (self.boxes[i], self.confidence[i], self.classes[i])

        self._i = i + 1

        return obj

class KerasYOLOv8DetectionModelOutput(
    DetectionModelOutput[
        KerasYOLOv8DetectionObj,
        KerasYOLOv8DetectionModelRawOutput,
        KerasYOLOv8DetectionModelOutputIter
    ]
):
    iter_cls = KerasYOLOv8DetectionModelOutputIter

    def _filter(self, obj: np.ndarray) -> np.ndarray:
        return obj[obj != -1]

    def get_box(self, obj: KerasYOLOv8DetectionObj) -> Sequence:
        return tuple(self._filter(obj[0]))

    def get_scores(self, obj: KerasYOLOv8DetectionObj) -> DictScores:
        return DictScores(dict(zip(self._filter(obj[2]), self._filter(obj[1]))))


class KerasYOLOv8DetectionModel(
    KerasPredictionModelWithClasses[DetectionModelOutput], DetectionModel
):
    __slots__: tuple

    external_NMS = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # from https://keras.io/guides/keras_cv/object_detection_keras_cv/
        self.resize_input = keras_cv.layers.Resizing(
            640, 640, pad_to_aspect_ratio=True, bounding_box_format="xywh"
        )


    def get_input(self, input: Image) -> np.ndarray:
        return self.resize_input([input])


    def get_output(self, predictions: KerasYOLOv8DetectionModelRawOutput) -> DetectionModelOutput:
        return KerasYOLOv8DetectionModelOutput(predictions)


class KerasYOLOv8DetectionModelConfig(
    KerasModelWithClassesConfig, DetectionModelConfig
):
    @classmethod
    def from_path(cls, path: str):
        return cls(
            model_path=os.path.join(path, "model.keras"),
            classes_path=os.path.join(path, "classes.txt"),
        )


factory = DetectionModelFactory[
    str, KerasYOLOv8DetectionModelConfig
](
    name="keras",
    model_cls=KerasYOLOv8DetectionModel,
    model_config_cls=KerasYOLOv8DetectionModelConfig,
    model_config_loader=KerasYOLOv8DetectionModelConfig.from_path,
    default_model_config_input="models/YOLOv8-pascalvoc",
)
