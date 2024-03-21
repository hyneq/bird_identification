from typing import Iterator

import numpy as np

from ..detection.models import (
    DetectionModelOutput,
    DetectionModelOutputIter
)


DetectionObj = int
DetectionRawOutput = tuple[np.ndarray, np.ndarray]


class YOLOv3DetectionModelOutputIter(
    DetectionModelOutputIter[DetectionObj, DetectionRawOutput]
):
    def __init__(self, raw_output: DetectionRawOutput):
        self._range_iterator = iter(range(len(raw_output[0])))
    
    def __next__(self):
        return next(self._range_iterator)


class YOLOv3DetectionModelOutput(
    DetectionModelOutput[
        DetectionObj,
        DetectionRawOutput,
        YOLOv3DetectionModelOutputIter
    ]
):
    iter_cls = YOLOv3DetectionModelOutputIter

    boxes: np.ndarray
    scores: np.ndarray

    def __init__(self, boxes: np.ndarray, scores: np.ndarray):
        super().__init__((boxes, scores))

        self.boxes = boxes
        self.scores = scores


    def get_box(self, obj: DetectionObj) -> tuple:
        x_min, y_min, x_max, y_max = self.boxes[obj]

        return (round(x_min), round(y_min), round(x_max - x_min), round(y_max - y_min))


    def get_scores(self, obj: DetectionObj) -> np.ndarray:
        return self.scores[obj]
