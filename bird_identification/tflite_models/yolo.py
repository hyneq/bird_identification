import numpy as np

from ..prediction.classes import VectorScores

from ..detection.models import (
    DetectionModelOutputIter,
    DetectionModelOutput
)


YOLODetectionObj = np.ndarray
YOLODetectionModelRawOutput = np.ndarray


class YOLODetectionModelOutputIter(
    DetectionModelOutputIter[YOLODetectionObj, YOLODetectionModelRawOutput]
):
    __slots__: tuple

    _len: int
    _index: int

    def __init__(self, raw_output: YOLODetectionModelRawOutput):
        super().__init__(raw_output)

        self._len = raw_output.shape[0]
        self._index = 0

    def __next__(self) -> YOLODetectionObj:
        if self._index >= self._len:
            raise StopIteration

        obj = self.raw_output[self._index]

        self._index += 1

        return obj


class YOLODetectionModelOutput(
    DetectionModelOutput[
        YOLODetectionObj,
        YOLODetectionModelRawOutput,
        YOLODetectionModelOutputIter,
    ]
):
    iter_cls = YOLODetectionModelOutputIter

    width: int
    height: int

    def __init__(
        self, raw_output: YOLODetectionModelRawOutput, width: int, height: int
    ):
        super().__init__(raw_output)

        self.width = width
        self.height = height

    def get_box(self, obj: YOLODetectionObj) -> tuple:
        box = obj[0:4] * np.array([self.width, self.height, self.width, self.height])

        x_center, y_center, box_width, box_height = box
        x_min = max(int(x_center - (box_width / 2)), 0)
        y_min = max(int(y_center - (box_height / 2)), 0)
        if x_min + box_width >= self.width:
            box_width = self.width - x_min
        if y_min + box_height >= self.height:
            box_height = self.width - y_min

        return (x_min, y_min, int(box_width), int(box_height))

    def get_scores(self, obj: YOLODetectionObj) -> VectorScores:
        return VectorScores(obj[4:])
