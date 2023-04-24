from abc import abstractmethod

import numpy as np

from image_utils import Image

from detection.models import DetectionModelConfig, DetectionModelOutputIter, DetectionModelOutput, DetectionModel

class YOLOv3DetectionModelConfig(DetectionModelConfig):
    pass

YOLOv3DetectionObj = np.ndarray
YOLOv3DetectionModelRawOutput = tuple[np.ndarray, np.ndarray, np.ndarray]

class YOLOv3DetectionModelOutputIter(DetectionModelOutputIter[YOLOv3DetectionObj, YOLOv3DetectionModelRawOutput]):
    __slots__: tuple

    _layer_len: int
    _layer_index: int

    _object_len: int
    _object_index: int


    def __init__(self, raw_output: YOLOv3DetectionModelRawOutput):
        super().__init__(raw_output)

        self._layer_len = len(self.raw_output)
        self._layer_index = 0
        self._object_len = raw_output[0].shape[0]
        self._object_index = 0
    
    def __next__(self) -> YOLOv3DetectionObj:
        if self._layer_index < self._layer_len:
            obj = self.raw_output[self._layer_index][self._object_index]

            self._object_index += 1
            if self._object_index == self._object_len:
                self._layer_index += 1
                if self._layer_index != self._layer_len: 
                    self._object_len = self.raw_output[self._layer_index].shape[0]
                
                self._object_index = 0
        
            return obj
        
        raise StopIteration

class YOLOv3DetectionModelOutput(DetectionModelOutput[YOLOv3DetectionObj, YOLOv3DetectionModelRawOutput, YOLOv3DetectionModelOutputIter]):
    
    iter_cls = YOLOv3DetectionModelOutputIter

    width: int
    height: int

    def __init__(self, raw_output: YOLOv3DetectionModelRawOutput, width: int, height: int):
        super().__init__(raw_output)

        self.width = width
        self.height = height

    def get_box(self, obj: YOLOv3DetectionObj) -> tuple:
        box = obj[0:4] * np.array([self.width, self.height, self.width, self.height])

        # From YOLO data format, we can get top left corner coordinates
        # that are x_min and y_min
        x_center, y_center, box_width, box_height = box
        x_min = int(x_center - (box_width / 2))
        y_min = int(y_center - (box_height / 2))

        return (x_min, y_min, int(box_width), int(box_height))
    
    def get_scores(self, obj: YOLOv3DetectionObj) -> np.ndarray:
        return obj[5:]

class YOLOv3DetectionModel(DetectionModel):
    __slots__: tuple

    @abstractmethod
    def predict(self, input: Image) -> YOLOv3DetectionModelOutput:
        pass