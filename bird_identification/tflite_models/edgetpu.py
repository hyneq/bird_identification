import os
from dataclasses import dataclass

from .prediction import (
    TFLiteModelConfig,
    TFLiteModelWithClassesConfig
)

EDGETPU_DELEGATE = 'libedgetpu.so.1'

@dataclass
class EdgeTPUTFLiteModelConfig(TFLiteModelConfig):

    delegate: str = EDGETPU_DELEGATE

    @classmethod
    def from_path(cls, path: str):
        return cls(model_path=os.path.join(path, "model_edgetpu.tflite"))


@dataclass
class EdgeTPUTFLiteModelWithClassesConfig(EdgeTPUTFLiteModelConfig, TFLiteModelWithClassesConfig):

    @classmethod
    def from_path(cls, path: str):
        return cls(
            model_path=os.path.join(path, "model_edgetpu.tflite"),
            classes_path=os.path.join(path, "classes.txt"),
        )
