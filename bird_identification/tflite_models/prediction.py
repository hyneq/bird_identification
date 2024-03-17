"""
Inspired by https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_python
"""

import os
from dataclasses import dataclass
from abc import abstractmethod
from threading import Lock

from typing import Any

import numpy as np
import cv2
from tensorflow import lite as tflite

from ..prediction.models import (
    PredictionModel,
    PredictionModelWithClasses,
    PredictionModelConfig,
    PredictionModelWithClassesConfig,
    PredictionModelInputT,
    PredictionModelOutputT,
)
from ..image_utils import Image, Size


@dataclass
class TFLiteModelConfig(PredictionModelConfig):
    model_path: str

    @classmethod
    def from_path(cls, path: str):
        return cls(model_path=os.path.join(path, "model.tflite"))


@dataclass
class TFLiteModelWithClassesConfig(TFLiteModelConfig, PredictionModelWithClassesConfig):
    @classmethod
    def from_path(cls, path: str):
        return cls(
            model_path=os.path.join(path, "model.tflite"),
            classes_path=os.path.join(path, "classes.txt"),
        )


class TFLitePredictionModel(
    PredictionModel[TFLiteModelConfig, PredictionModelInputT, PredictionModelOutputT]
):
    __slots__: tuple

    interpreter: tflite.Interpreter
    input_details: list[dict[str, Any]]
    output_details: list[dict[str, Any]]
    lock: Lock

    def __init__(self, cfg: TFLiteModelConfig):
        super().__init__(cfg)

        self.lock = Lock()
        self.interpreter = interpreter = tflite.Interpreter(model_path=cfg.model_path)
        self.input_details = interpreter.get_input_details()
        self.output_details = interpreter.get_output_details()

        self.interpreter.allocate_tensors()


    def predict(self, input: PredictionModelInputT) -> PredictionModelOutputT:
        with self.lock:
            self.set_input(input)
            self.interpreter.invoke()
            return self.get_output(input)


    @abstractmethod
    def set_input(self, input: PredictionModelInputT):
        pass

    @abstractmethod
    def get_output(self, input: PredictionModelInputT) -> PredictionModelOutputT:
        pass


class TFLitePredictionModelWithClasses(
    TFLitePredictionModel[PredictionModelInputT, PredictionModelOutputT],
    PredictionModelWithClasses[
        TFLiteModelWithClassesConfig, PredictionModelInputT, PredictionModelOutputT
    ]
):
    pass
