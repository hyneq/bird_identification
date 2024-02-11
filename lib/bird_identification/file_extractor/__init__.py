import os, glob
from typing import Generic, TypeVar
from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2

from ..image_utils import Image

from ..prediction.predictor import APredictor, IPredictionResultWithBoundingBoxes
from ..extraction.detection_extraction import extract_detection

PredictionResultT = TypeVar("PredictionResultT", bound=IPredictionResultWithBoundingBoxes)

@dataclass
class FileExtractor(ABC, Generic[PredictionResultT]):

    predictor: APredictor[Image, Image, list[PredictionResultT]]

    input_dir: str
    output_dir: str
    filename_template: str = "{source_name}-{i}.jpg"

    def extract(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        i = 0
        input_glob = glob.glob(os.path.join(self.input_dir, "**", "*.*"))
        for path in input_glob:
            name = "-".join(os.path.splitext(os.path.relpath(path, self.input_dir))[0].split(os.path.sep))

            img = cv2.imread(path)

            results = self.predictor.predict(img)

            for result in results:
                extracted_img = extract_detection(img, result)

                extracted_name = self.filename_template.format(source_name=name, i=str(i).zfill(6))

                extracted_path = self.get_path(extracted_name, result)

                cv2.imwrite(extracted_path, extracted_img)

                print("Extracted image from {} to {}".format(path, extracted_path))

                i += 1
    
    @abstractmethod
    def get_path(self, extracted_name: str, result: PredictionResultT) -> str:
        pass
