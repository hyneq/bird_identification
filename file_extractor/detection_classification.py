import os

from file_extractor import FileExtractor
from prediction.predictor import IPredictionResultWithClassesAndBoundingBoxes

class FileDetectionClassificationExtractor(FileExtractor[IPredictionResultWithClassesAndBoundingBoxes]):

    def get_path(self, extracted_name: str, result: IPredictionResultWithClassesAndBoundingBoxes) -> str:
        class_name = result.class_name

        if not class_name:
            class_name = "__not_recognized__"
        
        class_path = os.path.join(self.output_dir, class_name)

        if not os.path.exists(class_path):
            os.mkdir(class_path)
        
        return os.path.join(class_path, extracted_name)