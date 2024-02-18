import os

from . import FileExtractor
from ..prediction.predictor import IPredictionResultWithBoundingBoxes


class FileDetectionExtractor(FileExtractor[IPredictionResultWithBoundingBoxes]):
    def get_path(
        self, extracted_name: str, _: IPredictionResultWithBoundingBoxes
    ) -> str:
        return os.path.join(self.output_dir, extracted_name)
