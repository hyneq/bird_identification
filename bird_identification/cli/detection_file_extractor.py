from . import Optional

from .prediction import IPredictionCLIPart, PredictionCLIPart
from .file_extractor import FileExtractionCLI

from ..detection.detector import object_detector_factory

from ..file_extractor.detection import FileDetectionExtractor


class FileDetectionExtractionCLI(FileExtractionCLI):
    extractor_cls = FileDetectionExtractor

    def __init__(
        self, *args, prediction_cli_part: Optional[IPredictionCLIPart] = None, **kwargs
    ):
        if not prediction_cli_part:
            prediction_cli_part = PredictionCLIPart(object_detector_factory)

        super().__init__(prediction_cli_part, *args, **kwargs)


def cli_main():
    FileDetectionExtractionCLI().run()


if __name__ == "__main__":
    cli_main()
