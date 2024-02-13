import os,sys
sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__)), os.path.pardir, os.path.pardir))
if os.path.realpath(os.path.dirname(__file__)) in sys.path: sys.path.remove(os.path.realpath(os.path.dirname(__file__)))

from bird_identification.cli import Optional

from bird_identification.cli.prediction import IPredictionCLIPart
from bird_identification.cli.detection_classification import DetectionClassificationCLIPart
from bird_identification.cli.file_extractor import FileExtractionCLI

from bird_identification.file_extractor.detection_classification import FileDetectionClassificationExtractor

class FileDetectionCLIFileExtractionCLI(FileExtractionCLI):

    extractor_cls = FileDetectionClassificationExtractor

    def __init__(self, *args, prediction_cli_part: Optional[IPredictionCLIPart]=None, **kwargs):
        if not prediction_cli_part:
            prediction_cli_part = DetectionClassificationCLIPart()
        
        super().__init__(prediction_cli_part, *args, **kwargs)

def cli_main():
    FileDetectionCLIFileExtractionCLI().run()

if __name__ == "__main__":
    cli_main()