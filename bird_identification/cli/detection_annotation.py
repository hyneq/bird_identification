from ..detection.detector import object_detector_factory
from .prediction import PredictionCLIPart
from .prediction_annotation import StreamPredictionAnnotationCLI

def cli_main():
    StreamPredictionAnnotationCLI(PredictionCLIPart(object_detector_factory)).run()

if __name__ == "__main__":
    cli_main()
