from .detection_classification import DetectionClassificationCLIPart
from .prediction_annotation import StreamPredictionAnnotationCLI

def cli_main():
    StreamPredictionAnnotationCLI(DetectionClassificationCLIPart()).run()

if __name__ == "__main__":
    cli_main()