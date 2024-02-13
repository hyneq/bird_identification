import os,sys
sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__)), os.path.pardir, os.path.pardir))
sys.path.remove(os.path.dirname(__file__))

from bird_identification.cli.detection_classification import DetectionClassificationCLIPart
from bird_identification.cli.prediction_annotation import StreamPredictionAnnotationCLI

def cli_main():
    StreamPredictionAnnotationCLI(DetectionClassificationCLIPart()).run()

if __name__ == "__main__":
    cli_main()