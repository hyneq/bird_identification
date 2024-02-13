if __name__ == "__main__":
    import os,sys
    sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__)), os.path.pardir, os.path.pardir))
    sys.path.remove(os.path.dirname(__file__))

from bird_identification.cli import ArgumentParser, CLIPart, CLIPartPrefixWrapper, Namespace, args_required
from bird_identification.cli.prediction import IPredictionCLIPart, PredictionCLIPart, ImagePredictionCLI
from bird_identification.prediction.predictor import IPredictorFactory
from bird_identification.classification.classifier import image_classifier_factory
from bird_identification.detection.detector import object_detector_factory
from bird_identification.detection_classification import DetectionClassifier

class DetectionClassificationCLIPart(CLIPart, IPredictionCLIPart):

    detection_cli_part: CLIPartPrefixWrapper[IPredictionCLIPart]
    classification_cli_part: CLIPartPrefixWrapper[IPredictionCLIPart]

    def __init__(self,
                 detector_factory: IPredictorFactory=object_detector_factory,
                 classifier_factory: IPredictorFactory=image_classifier_factory
    ):
        super().__init__()

        self.detection_cli_part = CLIPartPrefixWrapper(
            PredictionCLIPart(detector_factory),
            prefix="detection",
            group=True,
            descr="Object detection options")
        self.classification_cli_part = CLIPartPrefixWrapper(
            PredictionCLIPart(classifier_factory),
            "classification",
            group=True,
            descr="Image classification options")
    
    def add_opts(self, parser: ArgumentParser):
        self.detection_cli_part.add_opts(parser)
        self.classification_cli_part.add_opts(parser)
    
    def add_args(self, args: Namespace):
        super().add_args(args)

        self.detection_cli_part.add_args(args)
        self.classification_cli_part.add_args(args)
    
    @args_required
    def get_predictor(self) -> DetectionClassifier:
        return DetectionClassifier(
            detector=self.detection_cli_part.get_predictor(),
            classifier=self.classification_cli_part.get_predictor()
        )

def cli_main():
    ImagePredictionCLI(DetectionClassificationCLIPart()).run()

if __name__ == "__main__":
    cli_main()