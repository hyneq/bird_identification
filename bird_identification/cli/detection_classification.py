from . import ArgumentParser, CLIPart, CLIPartPrefixWrapper, Namespace, args_required
from .prediction import IPredictionCLIPart, PredictionCLIPart, ImagePredictionCLI
from ..prediction.predictor import IPredictorFactory
from ..classification.classifier import image_classifier_factory
from ..detection.detector import object_detector_factory
from ..detection_classification import DetectionClassifier


class DetectionClassificationCLIPart(CLIPart, IPredictionCLIPart):
    detection_cli_part: CLIPartPrefixWrapper[IPredictionCLIPart]
    classification_cli_part: CLIPartPrefixWrapper[IPredictionCLIPart]

    def __init__(
        self,
        detector_factory: IPredictorFactory = object_detector_factory,
        classifier_factory: IPredictorFactory = image_classifier_factory,
    ):
        super().__init__()

        self.detection_cli_part = CLIPartPrefixWrapper(
            PredictionCLIPart(detector_factory),
            prefix="detection",
            group=True,
            descr="Object detection options",
        )
        self.classification_cli_part = CLIPartPrefixWrapper(
            PredictionCLIPart(classifier_factory),
            "classification",
            group=True,
            descr="Image classification options",
        )

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
            classifier=self.classification_cli_part.get_predictor(),
        )


def cli_main():
    ImagePredictionCLI(DetectionClassificationCLIPart()).run()


if __name__ == "__main__":
    cli_main()
