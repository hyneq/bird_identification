from typing import Optional, Callable

from argparse import ArgumentParser, Namespace
from enum_actions import enum_action

from prediction import predictor, classes
from . import args_required, CLIPart, CLIWithParts

class PredictionCLIPart(CLIPart):

    predictor_factory: predictor.IPredictorFactory

    def __init__(self, predictor_factory, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.predictor_factory = predictor_factory

    def add_opts(self, parser: ArgumentParser):
        self.add_cs_opts(parser)
        self.add_model_opts(parser)
    
    def add_cs_opts(self, parser: ArgumentParser):
        parser.add_argument(
            "-m", "--mode", "--classification-mode",
            dest="mode",
            action=enum_action(classes.ClassificationMode),
            help="The classification mode to use"
        )

        parser.add_argument(
            "-f", "--min-confidence",
            dest="min_confidence",
            type=int,
            choices=range(0,100),
            help="Minimum confidence to take into account"
        )

        parser.add_argument(
            "-c", "--class",
            dest="classes",
            type=str,
            nargs='*',
            help="class(es) to find"
        )

    def add_model_opts(self, parser: ArgumentParser):

        parser.add_argument(
            "--model-path",
            dest="model_path"
        )

        parser.add_argument(
            "--model-type",
            dest="model_type",
            default=self.predictor_factory.get_model_factory().default_factory,
            choices=self.predictor_factory.get_model_factory().get_factory_names(),
            help="model types that can be used"
        )
    
    @args_required
    def get_predictor(self) -> predictor.IPredictor:
        return self.predictor_factory.get_predictor(
            model_path=self.args.model_path,
            model_type=self.args.model_type,
            mode=self.args.mode,
            min_confidence=self.args.min_confidence,
            classes=self.args.classes
        )

class PredictionCLI(CLIWithParts):

    prediction_cli_part: PredictionCLIPart

    def __init__(self, predictor_factory: predictor.IPredictorFactory[predictor.FileImagePredictor, predictor.TPredictionModel, predictor.TPredictorConfig, predictor.TPathPredictionModelConfig]):
        self.prediction_cli_part = prediction_cli_part = PredictionCLIPart(predictor_factory)
        super().__init__(parts=[prediction_cli_part])
    
    def init_parser(self):
        super().init_parser()

        self.parser.add_argument("image", nargs="+")

    
    def run(self):
        super().run()

        predictor = self.prediction_cli_part.get_predictor()

        print([predictor.predict(image) for image in self.args.image])
