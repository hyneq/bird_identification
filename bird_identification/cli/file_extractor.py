from . import CLIWithParts
from .prediction import IPredictionCLIPart

from ..file_extractor import FileExtractor


class FileExtractionCLI(CLIWithParts):
    prediction_cli_part: IPredictionCLIPart

    extractor_cls: type[FileExtractor]

    def __init__(self, prediction_cli_part: IPredictionCLIPart, *args, **kwargs):
        self.prediction_cli_part = prediction_cli_part

        super().__init__(*args, parts=[prediction_cli_part], **kwargs)

    def init_parser(self):
        super().init_parser()

        self.parser.add_argument(
            "input_dir", help="Path to directory containing source files"
        )
        self.parser.add_argument("output_dir", help="Path to output directory")

    def run(self):
        super().run()

        extractor = self.extractor_cls(
            predictor=self.prediction_cli_part.get_predictor(),
            input_dir=self.args.input_dir,
            output_dir=self.args.output_dir,
        )

        extractor.extract()
