"""
The CLI representing the bird identification application
"""

from typing import Optional

from ..app import App

from . import CLIWithParts
from .streams import IStreamPairCLIPart, MultiPathVideoStreamPairCLIPart
from .prediction import IPredictionCLIPart
from .detection_classification import DetectionClassificationCLIPart
from .tracking import ITrackerCLIPart, MultiLoggingTrackerCLIPart

class AppCLI(CLIWithParts):
    prediction_cli_part: IPredictionCLIPart
    stream_cli_part: IStreamPairCLIPart
    tracking_cli_part: ITrackerCLIPart

    def __init__(
        self,
        prediction_cli_part: Optional[IPredictionCLIPart] = None,
        stream_cli_part: Optional[IStreamPairCLIPart] = None,
        tracking_cli_part: Optional[ITrackerCLIPart] = None
    ):
        self.stream_cli_part = stream_cli_part or MultiPathVideoStreamPairCLIPart()
        self.prediction_cli_part = prediction_cli_part or DetectionClassificationCLIPart()
        self.tracking_cli_part = tracking_cli_part or MultiLoggingTrackerCLIPart()

        super().__init__(parts=[
            self.stream_cli_part,
            self.prediction_cli_part,
            self.tracking_cli_part
        ])

    def get_app(self):
        predictor = self.prediction_cli_part.get_predictor()

        (in_stream, out_stream) = self.stream_cli_part.get_stream_pair()

        tracker = self.tracking_cli_part.get_tracker()

        return App(
            predictor,
            in_stream,
            out_stream,
            tracker
        )

    def run(self):
        super().run()

        app = self.get_app()

        app.run()

def cli_main():
    AppCLI().run()
