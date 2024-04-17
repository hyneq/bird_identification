"""
The CLI representing the bird identification application
"""

from typing import Optional

from . import CLIWithParts
from .streams import IStreamPairCLIPart, MultiPathVideoStreamPairCLIPart
from .prediction import IPredictionCLIPart
from .detection_classification import DetectionClassificationCLIPart
from .tracking import ITrackerCLIPart, MultiLoggingTrackingLogicCLIPart

from ..prediction_annotation import ImagePredictionStreamRunner
from ..tracking.prediction_callback import TrackerPredictionCallback

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
        self.tracking_cli_part = tracking_cli_part or MultiLoggingTrackingLogicCLIPart()

        super().__init__(parts=[
            self.stream_cli_part,
            self.prediction_cli_part,
            self.tracking_cli_part
        ])

    def get_runner(self):
        (in_stream, out_stream) = self.stream_cli_part.get_stream_pair()

        predictor = self.prediction_cli_part.get_predictor()

        callbacks = [TrackerPredictionCallback(self.tracking_cli_part.get_tracker())]

        return ImagePredictionStreamRunner(
            predictor=predictor, in_stream=in_stream, out_stream=out_stream, callbacks=callbacks
        )

    def run(self):
        super().run()

        runner = self.get_runner()

        runner.run()

def cli_main():
    AppCLI().run()
