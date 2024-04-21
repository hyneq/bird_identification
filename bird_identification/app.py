"""
The object recognition and logging application

Composes other object to get the desired functionality
"""

from .prediction.predictor import APredictor
from .streams import IVideoInStream, IVideoOutStream
from .tracking.tracker import ITracker
from .prediction_annotation import ImagePredictionStreamRunner
from .tracking.prediction_callback import TrackerPredictionCallback

class App:
    predictor: APredictor

    in_stream: IVideoInStream
    out_stream: IVideoOutStream

    tracker: ITracker

    prediction_stream_runner: ImagePredictionStreamRunner

    def __init__(self,
        predictor: APredictor,
        in_stream: IVideoInStream,
        out_stream: IVideoOutStream,
        tracker: ITracker
    ):
        self.predictor = predictor
        self.in_stream = in_stream
        self.out_stream = out_stream
        self.tracker = tracker

        callbacks = [TrackerPredictionCallback(tracker)]

        self.prediction_stream_runner = ImagePredictionStreamRunner(
            predictor=predictor, in_stream=in_stream, out_stream=out_stream, callbacks=callbacks
        )


    def run(self):
        self.prediction_stream_runner.run()
        self.tracker.close()
