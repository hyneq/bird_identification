import time
from time_utils import sleep_interval

from . import PredictionRunner

class IntervalPredictionRunner(PredictionRunner):
    __slots__: tuple

    interval: float

    def __init__(self, interval: float=1.0):
        self.interval = interval

    def _run(self):
        while self.running:
            start_time = time.time()
            self.prediction_stream_processor.predict()
            sleep_interval(start_time, self.interval)
    

DEFAULT_PREDICTION_RUNNER = IntervalPredictionRunner()
