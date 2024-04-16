from bird_identification.prediction.predictor import IPredictionResultWithClassesAndBoundingBoxes
from ..prediction_annotation import PredictionCallback

from .tracker import ITracker

class TrackerPredictionCallback(PredictionCallback):

    tracker: ITracker[list[IPredictionResultWithClassesAndBoundingBoxes]]

    def __init__(self, tracker: ITracker[list[IPredictionResultWithClassesAndBoundingBoxes]]):
        self.tracker = tracker


    def __call__(self, results: list[IPredictionResultWithClassesAndBoundingBoxes]):
        self.tracker.update(results)
