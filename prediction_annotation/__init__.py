from typing import Optional, Callable
from threading import Thread
from abc import ABC, abstractmethod

from image_utils import Image
from streams import IInStream, IOutStream
from streams.stream_processor import MultiFrameProcessor, FrameCache, SameTypeStreamProcessor
from prediction.predictor import APredictor, IPredictionResultWithClassesAndBoundingBoxes
from annotation import IAnnotation, MultiAnnotation, RectangleWithTextAnnotation
from annotation.stream_annotator import StreamAnnotator

ImageClassAndBoundingBoxPredictor = APredictor[Image, Image, list[IPredictionResultWithClassesAndBoundingBoxes]]

def get_prediction_annotation(results: list[IPredictionResultWithClassesAndBoundingBoxes]) -> MultiAnnotation:
        annotations: list[IAnnotation] = []
        for result in results:
            annotations.append(RectangleWithTextAnnotation(
                result.bounding_box,
                "{} {:.2f}%".format(result.class_name, result.confidence*100) if result.class_name is not None else "not recognized",
                (3, 186, 252)
            ))
        
        return MultiAnnotation(annotations)

class ImagePredictionStreamProcessor:
    __slots__: tuple
    
    predictor: ImageClassAndBoundingBoxPredictor
    annotator: StreamAnnotator
    cache: FrameCache[Image]
    frame_processor: MultiFrameProcessor
    
    def __init__(self,
                 predictor: ImageClassAndBoundingBoxPredictor,
        ):
        self.predictor = predictor
        self.annotator = StreamAnnotator()
        self.cache = FrameCache()
        self.frame_processor = MultiFrameProcessor([self.cache, self.annotator])
    
    def predict(self):
        if self.cache.cached is not None:
            self.set_annotations(self.predictor.predict(self.cache.cached))
    
    def set_annotations(self, results: list[IPredictionResultWithClassesAndBoundingBoxes]):
        self.annotator.set_annotation(get_prediction_annotation(results))

class PredictionRunner(ABC):
    __slots__: tuple

    prediction_stream_processor: Optional[ImagePredictionStreamProcessor]
    running: bool = False
    
    def run(self):
        if self.running:
            raise ValueError("Prediction runner already running")
        
        if not self.prediction_stream_processor:
            raise ValueError("Cannot start prediction runner as no prediction stream processor present")

        self.running = True
        self._run()

    @abstractmethod
    def _run(self):
        pass

    def stop(self):
        self.running = False

from .prediction_runners import DEFAULT_PREDICTION_RUNNER

class ImagePredictionStreamRunner:

    prediction_stream_processor: ImagePredictionStreamProcessor
    prediction_runner: PredictionRunner
    stream_processor: SameTypeStreamProcessor[Image]

    stream_thread: Thread
    prediction_thread: Thread

    def __init__(self,
                 predictor: ImageClassAndBoundingBoxPredictor,
                 in_stream: IInStream[Image],
                 out_stream: IOutStream[Image],
                 prediction_runner: PredictionRunner=DEFAULT_PREDICTION_RUNNER
        ):
        self.prediction_stream_processor = prediction_stream_processor = ImagePredictionStreamProcessor(predictor)

        prediction_runner.prediction_stream_processor = prediction_stream_processor
        self.prediction_runner = prediction_runner
        
        self.stream_processor = SameTypeStreamProcessor[Image](in_stream, out_stream, prediction_stream_processor.frame_processor)

    def run(self):
        self.start()
        self.wait()
        self.stop()
    
    def start(self):
        self.start_stream_processor()
        self.start_prediction()

    def start_stream_processor(self):
        self.stream_thread = t = Thread(target=self.stream_processor.run)
        t.start()
    
    def start_prediction(self):
        self.prediction_thread = t = Thread(target=self.prediction_runner.run)
        t.start()
    
    def wait(self):
        self.stream_thread.join()
    
    def stop(self):
        self.prediction_runner.stop()