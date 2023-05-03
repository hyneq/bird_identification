from typing import Optional

from cli import CLIWithParts
from cli.streams import IStreamPairCLIPart, FileStreamPairCLIPart
from cli.prediction import IPredictionCLIPart

from prediction_annotation import ImagePredictionStreamRunner

class StreamPredictionAnnotationCLI(CLIWithParts):

    prediction_cli_part: IPredictionCLIPart
    stream_cli_part: IStreamPairCLIPart

    def __init__(self,
                 prediction_cli_part: IPredictionCLIPart,
                 stream_cli_part: Optional[IStreamPairCLIPart]=None,
        ):

        if not stream_cli_part:
            stream_cli_part = FileStreamPairCLIPart()

        self.stream_cli_part = stream_cli_part
        self.prediction_cli_part = prediction_cli_part
        
        super().__init__(parts=[stream_cli_part, prediction_cli_part])
    
    def get_runner(self):
        (in_stream, out_stream) = self.stream_cli_part.get_stream_pair()

        predictor = self.prediction_cli_part.get_predictor()

        return ImagePredictionStreamRunner(
            predictor=predictor,
            in_stream=in_stream,
            out_stream=out_stream
        )
    
    def run(self):
        super().run()

        runner = self.get_runner()

        runner.run()
