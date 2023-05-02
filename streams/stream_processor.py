from typing import Generic, TypeVar
from abc import ABC, abstractmethod

from . import IInStream, IOutStream

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")

class IFrameProcessor(Generic[InputT, OutputT], ABC):

    @abstractmethod
    def process(self, InputT) -> OutputT:
        pass

class StreamProcessor(Generic[InputT, OutputT]):

    input_stream: IInStream[InputT]
    output_stream: IOutStream[OutputT]

    frame_processor: IFrameProcessor[InputT, OutputT]

    def __init__(self, input_stream: IInStream[InputT], output_stream: IOutStream[OutputT], frame_processor: IFrameProcessor[InputT, OutputT]):
        self.input_stream = input_stream
        self.output_stream = output_stream

        self.frame_processor = frame_processor
    
    def loop(self):
        for input in self.input_stream:
            output = self.process(input)
            self.output_stream.write(output)

    def process(self, input: InputT) -> OutputT:
        return self.frame_processor.process(input)