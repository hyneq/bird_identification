from abc import ABC, abstractmethod

from cli import ArgumentParser, CLIPart, Namespace, args_required

from streams import IInStream, IOutStream, IFileStreamPairFactory, get_file_video_stream_pair

class IStreamPairCLIPart(CLIPart, ABC):
    @abstractmethod
    @args_required
    def get_stream_pair(self) -> tuple[IInStream, IOutStream]:
        pass

class FileStreamPairCLIPart(IStreamPairCLIPart):

    stream_pair_factory: IFileStreamPairFactory

    def __init__(self, *args, stream_pair_factory: IFileStreamPairFactory=get_file_video_stream_pair, **kwargs):
        super().__init__(*args, **kwargs)
        self.stream_pair_factory = stream_pair_factory

    def add_opts(self, parser: ArgumentParser):
        parser.add_argument("-i", "--in", "--in-path", dest="in_path", required=True)
        parser.add_argument("-o", "--out", "--out-path", dest="out_path", required=True)
    
    @args_required
    def get_stream_pair(self) -> tuple[IInStream, IOutStream]:
        return self.stream_pair_factory(
            in_path=self.args.in_path,
            out_path=self.args.out_path
        )
