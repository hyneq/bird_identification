from typing import TextIO, Any, Optional
from dataclasses import dataclass
import sys
import pprint

from .logger import ListObjectLogger, IObjectLoggerFactory

DEFAULT_PATH: str = "-"
DEFAULT_FILE: TextIO = sys.stdout

class FileLogger(ListObjectLogger[Any]):
    __slots__: tuple

    printer: pprint.PrettyPrinter
    needs_close: bool

    def __init__(self, printer: pprint.PrettyPrinter, needs_close: bool = False):
        super().__init__()

        self.printer = printer
        self.needs_close = needs_close


    def _log(self):
        self.printer.pprint(self.objects)


    def close(self):
        self.printer._stream.close()


@dataclass(frozen=True)
class FileLoggerFactory(IObjectLoggerFactory[Any]):

    name = "file"
    default_path: str = DEFAULT_PATH
    default_file: TextIO = DEFAULT_FILE

    def __call__(self,
        path: Optional[str] = None,
        append: bool = False,
        file: Optional[TextIO] = None,
        printer: Optional[pprint.PrettyPrinter] = None,
        *_, **__
    ):
        needs_close = False
        if not printer:
            if not file:
                if path:
                    if append:
                        mode = 'w+'
                    else:
                        mode = 'w'

                    file = open(path, mode, encoding="UTF-8")
                    needs_close = True

            printer = pprint.PrettyPrinter(stream=file)
        
        return FileLogger(printer, needs_close=needs_close)


factory = FileLoggerFactory()

get_file_logger = factory
