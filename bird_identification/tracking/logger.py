from typing import TypeVar, Generic, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..factories import IFactory, MultiFactory

LoggedObjectT = TypeVar("LoggedObjectT")

class IObjectLogger(ABC, Generic[LoggedObjectT]):
    __slots__: tuple

    @abstractmethod
    def add(self, obj: LoggedObjectT):
        pass


    @abstractmethod
    def log(self):
        pass


    def close(self):
        pass


class IObjectLoggerFactory(IFactory[IObjectLogger[LoggedObjectT]]):
    __slots__: tuple


class ListObjectLogger(IObjectLogger[LoggedObjectT]):
    __slots__: tuple

    objects: list[LoggedObjectT]

    def __init__(self):
        self.objects = []


    def add(self, obj: LoggedObjectT):
        self.objects.append(obj)


    def log(self):
        if self.objects:
            self._log()
            self.objects.clear()


    @abstractmethod
    def _log(self):
        pass


@dataclass(frozen=True)
class ClassLoggedObject:
    class_name: Optional[str]
    start_time: float
    end_time: float


class MultiLoggerFactory(
    MultiFactory[
        IObjectLogger[LoggedObjectT]
    ],
    IObjectLoggerFactory[LoggedObjectT]
):
    pass


from ..factories import search_factories

from ..defaults.tracking import DEFAULT_LOGGER

object_logger_factory = MultiLoggerFactory[
    ClassLoggedObject
](factories=search_factories(prefix='logger_'), default_factory=DEFAULT_LOGGER)
