from typing import TypeVar, Generic
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..factories import IFactory

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

    def add(self, obj: LoggedObjectT):
        self.objects.append(obj)


@dataclass(frozen=True)
class ClassLoggedObject:
    class_name: Optional[str]
    start_time: float
    end_time: float