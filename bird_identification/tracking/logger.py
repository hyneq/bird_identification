from typing import TypeVar, Generic
from abc import ABC, abstractmethod

from ..factories import IFactory

LoggedObjectT = TypeVar("LoggedObjectT")

class IObjectLogger(ABC, Generic[LoggedObjectT]):
    __slots__: tuple

    @abstractmethod
    def log(self, objects: list[LoggedObjectT]):
        pass


class IObjectLoggerFactory(IFactory[IObjectLogger[LoggedObjectT]]):
    pass
