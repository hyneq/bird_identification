from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Union, Optional, Protocol, Sequence

ProductT = TypeVar("ProductT", covariant=True)


class IFactory(Protocol[ProductT]):
    name: str

    @abstractmethod
    def __call__(self, *args, **kwargs) -> ProductT:
        pass


FactoryT = TypeVar("FactoryT", bound=IFactory)


class MultiFactory(IFactory[ProductT]):
    name: str

    factories: dict[str, IFactory[ProductT]]
    default_factory: str

    def __init__(
        self,
        factories: Union[Sequence[IFactory[ProductT]], dict[str, IFactory[ProductT]]],
        default_factory: str,
        name="multi",
    ):
        if isinstance(factories, list):
            factories = {f.name: f for f in factories}

        self.name = name
        self.factories = factories
        self.default_factory = default_factory

    def __call__(self, *args, factory: Optional[str] = None, **kwargs) -> ProductT:
        if not factory:
            factory = self.default_factory

        return self.factories[factory](*args, **kwargs)

    @property
    def factory_names(self) -> list[str]:
        return list(self.factories.keys())
