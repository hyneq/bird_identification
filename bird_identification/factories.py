from abc import abstractmethod
from typing import TypeVar, Union, Optional, Protocol, Sequence
from types import ModuleType
from importlib import import_module
from pkgutil import iter_modules

from . import __name__ as _parent_name

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


class LazyImportFactory(IFactory[ProductT]):
    """
    When called, imports and calls a specified factory

    Delaying factory import to usage time saves resources if this factory is not called.
    """

    name: str

    module_name: str
    factory_attr: str

    def __init__(
        self,
        module_name: str,
        factory_attr: str,
        name=None
    ):
        if not name:
            name = module_name

        self.module_name = module_name
        self.factory_attr = factory_attr
        self.name = name

    def __call__(self, *args, **kwargs) -> ProductT:

        module = import_module(self.module_name)
        factory: IFactory[ProductT] = getattr(module, self.factory_attr)

        return factory(*args, **kwargs)

PARENT_MODULE = import_module(__package__)

def search_factories(
    parent: Optional[ModuleType]=PARENT_MODULE,
    path: Optional[Sequence[str]]=None,
    prefix: str='',
    factory_attr='factory'
) -> list[LazyImportFactory]:
    """
    Searches for relevant modules and creates a list of corresponding LazyImportFactories

    Since the goal of LazyImportFactory is not to import before the factory is used, the modules are not 
    inspected for the actual presence of a factory callable when searching.

    If parent module is defined, it overrides path with parent.__path__ and
    uses parent.__name__ + '.' as the prefix of found packages in order
    to treat them as submodules of the parent.
    """

    prepend_prefix = ''

    if parent:
        path = parent.__path__
        prepend_prefix = parent.__name__ + '.'

    prefix = prepend_prefix + prefix

    modules = filter(
        lambda m: m.name.startswith(prefix),
        iter_modules(path=path, prefix=prepend_prefix)
    )

    return [
        LazyImportFactory(module.name, factory_attr, module.name.removeprefix(prefix))
        for module in modules
    ]
