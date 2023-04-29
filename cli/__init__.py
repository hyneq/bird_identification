from typing import Union, Optional, Generic, TypeVar
from typing_extensions import Self
from abc import ABC, abstractmethod
from dataclasses import dataclass

import re
import argparse

# This code was inspired by Ansible's CLI (https://github.com/ansible/ansible/tree/ad9867ca5eb8ba27f827d5d5a7999cfb96ae0986/lib/ansible/cli)

class ArgumentParserModifier(ABC):

    parser: argparse.ArgumentParser

    def __init__(self, parser: argparse.ArgumentParser):
        self.parser = parser

    @abstractmethod
    def add_argument(self, *args, **kwargs):
        pass

    def __getattr__(self, name: str):
        return getattr(self.parser, name)

ArgumentParser = Union[argparse.ArgumentParser, ArgumentParserModifier]

class NamespaceModifier:

    @abstractmethod
    def __getattr__(self, name):
        pass

Namespace = Union[argparse.Namespace, NamespaceModifier]

def get_argument_name(prefix: str, name: str) -> str:
    return prefix + "-" + name


class OptionPrefixerNamespace(NamespaceModifier):
    
    ns: Namespace
    prefix: str

    def __init__(self, ns: Namespace, prefix: str):
        self.ns = ns
        self.prefix = prefix

    def __getattr__(self, name):
        return getattr(self.ns, get_argument_name(self.prefix, name))

class OptionPrefixerParser(ArgumentParserModifier):

    parser: Union[ArgumentParser, argparse._ArgumentGroup]
    prefix: str

    def __init__(self, parser:  ArgumentParser, prefix: str, group: bool=False, descr: Optional[str]=None):
        self.prefix = prefix

        if isinstance(parser, argparse.ArgumentParser) and group:
            self.parser = parser.add_argument_group(prefix, descr)
        else:
            self.parser = parser
    
    def add_argument(self, *name_or_flags: str, **kwargs):

        flags = self.modify_flags(name_or_flags)

        if name := kwargs.get("dest", None):
            kwargs["dest"] = get_argument_name(self.prefix, name)

        return self.parser.add_argument(*flags, **kwargs)
    
    def modify_flags(self, name_or_flags: tuple[str]) -> tuple[str, ...]:
        flags: list[str] = []
        for flag in name_or_flags:
            if re.match(r"^[" + self.parser.prefix_chars + r"]{2}.+", flag):
                flag = flag[0:2] + self.prefix + "-" + flag[3:]

                flags.append(flag)
        
        return tuple(flags)

class CLI(ABC):
    
    raw_args: Optional[list[str]]
    parser: argparse.ArgumentParser
    args: argparse.Namespace

    def __init__(self, raw_args: Optional[list[str]] = None):
        self.raw_args = raw_args

    @abstractmethod
    def init_parser(self):
        self.parser = argparse.ArgumentParser()
    
    def parse(self):
        self.init_parser()

        self.args = self.parser.parse_args(args = self.raw_args)
    
    @abstractmethod
    def run(self):
        self.parse()


def args_required(method):

    def inner(self, *args, **kwargs):
        if not self.args:
            raise RuntimeError("Method {} requires parsed args to be already available".format(method.__name__))
        
        return method(self, *args, **kwargs)
    
    return inner

class ICLIPart(ABC):

    @abstractmethod
    def add_opts(self, parser: ArgumentParser):
        pass

    @abstractmethod
    def add_args(self, args: Namespace):
        pass

CLIPartT = TypeVar("CLIPartT", bound=ICLIPart)

class CLIPart(ICLIPart):

    args: Optional[Namespace]

    def __init__(self):
        self.args = None

    @abstractmethod
    def add_opts(self, parser: ArgumentParser):
        pass

    def add_args(self, args: Namespace):
        self.args = args

class CLIPartPrefixWrapper(ICLIPart, Generic[CLIPartT]):

    cli_part: CLIPartT

    prefix: str
    group: bool
    descr: Optional[str]

    def __init__(self, cli_part: CLIPartT, prefix: str, group: bool=False, descr: Optional[str]=None):
        super().__init__()

        self.cli_part = cli_part

        self.prefix = prefix
        self.group = group
        self.descr = descr
    
    def add_opts(self, parser: ArgumentParser):
        return self.cli_part.add_opts(OptionPrefixerParser(parser, self.prefix, self.group, self.descr))
    
    def add_args(self, args: Namespace):
        return self.cli_part.add_args(OptionPrefixerNamespace(args, self.prefix))
    
    def __getattr__(self, name: str):
        return getattr(self.cli_part, name)


class CLIWithParts(CLI):

    parts: list[ICLIPart]

    def __init__(self, *args, parts: Optional[list[ICLIPart]]=None, **kwargs):
        if not parts:
            parts = []
        
        self.parts = parts
        
        super().__init__(*args, **kwargs)
    
    def init_parser(self):
        super().init_parser()

        for part in self.parts:
            part.add_opts(self.parser)
    
    def parse(self):
        super().parse()

        for part in self.parts:
            part.add_args(self.args)
