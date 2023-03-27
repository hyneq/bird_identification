from typing import Optional
from abc import ABC, abstractmethod

from argparse import ArgumentParser, Namespace

# This code was inspired by Ansible's CLI (https://github.com/ansible/ansible/tree/ad9867ca5eb8ba27f827d5d5a7999cfb96ae0986/lib/ansible/cli)

class CLI(ABC):
    
    raw_args: Optional[list[str]]
    parser: ArgumentParser
    args: Namespace

    def __init__(self, raw_args: Optional[list[str]] = None):
        self.raw_args = raw_args

    @abstractmethod
    def init_parser(self):
        self.parser = ArgumentParser()
    
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

class CLIPart(ABC):

    args: Optional[Namespace]

    @abstractmethod
    def add_opts(self, parser: ArgumentParser):
        pass

    def add_args(self, args: Namespace):
        self.args = args

class CLIWithParts(CLI):

    parts: list[CLIPart]

    def __init__(self, *args, parts: Optional[list[CLIPart]]=None, **kwargs):
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
