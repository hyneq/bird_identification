from typing import Optional
from abc import ABC, abstractmethod

from argparse import ArgumentParser, Namespace

# This code was inspired by Ansible's CLI (https://github.com/ansible/ansible/tree/ad9867ca5eb8ba27f827d5d5a7999cfb96ae0986/lib/ansible/cli)

class CLI(ABC):
    
    raw_args: list[str]
    parser: ArgumentParser
    args: Namespace

    @abstractmethod
    def init_parser(self):
        self.parser = ArgumentParser()
    
    def parse(self):
        self.init_parser()

        self.args = self.parser.parse_args()
    
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

    parser: ArgumentParser
    args: Optional[Namespace]

    def __init__(self, parser: ArgumentParser):
        self.parser = parser

    @abstractmethod
    def add_opts(self):
        pass

    def add_args(self, args: Namespace):
        self.args = args