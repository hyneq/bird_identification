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


class CLIPart(ABC):

    parser: ArgumentParser
    args: Namespace

    @abstractmethod
    def add_opts(self):
        pass

    def add_args(self, args: Namespace):
        self.args = args