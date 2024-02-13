#!/bin/env python3
"""
A generic entry point to the bird_identification package's command-line interfaces.

When linked to with the name of a module from bird_identification.cli package,
it imports the module and tries to execute its cli_main() function, which
is the entry point for the particular CLI.

This allows for calling the package's CLI from command-line as ordinary commands.
"""

import os
import sys
import importlib

def add_to_path():
    """
    Adds the parent package's path to sys.path, if not already present
    """

    package_path = os.path.realpath(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.pardir, os.path.pardir)
    )

    if package_path not in sys.path:
        sys.path.append(package_path)

def dispatch():
    """
    Tries to import a module from bird_identification.cli with the name
    taken from the filename of the symlink used to run this module
    """

    name = os.path.splitext(os.path.basename(__file__))[0]

    cli = importlib.import_module("bird_identification.cli." + name)

    cli.cli_main()

if __name__ == "__main__":
    add_to_path()
    dispatch()
