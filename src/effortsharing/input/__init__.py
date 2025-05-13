"""
Input module for processing various data sources.

Functions:
- process_all: Process all available data sources and save to disk

Submodules (for access to individual data reading functions):
- emissions
- socioeconomics
- ndcs
- all (source file for process_all)

"""

# Expose complete submodules for advanced usage
# These allow imports like: from effortsharing.input import emissions
# or: from effortsharing.input import process_all
from . import emissions, ndcs, socioeconomics
from .all import process_all
