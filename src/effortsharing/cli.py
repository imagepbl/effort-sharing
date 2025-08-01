"""CLI tool for Effort Sharing."""

import argparse
import logging
from pathlib import Path

import numpy as np
from rich.logging import RichHandler

from effortsharing.allocation import allocations_for_region, allocations_for_year, save_allocations
from effortsharing.config import Config
from effortsharing.input.policyscens import policy_scenarios
from effortsharing.pathways.global_pathways import global_pathways


def use_rich_logger(level: str | int = "INFO"):
    """Set up logging with RichHandler.

    Args:
        level: The logging level to set.
    """
    logging.basicConfig(level=level, format="%(message)s", handlers=[RichHandler(show_time=False)])


def add_gas_and_lulucf_args(parser: argparse.ArgumentParser):
    """Add gas and LULUCF arguments to the parser."""
    parser.add_argument("--gas", type=str, default="GHG", choices=["GHG", "CO2"], help="Gas type")
    parser.add_argument(
        "--lulucf",
        type=str,
        default="incl",
        choices=["incl", "excl"],
        help="Land Use, Land-Use Change, and Forestry inclusion/exclusion",
    )


def main():
    """Main entry point for the Effort Sharing CLI tool."""
    parser = argparse.ArgumentParser(
        description="Effort Sharing CLI Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yml"),
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Sub-command to run")
    subparsers.add_parser("global-pathways", help="Generate global pathways data")
    subparsers.add_parser("policy-scenarios", help="Generate policy scenarios data")
    allocate_parser = subparsers.add_parser("allocate", help="Allocate emissions to regions")
    allocate_parser.add_argument("region", type=str, help="Region to allocate emissions for")
    add_gas_and_lulucf_args(allocate_parser)
    aggregate_parser = subparsers.add_parser("aggregate", help="Aggregate emissions data")
    aggregate_parser.add_argument("year", type=int, help="Year to aggregate emissions for")
    add_gas_and_lulucf_args(aggregate_parser)

    args = parser.parse_args()

    use_rich_logger(args.log_level)

    config = Config.from_file(args.config)
    if args.command == "global-pathways":
        global_pathways(config)
    elif args.command == "policy-scenarios":
        policy_scenarios(config)
    elif args.command == "allocate":
        dss = allocations_for_region(config, args.region, args.gas, args.lulucf)
        save_allocations(
            dss=dss, region=args.region, config=config, gas=args.gas, lulucf=args.lulucf
        )
    elif args.command == "aggregate":
        regions_iso = np.load(config.paths.output / "all_regions.npy", allow_pickle=True)
        allocations_for_year(
            year=args.year, config=config, regions=regions_iso, gas=args.gas, lulucf=args.lulucf
        )
