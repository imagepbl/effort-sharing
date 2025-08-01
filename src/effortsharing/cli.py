"""CLI tool for Effort Sharing."""

import argparse
import logging
import sys
import urllib.request
from pathlib import Path

import numpy as np
from rich.logging import RichHandler

from effortsharing.allocation import allocations_for_region, allocations_for_year, save_allocations
from effortsharing.config import Config
from effortsharing.input.policyscens import policy_scenarios
from effortsharing.pathways.global_pathways import global_pathways

logger = logging.getLogger(__name__)


def use_rich_logger(level: str | int = "INFO"):
    """Set up logging with RichHandler.

    Args:
        level: The logging level to set.
    """
    logging.basicConfig(level=level, format="%(message)s", handlers=[RichHandler(show_time=False)])


def generate_config(dest: Path = Path("config.yml")):
    """Generate a configuration file by downloading the default one from GitHub."""
    branch = "main"
    url = f"https://github.com/imagepbl/effort-sharing/raw/refs/heads/{branch}/config.default.yml"
    try:
        urllib.request.urlretrieve(url, dest)
        logging.info(f"Downloaded config from {url} to {dest}")
    except Exception as e:
        logging.error(f"Failed to download config from {url}: {e}")


def get_input_data():
    """Placeholder for the 'get-input-data' command."""
    # TODO implement by fetching from Zenodo with pooch
    logger.error(
        "The 'get-input-data' command is not implemented yet. "
        "Please contact Mark (mark.dekker@pbl.nl) for download instructions."
    )
    sys.exit(1)


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
    subparsers.add_parser(
        "generate-config",
        help="Generate a configuration file by downloading the default one from GitHub.",
    )
    subparsers.add_parser("get-input-data", help="Download input data files")
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

    if args.command == "generate-config":
        generate_config()
        return

    config = Config.from_file(args.config)
    if args.command == "get-input-data":
        get_input_data()
    elif args.command == "global-pathways":
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
