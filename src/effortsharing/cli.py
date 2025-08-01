"""CLI tool for Effort Sharing."""

import logging
import sys
import urllib.request
from pathlib import Path
from typing import Literal

import cyclopts
import numpy as np
from rich.logging import RichHandler

from effortsharing.allocation import allocations_for_region, allocations_for_year, save_allocations
from effortsharing.allocation.utils import LULUCF, Gas
from effortsharing.config import Config
from effortsharing.input.policyscens import policy_scenarios as run_policy_scenarios
from effortsharing.pathways.global_pathways import global_pathways as run_global_pathways

logger = logging.getLogger(__name__)


def get_version() -> str:
    """Get the package version at runtime."""
    from importlib.metadata import version

    return version("effort-sharing")


app = cyclopts.App(name="effortsharing", help="Effort Sharing CLI Tool", version=get_version)

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] | int


def use_rich_logger(level: LogLevel = "INFO"):
    """Set up logging with RichHandler.

    Args:
        level: The logging level to set.
    """
    logging.basicConfig(level=level, format="%(message)s", handlers=[RichHandler(show_time=False)])


@app.command
def generate_config(
    config: Path = Path("config.yml"),
    log_level: LogLevel = "INFO",
):
    """Generate a configuration file by downloading the default one from GitHub.

    Args:
        config: Path to configuration YAML file to write.
        log_level: Set the logging level.
    """
    use_rich_logger(log_level)

    if config.exists():
        logger.error(f"Config file {config} already exists. Please remove first to re-generate.")
        sys.exit(1)

    branch = "main"
    url = f"https://github.com/imagepbl/effort-sharing/raw/refs/heads/{branch}/config.default.yml"
    try:
        urllib.request.urlretrieve(url, config)
        logging.info(f"Downloaded config from {url} to {config}")
    except Exception as e:
        logging.error(f"Failed to download config from {url}: {e}")
        sys.exit(1)


@app.command
def get_input_data(
    config: Path = Path("config.yml"),
    log_level: LogLevel = "INFO",
):
    """Download input data files.

    Args:
        config: Path to configuration YAML file.
        log_level: Set the logging level.
    """
    use_rich_logger(log_level)
    # TODO implement by fetching from Zenodo with pooch
    logger.error(
        "The 'get-input-data' command is not implemented yet. "
        "Please contact Mark (mark.dekker@pbl.nl) for download instructions."
    )
    sys.exit(1)


@app.command
def global_pathways(
    config: Path = Path("config.yml"),
    log_level: LogLevel = "INFO",
):
    """Generate global pathways data.

    Args:
        config: Path to configuration YAML file.
        log_level: Set the logging level.
    """
    use_rich_logger(log_level)
    config_obj = Config.from_file(config)
    run_global_pathways(config_obj)


@app.command
def policy_scenarios(
    config: Path = Path("config.yml"),
    log_level: LogLevel = "INFO",
):
    """Generate policy scenarios data.

    Args:
        config: Path to configuration YAML file.
        log_level: Set the logging level.
    """
    use_rich_logger(log_level)
    config_obj = Config.from_file(config)
    run_policy_scenarios(config_obj)


@app.command
def allocate(
    region: str,
    gas: Gas = "GHG",
    lulucf: LULUCF = "incl",
    config: Path = Path("config.yml"),
    log_level: LogLevel = "INFO",
):
    """Allocate emissions for a region.

    Args:
        region: Region to allocate emissions for.
        gas: Gas type.
        lulucf: Land Use, Land-Use Change, and Forestry inclusion/exclusion.
        config: Path to configuration YAML file.
        log_level: Set the logging level.
    """
    use_rich_logger(log_level)

    config_obj = Config.from_file(config)
    dss = allocations_for_region(config_obj, region, gas, lulucf)
    save_allocations(dss=dss, region=region, config=config_obj, gas=gas, lulucf=lulucf)


@app.command
def aggregate(
    year: int,
    gas: Gas = "GHG",
    lulucf: LULUCF = "incl",
    config: Path = Path("config.yml"),
    log_level: LogLevel = "INFO",
):
    """Aggregate emissions data.

    Expects that allocation has been generated for each region and the given gas and lulucf.

    Args:
        year: Year to aggregate emissions for.
        gas: Gas type.
        lulucf: Land Use, Land-Use Change, and Forestry inclusion/exclusion.
        config: Path to configuration YAML file.
        log_level: Set the logging level.
    """
    use_rich_logger(log_level)

    config_obj = Config.from_file(config)
    regions_iso = np.load(config_obj.paths.output / "all_regions.npy", allow_pickle=True)
    allocations_for_year(year=year, config=config_obj, regions=regions_iso, gas=gas, lulucf=lulucf)
