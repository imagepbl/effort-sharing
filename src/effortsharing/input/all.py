import logging

from effortsharing.config import Config
from effortsharing.input import emissions, ndcs, socioeconomics

# Set up logging
logger = logging.getLogger(__name__)


def load_all(config: Config, from_intermediate=True, save=True):
    """Load all input data.

    Args:
        config: effortsharing.config.Config object
        from_intermediate: Whether to read from intermediate files if available (default: True)
        save: Whether to save intermediate data to disk (default: True)
    """
    logger.info("Loading input data")

    socioeconomic_data = socioeconomics.load_socioeconomics(config, from_intermediate, save)
    emission_data, scenarios = emissions.load_emissions(config, from_intermediate, save)
    ndc_data = ndcs.load_ndcs(config, emission_data, from_intermediate, save)

    return emission_data, socioeconomic_data, scenarios, ndc_data


if __name__ == "__main__":
    import argparse

    from rich.logging import RichHandler

    # Set up logging
    logging.basicConfig(level="INFO", format="%(message)s", handlers=[RichHandler(show_time=False)])

    # Get the config file from command line arguments
    parser = argparse.ArgumentParser(description="Process all input data")
    parser.add_argument("config", help="Path to config file")
    args = parser.parse_args()

    config = Config.from_file(args.config)
    load_all(config, from_intermediate=False)
