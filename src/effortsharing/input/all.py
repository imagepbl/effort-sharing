import logging

from effortsharing.config import Config

from . import emissions, ndcs, socioeconomics

# Set up logging
logger = logging.getLogger(__name__)


def process_all(config: Config, save=True):
    """Process all input data.

    Args:
        config: effortsharing.config.Config object
        save: Whether to save intermediate data to disk (default: True)

    Returns:
        Dict containing all loaded data
    """
    logger.info("Processing input data")

    emission_data = emissions.process_emissions(config, save=save)
    socioeconomic_data = socioeconomics.process_socioeconomics(config, save=save)
    ndc_data = ndcs.process_ndcs(config, save=save)

    logger.info("Completed processing input data")

    return emission_data, socioeconomic_data, ndc_data


if __name__ == "__main__":
    import argparse

    # Get the config file from command line arguments
    parser = argparse.ArgumentParser(description="Process all input data")
    parser.add_argument("config", help="Path to config file")
    args = parser.parse_args()

    config = Config.from_file(args.config_file)
    process_all(config)
