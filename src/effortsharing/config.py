from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class DataPaths:
    input: Path
    intermediate: Path
    output: Path


@dataclass
class Parameters:
    convergence_year_gdr: int
    convergence_year_base: int
    gwp_ch4: float
    gwp_n2o: float
    start_year_analysis: int
    harmonization_year: int
    polynomial_fit_2030relation: int
    version_ndcs: str


@dataclass
class DimensionRanges:
    discount_rates: list[float]
    hist_emissions_startyears: list[int]
    convergence_years: list[int]
    peak_temperature: list[float]
    peak_temperature_saved: list[float]
    negative_emissions: list[float]
    risk_of_exceedance: list[float]
    non_co2_reduction: list[float]
    timing_of_mitigation_action: list[str]


@dataclass
class Config:
    """Configuration of effort-sharing experiments.

    For example:

        config = Config.from_file('config.yml')


    You can get a nice print of the config using rich:

        from rich import print
        print(config)

    """

    load_intermediate_files: bool
    save_intermediate_files: bool
    paths: DataPaths
    params: Parameters
    dimension_ranges: DimensionRanges

    @classmethod
    def from_file(cls, config_file: Path | str) -> "Config":
        # Open file
        config_file = Path(config_file)
        if not config_file.exists():
            raise FileNotFoundError(
                f"Config file {config_file} does not exist. "
                "Please generate with `effortsharing generate-config` command."
            )
        config = yaml.load(config_file.read_text(), Loader=yaml.FullLoader)

        # Convert strings to Path
        paths = {k: Path(v) for k, v in config.get("paths").items()}

        # Resolve relative paths (assume relative to config file)
        origin = Path(config_file).absolute().parent
        for k, v in paths.items():
            if not v.is_absolute():
                paths[k] = (origin / v).resolve()

        # Forge settings into a new Config object
        return Config(
            load_intermediate_files=config.get("load_intermediate_files", True),
            save_intermediate_files=config.get("save_intermediate_files", True),
            paths=DataPaths(**paths),
            params=Parameters(**config.get("params")),
            dimension_ranges=DimensionRanges(**config.get("dimension_ranges")),
        )


if __name__ == "__main__":
    import argparse

    from rich import print

    # Get the config file from command line arguments
    parser = argparse.ArgumentParser(description="Process all input data")
    parser.add_argument("config", help="Path to config file")
    args = parser.parse_args()

    config = Config.from_file(args.config)

    print(config)
