import logging

import numpy as np
import xarray as xr

from effortsharing.allocation.utils import LULUCF, Gas, load_future_emissions, load_population
from effortsharing.config import Config
from effortsharing.input.emissions import load_emissions, read_modelscenarios

logger = logging.getLogger(__name__)


def pc(config: Config, region, gas: Gas = "GHG", lulucf: LULUCF = "incl") -> xr.DataArray:
    """
    Per Capita: Divide the global budget equally per capita
    """
    logger.info(f"Computing Per Capita allocation for {region}")
    start_year_analysis = config.params.start_year_analysis
    analysis_timeframe = np.arange(start_year_analysis, 2101)

    population = load_population(config)
    # TODO use function compute countries or read from file
    countries_iso_path = config.paths.output / "all_countries.npy"
    countries_iso = np.load(countries_iso_path, allow_pickle=True)
    pop_region = population.sel(Region=region, Time=start_year_analysis)
    pop_earth = population.sel(Region=countries_iso, Time=start_year_analysis).sum(dim=["Region"])
    pop_fraction = pop_region / pop_earth

    # Multiplying the global budget with the population fraction to create
    # new allocation time series from start_year to 2101
    emission_data = load_emissions(config)
    scenarios = read_modelscenarios(config)
    emis_fut = load_future_emissions(config, emission_data, scenarios, gas, lulucf)

    xr_new = (pop_fraction * emis_fut).sel(Time=analysis_timeframe)
    logger.info(f"Computed Per Capita allocation for {region}")
    return xr_new.rename("PC")
