import logging

import numpy as np
import xarray as xr

from effortsharing.allocation.utils import LULUCF, Gas, config2hist_var, load_future_emissions
from effortsharing.config import Config
from effortsharing.input.emissions import load_emissions, read_modelscenarios

logger = logging.getLogger(__name__)


def gf(config: Config, region, gas: Gas = "GHG", lulucf: LULUCF = "incl") -> xr.DataArray:
    """
    Grandfathering: Divide the global budget over the regions based on
    their historical CO2 emissions
    """
    logger.info(f"Computing Grandfathering allocation for {region}")
    start_year_analysis = config.params.start_year_analysis
    analysis_timeframe = np.arange(start_year_analysis, 2101)

    emission_data = load_emissions(config)
    hist_var = config2hist_var(gas, lulucf)
    emis_fut = load_future_emissions(config, gas, lulucf)

    # Calculating the current CO2 fraction for region and world based on start_year_analysis
    current_co2_region = emission_data[hist_var].sel(Region=region, Time=start_year_analysis)

    current_co2_earth = 1e-9 + emission_data[hist_var].sel(Region="EARTH", Time=start_year_analysis)

    co2_fraction = current_co2_region / current_co2_earth

    # New CO2 time series from the start_year to 2101 by multiplying global budget with fraction
    xr_new_co2 = co2_fraction * emis_fut.sel(Time=analysis_timeframe)
    logger.info(f"Computed Grandfathering allocation for {region}")
    return xr_new_co2.rename("GF")
