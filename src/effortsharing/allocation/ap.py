import logging

import numpy as np
import xarray as xr

from effortsharing.allocation.utils import (
    LULUCF,
    Gas,
    config2base_var,
    load_dataread,
    load_future_emissions,
)
from effortsharing.config import Config
from effortsharing.input.emissions import load_emissions, read_modelscenarios

logger = logging.getLogger(__name__)


def ap(config: Config, region, gas: Gas = "GHG", lulucf: LULUCF = "incl") -> xr.DataArray:
    """
    Ability to Pay: Uses GDP per capita to allocate the global budget
    Equation from van den Berg et al. (2020)
    """
    logger.info(f"Computing Ability to Pay allocation for {region}")
    start_year_analysis = config.params.start_year_analysis
    analysis_timeframe = np.arange(start_year_analysis, 2101)
    focus_region = region

    # Step 1: Reductions before correction factor
    # xr_social = load_socioeconomics(config).sel(Time=analysis_timeframe, Region="EARTH")
    # GDP_sum_w = xr_social.GDP
    # pop_sum_w = xr_social.Population
    # TODO replace with load_socioeconomics() function, see #145
    # need to check if commented code above is equivalent to below
    # aka is xrt.GDP same as xr_social.GDP?
    xrt = load_dataread(config)
    GDP_sum_w = xrt.GDP.sel(Region="EARTH")
    pop_sum_w = xrt.Population.sel(Region="EARTH")
    # Global average GDP per capita
    r1_nom = GDP_sum_w / pop_sum_w

    emission_data = load_emissions(config)
    scenarios = read_modelscenarios(config)
    emis_base_var = config2base_var(gas, lulucf)
    emis_base = emission_data[emis_base_var]
    emis_fut = load_future_emissions(config, emission_data, scenarios, gas, lulucf)

    base_worldsum = emis_base.sel(Time=analysis_timeframe).sel(Region="EARTH")
    rb_part1 = (
        xrt.GDP.sel(Region=focus_region) / xrt.Population.sel(Region=focus_region) / r1_nom
    ) ** (1 / 3.0)
    rb_part2 = (
        emis_base.sel(Time=analysis_timeframe).sel(Region=focus_region)
        * (base_worldsum - emis_fut.sel(Time=analysis_timeframe))
        / base_worldsum
    )
    rb = rb_part1 * rb_part2

    # Step 2: Correction factor
    # TODO replace open with load_rbw() function, will need to find where files are written
    rbw_path = (
        config.paths.output / f"startyear_{start_year_analysis}" / f"xr_rbw_{gas}_{lulucf}.nc"
    )
    rbw = xr.open_dataset(rbw_path).load()
    corr_factor = (1e-9 + rbw.__xarray_dataarray_variable__) / (
        base_worldsum - emis_fut.sel(Time=analysis_timeframe)
    )

    # Step 3: Budget after correction factor
    ap = emis_base.sel(Region=focus_region) - rb / corr_factor

    ap = ap.sel(Time=analysis_timeframe)
    logger.info(f"Computed Ability to Pay allocation for {region}")
    return ap.rename("AP")
