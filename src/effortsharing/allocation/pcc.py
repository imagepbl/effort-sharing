import logging

import numpy as np
import pandas as pd
import xarray as xr

from effortsharing.allocation.gf import gf
from effortsharing.allocation.pc import pc
from effortsharing.allocation.utils import LULUCF, Gas
from effortsharing.config import Config

logger = logging.getLogger(__name__)


def pcc(
    config: Config,
    region,
    gas: Gas = "GHG",
    lulucf: LULUCF = "incl",
    gf_da: xr.DataArray | None = None,
    pc_da: xr.DataArray | None = None,
) -> xr.DataArray:
    """
    Per Capita Convergence: Grandfathering converging into per capita
    """
    logger.info(f"Computing Per Capita Convergence allocation for {region}")
    start_year_analysis = config.params.start_year_analysis
    if gf_da is None:
        gf_da = gf(config, region, gas, lulucf)
    if pc_da is None:
        pc_da = pc(config, region, gas, lulucf)

    def transform_time(time, convyear):
        """
        Function that calculates the convergence based on the convergence time frame
        """
        fractions = pd.DataFrame({"Year": time, "Convergence": 0}, dtype=float)

        before_analysis_year = fractions["Year"] < start_year_analysis + 1
        fractions.loc[before_analysis_year, "Convergence"] = 1.0

        start_conv = start_year_analysis + 1

        during_conv = (fractions["Year"] >= start_conv) & (fractions["Year"] < convyear)
        year_diff = fractions["Year"] - start_year_analysis
        conv_range = convyear - start_year_analysis
        fractions.loc[during_conv, "Convergence"] = 1.0 - (year_diff / conv_range)

        return fractions["Convergence"].tolist()

    gfdeel = []
    pcdeel = []

    dim_convyears = config.dimension_ranges.convergence_years

    times = np.arange(1850, 2101)
    for year in dim_convyears:
        ar = np.array([transform_time(times, year)]).T
        coords = {"Time": times, "Convergence_year": [year]}
        dims = ["Time", "Convergence_year"]
        gfdeel.append(xr.DataArray(data=ar, dims=dims, coords=coords).to_dataset(name="PCC"))
        pcdeel.append(xr.DataArray(data=1 - ar, dims=dims, coords=coords).to_dataset(name="PCC"))

    # Merging the list of DataArays into one Dataset
    gfdeel_single = xr.merge(gfdeel)
    pcdeel_single = xr.merge(pcdeel)

    # Creating new allocation time series by multiplying convergence fractions
    # with existing GF and PC allocations
    xr_new = (gfdeel_single * gf_da + pcdeel_single * pc_da)["PCC"]
    logger.info(f"Computed Per Capita Convergence allocation for {region}")
    return xr_new
