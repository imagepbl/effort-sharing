import logging

import numpy as np
import pandas as pd
import xarray as xr

from effortsharing.allocation.ap import ap
from effortsharing.allocation.utils import LULUCF, Gas, config2base_var, load_dataread, load_future_emissions
from effortsharing.config import Config
from effortsharing.input.emissions import load_emissions, read_modelscenarios
from effortsharing.save import load_rci

logger = logging.getLogger(__name__)


def gdr(
    config: Config,
    region,
    gas: Gas = "GHG",
    lulucf: LULUCF = "incl",
    ap_da: xr.DataArray | None = None,
) -> xr.DataArray:
    """
    Greenhouse Development Rights: Uses the Responsibility-Capability Index
    (RCI) weighed at 50/50 to allocate the global budget
    Calculations from van den Berg et al. (2020)
    """
    logger.info(f"Computing Greenhouse Development Rights allocation for {region}")
    start_year_analysis = config.params.start_year_analysis
    analysis_timeframe = np.arange(start_year_analysis, 2101)
    focus_region = region
    convergence_year_gdr = config.params.convergence_year_gdr
    if ap_da is None:
        ap_da = ap(config, region, gas, lulucf)


    xr_version = load_dataread(config)
    xr_rci = load_rci(config, region_dim=xr_version.Region)
    yearfracs = xr.Dataset(
        data_vars={
            "Value": (
                ["Time"],
                (analysis_timeframe - 2030) / (convergence_year_gdr - 2030),
            )
        },
        coords={"Time": analysis_timeframe},
    )

    # Get the regional RCI values
    # If region is EU, we have to sum over the EU countries
    if focus_region != "EU":
        rci_reg = xr_rci.rci.sel(Region=focus_region)
    else:
        fn = config.paths.input / "UNFCCC_Parties_Groups_noeu.xlsx"
        df = pd.read_excel(fn, sheet_name="Country groups")
        countries_iso = np.array(df["Country ISO Code"])
        group_eu = countries_iso[np.array(df["EU"]) == 1]
        rci_reg = xr_rci.rci.sel(Region=group_eu).sum(dim="Region")

    # Compute GDR until 2030
    emission_data = load_emissions(config)
    emis_base_var = config2base_var(gas, lulucf)
    emis_base = emission_data[emis_base_var]
    emis_fut = load_future_emissions(config, gas, lulucf)
    baseline = emis_base
    global_traject = emis_fut

    gdr = (
        baseline.sel(Region=focus_region)
        - (baseline.sel(Region="EARTH") - global_traject) * rci_reg
    )
    gdr = gdr.rename("Value")

    # GDR Post 2030
    # Calculate the baseline difference
    baseline_earth = baseline.sel(Region="EARTH", Time=analysis_timeframe)
    global_traject_time = global_traject.sel(Time=analysis_timeframe)
    baseline_diff = baseline_earth - global_traject_time

    rci_2030 = baseline_diff * rci_reg.sel(Time=2030)
    part1 = (1 - yearfracs) * (baseline.sel(Region=focus_region) - rci_2030)
    part2 = yearfracs * ap_da.sel(Time=analysis_timeframe)
    gdr_post2030 = (part1 + part2).sel(Time=np.arange(2031, 2101))

    gdr_total = xr.merge([gdr, gdr_post2030])
    gdr_total = gdr_total.rename({"Value": "GDR"})
    logger.info(f"Computed Greenhouse Development Rights allocation for {region}")
    return gdr_total.GDR
