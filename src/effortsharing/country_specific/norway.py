import logging

import numpy as np
import xarray as xr

from effortsharing.config import Config
from effortsharing.save import ENCODING

logger = logging.getLogger(__name__)


def datareader_norway(config: Config, xr_total, xr_primap):
    # Norwegian emissions - harmonized with EDGAR
    logger.info("Processing custom emission data for Norway")

    savepath = config.paths.output / f"startyear_{config.params.start_year_analysis}"
    xr_dataread_nor = xr.open_dataset(savepath / "xr_dataread.nc").load().copy()

    time_future = np.arange(config.params.start_year_analysis, 2101)
    time_past = np.arange(1850, config.params.start_year_analysis + 1)

    # Get data and interpolate
    time_axis = np.arange(1990, config.params.start_year_analysis + 1)
    ghg_axis = np.array(
        xr_primap.sel(Scenario="HISTCR", Region="NOR", time=time_axis, Category="M.0.EL")[
            "KYOTOGHG (AR6GWP100)"
        ]
    )
    time_interp = np.arange(np.min(time_axis), np.max(time_axis) + 1)
    ghg_interp = np.interp(time_interp, time_axis, ghg_axis)

    # Get older data by linking to Jones
    fraction_minyear = float(
        ghg_axis[0] / xr_total.GHG_hist_excl.sel(Region="NOR", Time=np.min(time_axis))
    )
    pre_minyear_raw = (
        np.array(xr_total.GHG_hist_excl.sel(Region="NOR", Time=np.arange(1850, np.min(time_axis))))
        * fraction_minyear
    )
    total_ghg_nor = np.array(list(pre_minyear_raw) + list(ghg_interp)) / 1e3
    fractions = np.array(
        xr_dataread_nor.GHG_hist_excl.sel(Region="NOR", Time=time_past) / total_ghg_nor
    )
    for t_i, t in enumerate(time_past):
        xr_dataread_nor.GHG_hist_excl.loc[dict(Time=t, Region="NOR")] = total_ghg_nor[t_i]

    xr_dataread_nor.CO2_base_incl.loc[dict(Region="NOR", Time=time_future)] = (
        xr_dataread_nor.CO2_base_incl.sel(Region="NOR", Time=time_future) / fractions[-1]
    )
    xr_dataread_nor.CO2_base_excl.loc[dict(Region="NOR", Time=time_future)] = (
        xr_dataread_nor.CO2_base_excl.sel(Region="NOR", Time=time_future) / fractions[-1]
    )
    xr_dataread_nor.GHG_base_incl.loc[dict(Region="NOR", Time=time_future)] = (
        xr_dataread_nor.GHG_base_incl.sel(Region="NOR", Time=time_future) / fractions[-1]
    )
    xr_dataread_nor.GHG_base_excl.loc[dict(Region="NOR", Time=time_future)] = (
        xr_dataread_nor.GHG_base_excl.sel(Region="NOR", Time=time_future) / fractions[-1]
    )

    xr_dataread_nor.CO2_hist.loc[dict(Region="NOR", Time=time_past)] = (
        xr_dataread_nor.CO2_hist.sel(Region="NOR", Time=time_past) / fractions
    )
    xr_dataread_nor.CO2_hist_excl.loc[dict(Region="NOR", Time=time_past)] = (
        xr_dataread_nor.CO2_hist_excl.sel(Region="NOR", Time=time_past) / fractions
    )
    xr_dataread_nor.GHG_hist.loc[dict(Region="NOR", Time=time_past)] = (
        xr_dataread_nor.GHG_hist.sel(Region="NOR", Time=time_past) / fractions
    )

    # Save the data
    logger.info(f"Saving Norway data to {savepath / 'xr_dataread_NOR.nc'}")
    xr_dataread_nor.sel(
        Temperature=np.array(config.dimension_ranges.peak_temperature_saved).astype(float).round(2)
    ).to_netcdf(
        savepath / "xr_dataread_NOR.nc",
        encoding=ENCODING,
        format="NETCDF4",
        engine="netcdf4",
    )
