import logging

import numpy as np
import xarray as xr

from effortsharing.config import Config
from effortsharing.save import ENCODING

logger = logging.getLogger(__name__)


def datareader_netherlands(config: Config, xr_total):
    logger.info("Processing custom emission data for Norway")

    savepath = config.paths.output / f"startyear_{config.params.start_year_analysis}"
    time_future = np.arange(config.params.start_year_analysis, 2101)
    time_past = np.arange(1850, config.params.start_year_analysis + 1)

    # Dutch emissions - harmonized with the KEV # TODO harmonize global emissions with this, as well.
    xr_dataread_nld = xr.open_dataset(savepath / "xr_dataread.nc").load().copy()
    dutch_time = np.array(
        [
            1990,
            1995,
            2000,
            2005,
            2010,
            2011,
            2012,
            2013,
            2014,
            2015,
            2016,
            2017,
            2018,
            2019,
            2020,
            2021,
        ]
    )
    dutch_ghg = np.array(
        [
            228.9,
            238.0,
            225.7,
            220.9,
            219.8,
            206,
            202,
            201.2,
            192.9,
            199.8,
            200.2,
            196.5,
            191.4,
            185.6,
            168.9,
            172.0,
        ]
    )
    dutch_time_interp = np.arange(1990, config.params.start_year_analysis + 1)
    dutch_ghg_interp = np.interp(dutch_time_interp, dutch_time, dutch_ghg)
    fraction_1990 = float(dutch_ghg[0] / xr_total.GHG_hist.sel(Region="NLD", Time=1990))
    pre_1990_raw = (
        np.array(xr_total.GHG_hist.sel(Region="NLD", Time=np.arange(1850, 1990))) * fraction_1990
    )
    total_ghg_nld = np.array(list(pre_1990_raw) + list(dutch_ghg_interp))
    fractions = np.array(
        xr_dataread_nld.GHG_hist.sel(
            Region="NLD",
            Time=np.arange(1850, config.params.start_year_analysis + 1),
        )
        / total_ghg_nld
    )
    for t_i, t in enumerate(time_past):
        xr_dataread_nld.GHG_hist.loc[dict(Time=t, Region="NLD")] = total_ghg_nld[t_i]

    xr_dataread_nld.CO2_base_incl.loc[dict(Region="NLD", Time=time_future)] = (
        xr_dataread_nld.CO2_base_incl.sel(Region="NLD", Time=time_future) / fractions[-1]
    )
    xr_dataread_nld.CO2_base_excl.loc[dict(Region="NLD", Time=time_future)] = (
        xr_dataread_nld.CO2_base_excl.sel(Region="NLD", Time=time_future) / fractions[-1]
    )
    xr_dataread_nld.GHG_base_incl.loc[dict(Region="NLD", Time=time_future)] = (
        xr_dataread_nld.GHG_base_incl.sel(Region="NLD", Time=time_future) / fractions[-1]
    )
    xr_dataread_nld.GHG_base_excl.loc[dict(Region="NLD", Time=time_future)] = (
        xr_dataread_nld.GHG_base_excl.sel(Region="NLD", Time=time_future) / fractions[-1]
    )

    xr_dataread_nld.CO2_hist.loc[dict(Region="NLD", Time=time_past)] = (
        xr_dataread_nld.CO2_hist.sel(Region="NLD", Time=time_past) / fractions
    )
    xr_dataread_nld.CO2_hist_excl.loc[dict(Region="NLD", Time=time_past)] = (
        xr_dataread_nld.CO2_hist_excl.sel(Region="NLD", Time=time_past) / fractions
    )
    xr_dataread_nld.GHG_hist_excl.loc[dict(Region="NLD", Time=time_past)] = (
        xr_dataread_nld.GHG_hist_excl.sel(Region="NLD", Time=time_past) / fractions
    )

    # Save the data
    logger.info(f"Saving Netherlands data to {savepath / 'xr_dataread_NLD.nc'}")
    xr_dataread_nld.sel(
        Temperature=np.array(config.dimension_ranges.peak_temperature_saved).astype(float).round(2)
    ).to_netcdf(
        savepath / "xr_dataread_NLD.nc",
        encoding=ENCODING,
        format="NETCDF4",
        engine="netcdf4",
    )
