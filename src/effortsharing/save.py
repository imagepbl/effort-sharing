import logging

import numpy as np
import pandas as pd
import xarray as xr

from effortsharing.cache import intermediate_file
from effortsharing.config import Config

logger = logging.getLogger(__name__)


# TODO: Is this really necessary? Or can we remove it?
def save_regions(config, countries, regions):
    logger.info(f"Saving regions and countries to {config.paths.output}")

    # Save regions and countries
    regions_name = np.array(list(regions.keys()))
    regions_iso = np.array(list(regions.values()))
    countries_name = np.array(list(countries.keys()))
    countries_iso = np.array(list(countries.values()))

    np.save(config.paths.output / "all_regions.npy", regions_iso)
    np.save(config.paths.output / "all_regions_names.npy", regions_name)
    np.save(config.paths.output / "all_countries.npy", countries_iso)
    np.save(config.paths.output / "all_countries_names.npy", countries_name)


# TODO: Probably not necessary, or could be done more compactly by looping over data_vars
ENCODING = {
    "Region": {"dtype": "str"},
    "Scenario": {"dtype": "str"},
    "Time": {"dtype": "int"},
    "Temperature": {"dtype": "float"},
    "NonCO2red": {"dtype": "float"},
    "NegEmis": {"dtype": "float"},
    "Risk": {"dtype": "float"},
    "Timing": {"dtype": "str"},
    "Conditionality": {"dtype": "str"},
    "Ambition": {"dtype": "str"},
    "GDP": {"zlib": True, "complevel": 9},
    "Population": {"zlib": True, "complevel": 9},
    "GHG_hist": {"zlib": True, "complevel": 9},
    "GHG_hist_excl": {"zlib": True, "complevel": 9},
    "CO2_hist": {"zlib": True, "complevel": 9},
    "CO2_hist_excl": {"zlib": True, "complevel": 9},
    "GHG_globe": {"zlib": True, "complevel": 9},
    "GHG_globe_excl": {"zlib": True, "complevel": 9},
    "CO2_globe": {"zlib": True, "complevel": 9},
    "CO2_globe_excl": {"zlib": True, "complevel": 9},
    "GHG_base_incl": {"zlib": True, "complevel": 9},
    "GHG_base_excl": {"zlib": True, "complevel": 9},
    "CO2_base_incl": {"zlib": True, "complevel": 9},
    "CO2_base_excl": {"zlib": True, "complevel": 9},
    "GHG_excl_C": {"zlib": True, "complevel": 9},
    "CO2_excl_C": {"zlib": True, "complevel": 9},
    "CO2_neg_C": {"zlib": True, "complevel": 9},
    "CO2_bunkers_C": {"zlib": True, "complevel": 9},
    "GHG_ndc": {"zlib": True, "complevel": 9},
    "GHG_ndc_excl": {"zlib": True, "complevel": 9},
    "GHG_ndc_excl_CR": {"zlib": True, "complevel": 9},
}


def save_total(config: Config, xr_version):
    """Save xr_total to netcdf file."""

    startyear = config.params.start_year_analysis
    root = config.paths.output / f"startyear_{startyear}"
    root.mkdir(parents=True, exist_ok=True)
    savepath = root / "xr_dataread.nc"

    logger.info(f"Saving xr_total to {savepath}")

    xr_version.to_netcdf(
        savepath,
        encoding=ENCODING,
        format="NETCDF4",
        engine="netcdf4",
    )


def save_rbw(config: Config, xr_version, countries):
    """Save rbw factors to netcdf file."""
    startyear = config.params.start_year_analysis
    savepath = config.paths.output / f"startyear_{startyear}"

    logger.info(f"Saving rbw factors to {savepath}")

    countries_iso = np.array(list(countries.values()))

    # AP rbw factors
    # TODO write as single file with 
    # CO2-incl, CO2-excl, GHG-incl, GHG-excl columns
    # so we can use @intermediate_file decorator
    for gas in ["CO2", "GHG"]:
        for lulucf_i, lulucf in enumerate(["incl", "excl"]):
            luext = ["", "_excl"][lulucf_i]
            xrt = xr_version.sel(Time=np.arange(config.params.start_year_analysis, 2101))
            r1_nom = xrt.GDP.sel(Region="EARTH") / xrt.Population.sel(Region="EARTH")
            base_worldsum = xrt[gas + "_base_" + lulucf].sel(Region="EARTH")
            rb_part1 = (xrt.GDP / xrt.Population / r1_nom) ** (1 / 3.0)
            rb_part2 = (
                xrt[gas + "_base_" + lulucf]
                * (base_worldsum - xrt[gas + "_globe" + luext])
                / base_worldsum
            )
            rbw = (rb_part1 * rb_part2).sel(Region=countries_iso).sum(dim="Region")
            rbw = rbw.where(rbw != 0)
            rbw.to_netcdf(savepath / f"xr_rbw_{gas}_{lulucf}.nc")


@intermediate_file("xr_rci.nc")
def load_rci(config: Config, region_dim) -> xr.Dataset:
    """Load responsibility capability index (RCI) data from netcdf file."""

    # GDR RCI indices
    r = 0
    hist_emissions_startyears = [1850, 1950, 1990]
    capability_thresholds = ["No", "Th", "PrTh"]
    rci_weights = ["Resp", "Half", "Cap"]
    for startyear_i, startyear in enumerate(hist_emissions_startyears):
        for th_i, th in enumerate(capability_thresholds):
            for weight_i, weight in enumerate(rci_weights):
                # Read RCI
                df_rci = pd.read_csv(
                    config.paths.input / "RCI" / f"GDR_15_{startyear}_{th}_{weight}.xls",
                    delimiter="\t",
                    skiprows=30,
                )[:-2]
                df_rci = df_rci[["iso3", "year", "rci"]]
                iso3 = np.array(df_rci.iso3)
                iso3[iso3 == "CHK"] = "CHN"
                df_rci["iso3"] = iso3
                df_rci["year"] = df_rci["year"].astype(int)
                df_rci = df_rci.rename(columns={"iso3": "Region", "year": "Time"})
                df_rci["Historical_startyear"] = startyear
                df_rci["Capability_threshold"] = th
                df_rci["RCI_weight"] = weight
                if r == 0:
                    fulldf = df_rci
                    r += 1
                else:
                    fulldf = pd.concat([fulldf, df_rci])
    dfdummy = fulldf.set_index(
        ["Region", "Time", "Historical_startyear", "Capability_threshold", "RCI_weight"]
    )

    return xr.Dataset.from_dataframe(dfdummy).reindex({"Region": region_dim})
