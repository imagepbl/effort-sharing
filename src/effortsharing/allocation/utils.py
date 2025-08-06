from typing import Literal

import xarray as xr

from effortsharing.config import Config
from effortsharing.pathways.co2_trajectories import determine_global_co2_trajectories

Gas = Literal["CO2", "GHG"]
LULUCF = Literal["incl", "excl"]

# =========================================================== #
# =========================================================== #


class InvalidAssumptionSetError(Exception):
    def __init__(self, gas: Gas, lulucf: LULUCF):
        super().__init__(
            f"Invalid assumption set with gas={gas} and lulucf={lulucf}."
            "Please use 'CO2' or 'GHG' for gas and 'incl' or 'excl' for LULUCF."
        )


def config2base_var(
    gas: Gas, lulucf: LULUCF
) -> Literal["CO2_base_incl", "CO2_base_excl", "GHG_base_incl", "GHG_base_excl"]:
    if lulucf == "incl" and gas == "CO2":
        return "CO2_base_incl"
    elif lulucf == "incl" and gas == "GHG":
        return "GHG_base_incl"
    elif lulucf == "excl" and gas == "CO2":
        return "CO2_base_excl"
    elif lulucf == "excl" and gas == "GHG":
        return "GHG_base_excl"
    raise InvalidAssumptionSetError(gas, lulucf)


def config2hist_var(
    gas: Gas, lulucf: LULUCF
) -> Literal["GHG_hist", "GHG_hist_excl", "CO2_hist", "CO2_hist_excl"]:
    if lulucf == "incl" and gas == "GHG":
        return "GHG_hist"
    elif lulucf == "excl" and gas == "GHG":
        return "GHG_hist_excl"
    elif lulucf == "incl" and gas == "CO2":
        return "CO2_hist"
    elif lulucf == "excl" and gas == "CO2":
        return "CO2_hist_excl"
    raise InvalidAssumptionSetError(gas, lulucf)


def config2globe_var(
    gas: Gas, lulucf: LULUCF
) -> Literal["GHG_globe", "GHG_globe_excl", "CO2_globe", "CO2_globe_excl"]:
    if lulucf == "incl" and gas == "GHG":
        return "GHG_globe"
    elif lulucf == "excl" and gas == "GHG":
        return "GHG_globe_excl"
    elif lulucf == "incl" and gas == "CO2":
        return "CO2_globe"
    elif lulucf == "excl" and gas == "CO2":
        return "CO2_globe_excl"
    raise InvalidAssumptionSetError(gas, lulucf)


def load_future_emissions(config: Config, gas: Gas, lulucf: LULUCF):
    all_projected_gases = determine_global_co2_trajectories(config)
    globe_var = config2globe_var(gas, lulucf)
    return all_projected_gases[globe_var]


def load_dataread(config: Config) -> xr.Dataset:
    start_year_analysis = config.params.start_year_analysis
    total_xr = xr.open_dataset(
        config.paths.output / f"startyear_{start_year_analysis}" / "xr_dataread.nc"
    )
    return total_xr


def load_population(config: Config) -> xr.DataArray:
    socioeconomic_data = load_dataread(config)
    return socioeconomic_data.Population
    # TODO find socioeconomic_data that has Time=2021 as socioeconomics.nc does not,
    # TODO and remove reading of xr_dataread.nc
    socioeconomic_data = load_socioeconomics(config.config)
    return socioeconomic_data.Population


# TODO Move load functions elsewhere
