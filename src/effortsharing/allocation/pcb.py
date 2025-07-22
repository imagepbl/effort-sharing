import logging

import numpy as np
import xarray as xr

from effortsharing.allocation.utils import (
    LULUCF,
    Gas,
    config2hist_var,
    load_dataread,
    load_future_emissions,
    load_global_co2_trajectories,
    load_population,
)
from effortsharing.config import Config
from effortsharing.input.emissions import load_emissions, read_modelscenarios

logger = logging.getLogger(__name__)


def pcb(
    config: Config, region, gas: Gas = "GHG", lulucf: LULUCF = "incl"
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Per capita on a budget basis
    """
    logger.info(f"Computing Per Capita Budget allocation for {region}")
    start_year = config.params.start_year_analysis
    focus_region = region

    # co2 part
    def budget_harm(nz):
        end_year = 2101
        compensation_form = np.sqrt(np.arange(0, end_year - start_year))
        xr_comp2 = xr.DataArray(
            compensation_form, dims=["Time"], coords={"Time": np.arange(start_year, end_year)}
        )
        return xr_comp2 / ((nz - start_year) ** (3 / 2) * (2 / 3))
        # TODO later: should be , but I now calibrated to 0.5.
        # Not a problem because we have the while loop later.

    def pcb_new_factor(path, f):
        positive_path = path.where(path > 0, 0)
        negative_path = positive_path.where(path < 0, 1)

        netzeros = start_year + negative_path.sum(dim="Time")
        netzeros = netzeros.where(netzeros < 2100, 2100)

        return path + budget_harm(netzeros) * f

    population = load_population(config)
    pop_region = population.sel(Time=start_year)
    pop_earth = population.sel(Region="EARTH", Time=start_year)
    pop_fraction = (pop_region / pop_earth).mean(dim="Scenario")

    emission_data = load_emissions(config)
    scenarios = read_modelscenarios(config)
    emis_fut = load_future_emissions(config, emission_data, scenarios, gas, lulucf)
    globalpath = emis_fut

    hist_var_co2 = config2hist_var(gas="CO2", lulucf=lulucf)
    emis_start_i = emission_data[hist_var_co2].sel(Time=start_year)
    emis_start_w = emission_data[hist_var_co2].sel(Time=start_year, Region="EARTH")

    time_range = np.arange(start_year, 2101)
    path_scaled_0 = (
        (emis_start_i / emis_start_w * globalpath).sel(Time=time_range).sel(Region=focus_region)
    )

    budget_left = (
        emis_fut.where(emis_fut > 0, 0).sel(Time=time_range).sum(dim="Time") * pop_fraction
    ).sel(Region=focus_region)
    # TODO compute budget on the fly or read from file. Instead of reading xr_dataread.nc
    xr_total = load_dataread(config)
    co2_budget_left = (xr_total.Budget * pop_fraction).sel(Region=focus_region) * 1e3

    budget_without_assumptions_prepeak = path_scaled_0.where(path_scaled_0 > 0, 0).sum(dim="Time")

    budget_surplus = co2_budget_left - budget_without_assumptions_prepeak
    pcb = pcb_new_factor(path_scaled_0, budget_surplus).to_dataset(name="PCB")

    # Optimize to bend the CO2 curves as close as possible to the CO2 budgets
    iterations = 3

    for _ in range(iterations):
        # Calculate the positive part of the CO2 path
        pcb_pos = pcb.where(pcb > 0, 0).sum(dim="Time")

        # Calculate the budget surplus
        budget_surplus = (co2_budget_left - pcb_pos).PCB

        # Adjust the CO2 path based on the budget surplus
        pcb = pcb_new_factor(pcb.PCB, budget_surplus).to_dataset(name="PCB")

    # CO2, but now linear
    co2_hist = emission_data[hist_var_co2].sel(Region=focus_region, Time=start_year)
    time_range = np.arange(start_year, 2101)

    nz = co2_budget_left * 2 / co2_hist + start_year - 1
    coef = co2_hist / (nz - start_year)

    linear_co2 = (
        -coef
        * xr.DataArray(np.arange(0, 2101 - start_year), dims=["Time"], coords={"Time": time_range})
        + co2_hist
    )

    linear_co2_pos = linear_co2.where(linear_co2 > 0, 0).to_dataset(name="PCB_lin")

    # Now, if we want GHG, the non-CO2 part is added:
    if gas == "GHG":
        # Non-co2 part
        hist_var_ghg = config2hist_var(gas="GHG", lulucf=lulucf)
        nonco2_current = emission_data[hist_var_ghg].sel(Time=start_year) - emission_data[
            hist_var_co2
        ].sel(Time=start_year)

        nonco2_fraction = nonco2_current / nonco2_current.sel(Region="EARTH")
        nonco2_globe = load_global_co2_trajectories(
            config=config, emission_data=emission_data, scenarios=scenarios
        ).NonCO2_globe
        nonco2_part_gf = nonco2_fraction * nonco2_globe

        pc_fraction = pop_region / pop_earth
        nonco2_part_pc = pc_fraction * nonco2_globe

        # Create an array that transitions linearly from 0 to 1 from start_year to 2039,
        # and then remains constant at 1 from 2040 to 2100.
        compensation_form = np.concatenate(
            [
                np.linspace(0, 1, len(np.arange(start_year, 2040))),
                np.ones(len(np.arange(2040, 2101))),
            ]
        )

        xr_comp = xr.DataArray(compensation_form, dims=["Time"], coords={"Time": time_range})

        nonco2_part = nonco2_part_gf * (1 - xr_comp) + nonco2_part_pc * xr_comp

        # together:
        nonco2_focus_region = nonco2_part.sel(Region=focus_region)
        ghg_pcb = pcb + nonco2_focus_region
        ghg_pcb_lin = linear_co2_pos + nonco2_focus_region
    elif gas == "CO2":
        # together:
        ghg_pcb = pcb
        ghg_pcb_lin = linear_co2_pos
    else:
        raise ValueError("Invalid gas type. Please use 'GHG' or 'CO2'.")

    logger.info(f"Computed Per Capita Budget allocation for {region}")
    return ghg_pcb.PCB.rename("PCB"), ghg_pcb_lin.PCB_lin.rename("PCB_lin")
