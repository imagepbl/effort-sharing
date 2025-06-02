# ======================================== #
# Class that does the budget allocation
# ======================================== #

# =========================================================== #
# PREAMBULE
# Put in packages that we need
# =========================================================== #

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

from effortsharing.config import Config
from effortsharing.input.emissions import load_emissions
from effortsharing.input.socioeconomics import load_socioeconomics
from effortsharing.world import (
    determine_global_budgets,
    determine_global_co2_trajectories,
    determine_global_nonco2_trajectories,
    nonco2variation,
)


@dataclass
class AllocationConfig:
    config: Config
    region: str
    gas: Literal["CO2", "GHG"] = "GHG"
    lulucf: Literal["incl", "excl"] = "incl"


# =========================================================== #
# =========================================================== #

# TODO move config2*_var elsewhere. 

def config2base_var(aconfig: AllocationConfig) -> Literal['CO2_base_incl', 'CO2_base_excl',
                                                            'GHG_base_incl', 'GHG_base_excl']:
    if aconfig.lulucf == "incl" and aconfig.gas == "CO2":
        return "CO2_base_incl"
    elif aconfig.lulucf == "incl" and aconfig.gas == "GHG":
        return "GHG_base_incl"
    elif aconfig.lulucf == "excl" and aconfig.gas == "CO2":
        return "CO2_base_excl"
    elif aconfig.lulucf == "excl" and aconfig.gas == "GHG":
        return "GHG_base_excl"
    raise ValueError(
        "Invalid combination of LULUCF and gas. "
        "Please use 'incl' or 'excl' for LULUCF and 'CO2' or 'GHG' for gas."
    )


def config2hist_var(aconfig: AllocationConfig) -> Literal['GHG_hist', 'GHG_hist_excl', 
                                                         'CO2_hist', 'CO2_hist_excl']:
    if aconfig.lulucf == "incl" and aconfig.gas == "GHG":
        return "GHG_hist"
    elif aconfig.lulucf == "excl" and aconfig.gas == "GHG":
        return "GHG_hist_excl"
    elif aconfig.lulucf == "incl" and aconfig.gas == "CO2":
        return "CO2_hist"
    elif aconfig.lulucf == "excl" and aconfig.gas == "CO2":
        return "CO2_hist_excl"
    raise ValueError(
        "Invalid combination of LULUCF and gas. "
        "Please use 'incl' or 'excl' for LULUCF and 'CO2' or 'GHG' for gas."
    )


def config2globe_var(aconfig: AllocationConfig) -> Literal['GHG_globe', 'GHG_globe_excl',
                                                            'CO2_globe', 'CO2_globe_excl']:
    if aconfig.lulucf == "incl" and aconfig.gas == "GHG":
        return "GHG_globe"
    elif aconfig.lulucf == "excl" and aconfig.gas == "GHG":
        return "GHG_globe_excl"
    elif aconfig.lulucf == "incl" and aconfig.gas == "CO2":
        return "CO2_globe"
    elif aconfig.lulucf == "excl" and aconfig.gas == "CO2":
        return "CO2_globe_excl"
    raise ValueError(
        "Invalid combination of LULUCF and gas. "
        "Please use 'incl' or 'excl' for LULUCF and 'CO2' or 'GHG' for gas."
    )

# TODO Move load functions elsewhere

def load_emissions_and_scenarios(aconfig: AllocationConfig):
    hist_var = config2hist_var(aconfig)
    emission_data, scenarios = load_emissions(aconfig.config)
    return hist_var, emission_data, scenarios


def load_future_emissions(aconfig: AllocationConfig, emission_data, scenarios):
    all_projected_gases = load_global_co2_trajectories(aconfig.config, emission_data, scenarios)
    globe_var = config2globe_var(aconfig)
    return all_projected_gases[globe_var]

def load_global_co2_trajectories(config: Config, emission_data, scenarios):
    xr_temperatures, xr_nonco2warming_wrt_start = nonco2variation(config)
    (xr_traj_nonco2,) = determine_global_nonco2_trajectories(
        config, emission_data, scenarios, xr_temperatures
    )
    _, xr_co2_budgets = determine_global_budgets(
        config, emission_data, xr_temperatures, xr_nonco2warming_wrt_start
    )
    (all_projected_gases,) = determine_global_co2_trajectories(
        config=config,
        emissions=emission_data,
        scenarios=scenarios,
        xr_temperatures=xr_temperatures,
        xr_co2_budgets=xr_co2_budgets,
        xr_traj_nonco2=xr_traj_nonco2,
    )
    
    return all_projected_gases


def load_dataread(config: Config) -> xr.Dataset:
    start_year_analysis= config.params.start_year_analysis
    total_xr = xr.open_dataset(
        config.paths.output / f'startyear_{start_year_analysis}' / "xr_dataread.nc"
    ).load()
    return total_xr

def load_population(config: Config) -> xr.DataArray:
    socioeconomic_data = load_dataread(config)
    # TODO find socioeconomic_data that has Time=2021 as socioeconomics.nc does not, 
    # TODO and remove reading of xr_dataread.nc
    # socioeconomic_data = load_socioeconomics(config.config)

    return socioeconomic_data.Population

# =========================================================== #
# allocation methods
# =========================================================== #

def gf(aconfig: AllocationConfig) -> xr.DataArray:
    """
    Grandfathering: Divide the global budget over the regions based on
    their historical CO2 emissions
    """
    start_year_analysis= aconfig.config.params.start_year_analysis
    analysis_timeframe = np.arange(start_year_analysis, 2101)

    hist_var, emission_data, scenarios = load_emissions_and_scenarios(aconfig)
    emis_fut = load_future_emissions(aconfig, emission_data, scenarios)

    # Calculating the current CO2 fraction for region and world based on start_year_analysis
    current_co2_region = emission_data[hist_var].sel(Region=aconfig.region, Time=start_year_analysis)

    current_co2_earth = 1e-9 + emission_data[hist_var].sel(Region="EARTH", Time=start_year_analysis)

    co2_fraction = current_co2_region / current_co2_earth

    # New CO2 time series from the start_year to 2101 by multiplying global budget with fraction
    xr_new_co2 = (co2_fraction * emis_fut.sel(Time=analysis_timeframe))

    return xr_new_co2


# =========================================================== #
# =========================================================== #

def pc(aconfig: AllocationConfig) -> xr.DataArray:
    """
    Per Capita: Divide the global budget equally per capita
    """
    start_year_analysis= aconfig.config.params.start_year_analysis
    analysis_timeframe = np.arange(start_year_analysis, 2101)

    population = load_population(aconfig.config)
    # TODO use function compute countries or read from file
    countries_iso_path = aconfig.config.paths.output / "all_countries.npy"
    countries_iso = np.load(
            countries_iso_path, allow_pickle=True
    )
    pop_region = population.sel(
        Region=aconfig.region, Time=start_year_analysis
    ).Population
    pop_earth = population.sel(
        Region=countries_iso, Time=start_year_analysis
    ).sum(dim=["Region"])
    pop_fraction = pop_region / pop_earth

    # Multiplying the global budget with the population fraction to create
    # new allocation time series from start_year to 2101
    hist_var, emission_data, scenarios = load_emissions_and_scenarios(aconfig)
    emis_fut = load_future_emissions(aconfig, emission_data, scenarios)

    xr_new = (pop_fraction * emis_fut).sel(Time=analysis_timeframe)
    return xr_new

# =========================================================== #
# =========================================================== #

def pcc(aconfig: AllocationConfig, gf_da: xr.DataArray, pc_da: xr.DataArray) -> xr.DataArray:
    """
    Per Capita Convergence: Grandfathering converging into per capita
    """
    start_year_analysis= aconfig.config.params.start_year_analysis

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

    dim_convyears = aconfig.config.dimension_ranges.convergence_years

    times=np.arange(1850, 2101)
    for year in dim_convyears:
        ar = np.array([transform_time(times, year)]).T
        coords = {"Time": times, "Convergence_year": [year]}
        dims = ["Time", "Convergence_year"]
        gfdeel.append(xr.DataArray(data=ar, dims=dims, coords=coords).to_dataset(name="PCC"))
        pcdeel.append(
            xr.DataArray(data=1 - ar, dims=dims, coords=coords).to_dataset(name="PCC")
        )

    # Merging the list of DataArays into one Dataset
    gfdeel_single = xr.merge(gfdeel)
    pcdeel_single = xr.merge(pcdeel)

    # Creating new allocation time series by multiplying convergence fractions
    # with existing GF and PC allocations
    xr_new = (gfdeel_single * gf_da + pcdeel_single * pc_da)["PCC"]
    return xr_new

# =========================================================== #
# =========================================================== #

def pcb(aconfig: AllocationConfig) -> xr.DataArray:
    """
    Per capita on a budget basis
    """
    start_year= aconfig.config.params.start_year_analysis
    focus_region = aconfig.region

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

    population = load_population(aconfig.config)
    pop_region = population.sel(Time=start_year)
    pop_earth = population.sel(Region="EARTH", Time=start_year)
    pop_fraction = (pop_region / pop_earth).mean(dim="Scenario")

    hist_var, emission_data, scenarios = load_emissions_and_scenarios(aconfig)
    emis_fut = load_future_emissions(aconfig, emission_data, scenarios)
    globalpath = emis_fut

    hist_var_co2, emission_data_co2, scenarios_co2 = load_emissions_and_scenarios(
        # TODO is it correct that global_path can use GHG but emis_start_* always uses CO2?
        AllocationConfig(
            config=aconfig.config, 
            region=aconfig.region, 
            lulucf=aconfig.lulucf,
            gas="CO2",
        )
    )
    emis_start_i = emission_data_co2[hist_var_co2].sel(Time=start_year)
    emis_start_w = emission_data_co2[hist_var_co2].sel(Time=start_year, Region="EARTH")

    time_range = np.arange(start_year, 2101)
    path_scaled_0 = (
        (emis_start_i / emis_start_w * globalpath).sel(Time=time_range).sel(Region=focus_region)
    )

    budget_left = (
        emis_fut.where(emis_fut > 0, 0).sel(Time=time_range).sum(dim="Time")
        * pop_fraction
    ).sel(Region=focus_region)
    # TODO compute budget on the fly or read from file. Instea of reading xr_dataread.nc
    xr_total = load_dataread(aconfig.config)
    co2_budget_left = (xr_total.Budget * pop_fraction).sel(Region=focus_region) * 1e3

    budget_without_assumptions_prepeak = path_scaled_0.where(path_scaled_0 > 0, 0).sum(
        dim="Time"
    )

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
    co2_hist = emission_data_co2[hist_var_co2].sel(Region=focus_region, Time=start_year)
    time_range = np.arange(start_year, 2101)

    nz = co2_budget_left * 2 / co2_hist + start_year - 1
    coef = co2_hist / (nz - start_year)

    linear_co2 = (
        -coef
        * xr.DataArray(
            np.arange(0, 2101 - start_year), dims=["Time"], coords={"Time": time_range}
        )
        + co2_hist
    )

    linear_co2_pos = linear_co2.where(linear_co2 > 0, 0).to_dataset(name="PCB_lin")

    # Now, if we want GHG, the non-CO2 part is added:
    if aconfig.gas == "GHG":
        # Non-co2 part
        hist_var_ghg, emission_data_ghg, scenarios_ghg = load_emissions_and_scenarios(
            # TODO is it correct that global_path can use GHG but emis_start_* always uses CO2?
            AllocationConfig(
                config=aconfig.config, 
                region=aconfig.region, 
                lulucf=aconfig.lulucf,
                gas="GHG",
            )
        )
        nonco2_current = emission_data_ghg[hist_var_ghg].sel(
            Time=start_year
        ) - emission_data_co2[hist_var_co2].sel(Time=start_year)

        nonco2_fraction = nonco2_current / nonco2_current.sel(Region="EARTH")
        nonco2_globe = load_global_co2_trajectories(
            config=aconfig.config,
            emission_data=emission_data_ghg,
            scenarios=scenarios_ghg
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
    elif aconfig.gas == "_CO2":
        # together:
        ghg_pcb = pcb
        ghg_pcb_lin = linear_co2_pos
    else:
        raise ValueError(
            "Invalid gas type. Please use 'GHG' or 'CO2'."
        )

    return ghg_pcb.PCB, ghg_pcb_lin.PCB_lin


# =========================================================== #
# =========================================================== #
def ecpc(aconfig: AllocationConfig) -> xr.DataArray:
    """
    Equal Cumulative per Capita: Uses historical emissions, discount factors and
    population shares to allocate the global budget
    """
    start_year_analysis= aconfig.config.params.start_year_analysis
    focus_region = aconfig.region
    dim_discountrates = aconfig.config.dimension_ranges.discount_rates
    dim_histstartyear = aconfig.config.dimension_ranges.hist_emissions_startyears
    dim_convyears = aconfig.config.dimension_ranges.convergence_years
    analysis_timeframe = np.arange(start_year_analysis, 2101)

    # Defining the timeframes for historical and future emissions
    population_data = load_population(aconfig.config)
    current_population_data = population_data.sel(Time=analysis_timeframe)

    hist_var = config2hist_var(aconfig)
    emission_data, scenarios = load_emissions(aconfig.config)
    global_emissions_future = load_future_emissions(AllocationConfig(
        config=aconfig.config,
        region=focus_region,
        lulucf='incl',
        gas='GHG'
    ), emission_data, scenarios).sel(Time=analysis_timeframe)
    GHG_hist = emission_data.GHG_hist

    GF_frac = GHG_hist.sel(
        Time=start_year_analysis, Region=focus_region
    ) / GHG_hist.sel(Time=start_year_analysis, Region="EARTH")
    share_popt = current_population_data / current_population_data.sel(Region="EARTH")
    share_popt_past = population_data / population_data.sel(Region="EARTH")

    xr_ecpc_all_list = []

    # Precompute reusable variables
    hist_emissions_timeframes = [
        np.arange(startyear, 1 + start_year_analysis)
        for startyear in dim_histstartyear
    ]
    past_timelines = [
        np.arange(startyear, start_year_analysis + 1)
        for startyear in dim_histstartyear
    ]
    discount_factors = np.array(dim_discountrates)

    for startyear, hist_emissions_timeframe, past_timeline in zip(
        dim_histstartyear, hist_emissions_timeframes, past_timelines
    ):
        hist_emissions = emission_data[hist_var].sel(Time=hist_emissions_timeframe)
        discount_period = start_year_analysis - past_timeline

        # Vectorize discount factor application
        xr_discount = xr.DataArray(
            (1 - discount_factors[:, None] / 100) ** discount_period,
            dims=["Discount_factor", "Time"],
            coords={"Discount_factor": discount_factors, "Time": past_timeline},
        )
        hist_emissions_rt = hist_emissions * xr_discount
        hist_emissions_wt = hist_emissions_rt.sel(Region="EARTH")
        historical_leftover = (
            (share_popt_past * hist_emissions_wt - hist_emissions_rt)
            .sel(Time=np.arange(startyear, 2020 + 1))
            .sum(dim="Time")
            .sel(Region=focus_region)
        )

        for conv_year in dim_convyears:
            max_time_steps = conv_year - 2021
            emissions_ecpc = global_emissions_future.sel(Time=2021) * GF_frac
            emissions_rightful_at_year = (
                global_emissions_future
                * current_population_data.sel(Region=focus_region)
                / current_population_data.sel(Region="EARTH")
            )
            historical_leftover_updated = (
                historical_leftover
                - emissions_ecpc
                + emissions_rightful_at_year.sel(Time=[2021]).sum(dim="Time")
            )

            # Precompute sine values
            sine_values = np.sin(np.arange(1, max_time_steps) / max_time_steps * np.pi) * 3

            # Initialize list to store emissions
            es = [emissions_ecpc]

            # Emissions calculation
            for t in range(2100 - start_year_analysis):
                time_step = 2022 + t
                globe_new = global_emissions_future.sel(Time=time_step)
                pop_frac = share_popt.sel(Time=time_step, Region=focus_region)
                if t < max_time_steps - 1:
                    Delta_L = historical_leftover_updated / (max_time_steps - t)
                    emissions_ecpc = Delta_L * sine_values[t] + globe_new * (
                        GF_frac * (1 - (t + 1) / max_time_steps)
                        + pop_frac * ((t + 1) / max_time_steps)
                    )
                    historical_leftover_updated = (
                        historical_leftover_updated
                        - emissions_ecpc
                        + emissions_rightful_at_year.sel(Time=time_step)
                    )
                    es.append(emissions_ecpc.expand_dims({"Time": [time_step]}))
                elif t == max_time_steps - 1:
                    emissions_ecpc = (
                        pop_frac * globe_new * 0.67 + es[-1].sel(Time=time_step - 1) * 0.33
                    )
                    es.append(emissions_ecpc.expand_dims({"Time": [time_step]}))
                else:
                    emissions_ecpc = pop_frac * globe_new
                    es.append(emissions_ecpc.expand_dims({"Time": [time_step]}))

            # TODO is coords='minimal' correct here? Without gave error:
            # ValueError: 'Region' not present in all datasets and coords='different'. Either add 'Region' to datasets where it is missing or specify coords='minimal'.
            xr_ecpc_alloc = xr.concat(es, dim="Time", coords='minimal')
            xr_ecpc_all_list.append(
                xr_ecpc_alloc.expand_dims(
                    {"Historical_startyear": [startyear], "Convergence_year": [conv_year]}
                ).to_dataset(name="ECPC")
            )

    xr_ecpc_all = xr.merge(xr_ecpc_all_list)
    # Create the correct order of dimensions
    xr_ecpc_all = xr_ecpc_all.transpose(
        "Discount_factor",
        "Historical_startyear",
        "Convergence_year",
        "NegEmis",
        "NonCO2red",
        "Temperature",
        "Risk",
        "Timing",
        "Time",
        "Scenario",
    )

    return xr_ecpc_all.ECPC


# =========================================================== #
# =========================================================== #

def ap(aconfig: AllocationConfig) -> xr.DataArray:
    """
    Ability to Pay: Uses GDP per capita to allocate the global budget
    Equation from van den Berg et al. (2020)
    """
    start_year_analysis= aconfig.config.params.start_year_analysis
    analysis_timeframe = np.arange(start_year_analysis, 2101)
    focus_region = aconfig.region

    # Step 1: Reductions before correction factor
    # TODO replace with load_socioeconomics() function
    xrt = load_dataread(aconfig.config).sel(Time=analysis_timeframe)
    GDP_sum_w = xrt.GDP.sel(Region="EARTH")
    pop_sum_w = xrt.Population.sel(Region="EARTH")
    # Global average GDP per capita
    r1_nom = GDP_sum_w / pop_sum_w

    emission_data, scenarios = load_emissions(aconfig.config)
    emis_base_var = config2base_var(aconfig)
    emis_base = emission_data[emis_base_var]
    emis_fut = load_future_emissions(aconfig, emission_data, scenarios)

    base_worldsum = emis_base.sel(Time=analysis_timeframe).sel(Region="EARTH")
    rb_part1 = (
        xrt.GDP.sel(Region=focus_region)
        / xrt.Population.sel(Region=focus_region)
        / r1_nom
    ) ** (1 / 3.0)
    rb_part2 = (
        emis_base.sel(Time=analysis_timeframe).sel(Region=focus_region)
        * (base_worldsum - emis_fut.sel(Time=analysis_timeframe))
        / base_worldsum
    )
    rb = rb_part1 * rb_part2

    # Step 2: Correction factor
    # TODO replace open with load_rbw() function, will need to find where files are written
    rbw_path = aconfig.config.paths.output / f"startyear_{start_year_analysis}" / f"xr_rbw_{aconfig.gas}_{aconfig.lulucf}.nc"
    rbw = xr.open_dataset(rbw_path).load()
    corr_factor = (1e-9 + rbw.__xarray_dataarray_variable__) / (
        base_worldsum - emis_fut.sel(Time=analysis_timeframe)
    )

    # Step 3: Budget after correction factor
    ap = emis_base.sel(Region=focus_region) - rb / corr_factor

    ap = ap.sel(Time=analysis_timeframe)
    return ap

# =========================================================== #
# =========================================================== #

def gdr(aconfig: AllocationConfig, ap_da: xr.DataArray) -> xr.DataArray:
    """
    Greenhouse Development Rights: Uses the Responsibility-Capability Index
    (RCI) weighed at 50/50 to allocate the global budget
    Calculations from van den Berg et al. (2020)
    """
    start_year_analysis= aconfig.config.params.start_year_analysis
    analysis_timeframe = np.arange(start_year_analysis, 2101)
    focus_region = aconfig.region
    convergence_year_gdr = aconfig.config.params.convergence_year_gdr

    # TODO if file does not exist, create it with world.save_rci() + @intermediate_file decorator
    xr_rci_path = aconfig.config.paths.output / "xr_rci.nc"
    xr_rci = xr.open_dataset(xr_rci_path).load()
    yearfracs = xr.Dataset(
        data_vars={
            "Value": (
                ["Time"],
                (analysis_timeframe - 2030)
                / (convergence_year_gdr - 2030),
            )
        },
        coords={"Time": analysis_timeframe},
    )

    # Get the regional RCI values
    # If region is EU, we have to sum over the EU countries
    if focus_region != "EU":
        rci_reg = xr_rci.rci.sel(Region=focus_region)
    else:
        fn = aconfig.config.paths.input / "UNFCCC_Parties_Groups_noeu.xlsx"
        df = pd.read_excel(
            fn, sheet_name="Country groups"
        )
        countries_iso = np.array(df["Country ISO Code"])
        group_eu = countries_iso[np.array(df["EU"]) == 1]
        rci_reg = xr_rci.rci.sel(Region=group_eu).sum(dim="Region")

    # Compute GDR until 2030
    emission_data, scenarios = load_emissions(aconfig.config)
    emis_base_var = config2base_var(aconfig)
    emis_base = emission_data[emis_base_var]
    emis_fut = load_future_emissions(aconfig, emission_data, scenarios)
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
    return gdr_total.GDR


def save(config: AllocationConfig, 
         dss: dict[str, xr.DataArray]
         ):
    """
    Combine data arrays returned by each allocation method into a NetCDF file
    """
    fn = f"xr_alloc_{config.region}.nc"
    # TODO refactor or remove?
    # if self.dataread_file != "xr_dataread.nc":
    #     savename = "xr_alloc_" + self.focus_region + "_adapt.nc"
    save_path = config.config.paths.output / f"Allocations_{config.gas}_{config.lulucf}" / fn

    start_year_analysis = config.config.params.start_year_analysis
    # TODO move to config.config.params
    end_year_analysis = 2101

    combined = (
        xr.Dataset(data_vars=dss)
            .sel(Time=np.arange(start_year_analysis, end_year_analysis))
            .astype("float32")
    )
    combined.to_netcdf(save_path, format="NETCDF4")


        

if __name__ == "__main__":
    # region = input("Choose a focus country or region: ")
    region = 'BRA'
    config = Config.from_file("notebooks/config.yml")
    aconfig = AllocationConfig(
        config=config,
        region=region,
        gas="GHG",
        lulucf="incl"
    )
    # gf_da = gf(aconfig)
    # pc_da = pc(aconfig)
    # pcc_da = pcc(aconfig, gf_da, pc_da)
    # pcb_da, pcb_lin_da = pcb(aconfig)
    # ecpc_da = ecpc(aconfig)
    ap_da = ap(aconfig)
    # print(ap_da)
    gdr_da = gdr(aconfig, ap_da)
    print(gdr_da)
    # save(
    #     config=aconfig,
    #     dss=dict(
    #         gf=gf_da,
    #         pc=pc_da,
    #         pcc=pcc_da,
    #         pcb=pcb_da,
    #         ecpc=ecpc_da,
    #         ap=ap_da,
    #         gdr=gdr_da,
    #     )
    # )

