import logging

import numpy as np
import xarray as xr

from effortsharing.allocation.utils import (
    LULUCF,
    Gas,
    config2hist_var,
    load_future_emissions,
    load_population,
)
from effortsharing.config import Config
from effortsharing.input.emissions import load_emissions, read_modelscenarios

logger = logging.getLogger(__name__)


def ecpc(config: Config, region, gas: Gas = "GHG", lulucf: LULUCF = "incl") -> xr.DataArray:
    """
    Equal Cumulative per Capita: Uses historical emissions, discount factors and
    population shares to allocate the global budget
    """
    logger.info(f"Computing Equal Cumulative per Capita allocation for {region}")
    start_year_analysis = config.params.start_year_analysis
    focus_region = region
    dim_discountrates = config.dimension_ranges.discount_rates
    dim_histstartyear = config.dimension_ranges.hist_emissions_startyears
    dim_convyears = config.dimension_ranges.convergence_years
    analysis_timeframe = np.arange(start_year_analysis, 2101)

    # Defining the timeframes for historical and future emissions
    population_data = load_population(config)
    current_population_data = population_data.sel(Time=analysis_timeframe)

    hist_var = config2hist_var(gas, lulucf)
    emission_data = load_emissions(config)
    scenarios = read_modelscenarios(config)
    global_emissions_future = load_future_emissions(
        config=config,
        emission_data=emission_data,
        scenarios=scenarios,
        lulucf="incl",
        gas="GHG",
    ).sel(Time=analysis_timeframe)
    GHG_hist = emission_data.GHG_hist

    GF_frac = GHG_hist.sel(Time=start_year_analysis, Region=focus_region) / GHG_hist.sel(
        Time=start_year_analysis, Region="EARTH"
    )
    share_popt = current_population_data / current_population_data.sel(Region="EARTH")
    share_popt_past = population_data / population_data.sel(Region="EARTH")

    xr_ecpc_all_list = []

    # Precompute reusable variables
    hist_emissions_timeframes = [
        np.arange(startyear, 1 + start_year_analysis) for startyear in dim_histstartyear
    ]
    past_timelines = [
        np.arange(startyear, start_year_analysis + 1) for startyear in dim_histstartyear
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
            # ValueError: 'Region' not present in all datasets and coords='different'.
            #   Either add 'Region' to datasets where it is missing or specify coords='minimal'.
            xr_ecpc_alloc = xr.concat(es, dim="Time", coords="minimal")
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

    logger.info(f"Computed Equal Cumulative per Capita allocation for {region}")
    return xr_ecpc_all.ECPC
