import logging

import numpy as np
import pandas as pd
import xarray as xr

from effortsharing.cache import intermediate_file
from effortsharing.config import Config
from effortsharing.input.emissions import load_emissions
from effortsharing.nonco2 import nonco2variation

logger = logging.getLogger(__name__)


@intermediate_file("global_co2_budgets.nc")
def determine_global_budgets(
    config: Config, emissions=None, temperatures=None, xr_nonco2warming_wrt_start=None
):
    logger.info("Get global CO2 budgets")

    if emissions is None:
        emissions = load_emissions(config)

    if temperatures is None or xr_nonco2warming_wrt_start is None:
        temperatures, xr_nonco2warming_wrt_start = nonco2variation(config)

    # Define input
    data_root = config.paths.input
    budget_data = "update_MAGICC_and_scenarios-budget.csv"

    # TODO: this can probably do without the rounding or casting to array
    dim_temp = np.array(config.dimension_ranges.peak_temperature).astype(float).round(2)
    dim_prob = np.array(config.dimension_ranges.risk_of_exceedance).round(2)
    dim_nonco2 = np.array(config.dimension_ranges.non_co2_reduction).round(2)

    # CO2 budgets from Forster,
    # Now without the warming update in Forster, to link to IPCC AR6
    df_budgets = pd.read_csv(data_root / budget_data)
    df_budgets = df_budgets[["dT_targets", "0.1", "0.17", "0.33", "0.5", "0.66", "0.83", "0.9"]]
    dummy = df_budgets.melt(id_vars=["dT_targets"], var_name="Probability", value_name="Budget")
    ar = np.array(dummy["Probability"])
    ar = ar.astype(float).round(2)
    ar[ar == 0.66] = 0.67
    dummy["Probability"] = ar
    dummy["dT_targets"] = dummy["dT_targets"].astype(float).round(1)
    dummy = dummy.set_index(["dT_targets", "Probability"])

    # Correct budgets based on startyear (Forster is from Jan 2020 and on)
    if config.params.start_year_analysis == 2020:
        budgets = dummy["Budget"]
    elif config.params.start_year_analysis > 2020:
        budgets = dummy["Budget"]
        for year in np.arange(2020, config.params.start_year_analysis):
            budgets -= float(emissions.sel(Region="EARTH", Time=year).CO2_hist) / 1e3
    elif config.params.start_year_analysis < 2020:
        budgets = dummy["Budget"]
        for year in np.arange(config.params.start_year_analysis, 2020):
            budgets += float(emissions.sel(Region="EARTH", Time=year).CO2_hist) / 1e3
    dummy["Budget"] = budgets

    xr_bud_co2 = xr.Dataset.from_dataframe(dummy)
    xr_bud_co2 = xr_bud_co2.rename(
        {"dT_targets": "Temperature"}
    )  # .sel(Temperature = [1.5, 1.7, 2.0])
    xr_bud_co2 = xr_bud_co2

    # Determine bunker emissions to subtract from global budget
    bunker_subtraction = []
    for t_i, t in enumerate(dim_temp):
        # Assuming bunker emissions have a constant fraction of global emissions (3.3%) -
        # https://www.pbl.nl/sites/default/files/downloads/pbl-2020-analysing-international-shipping-and-aviation-emissions-projections_4076.pdf
        bunker_subtraction += [3.3 / 100]

    Blist = np.zeros(shape=(len(dim_temp), len(dim_prob), len(dim_nonco2))) + np.nan

    def ms_temp(
        temp, risk
    ):  # 0.2 is quite wide, but useful for studying nonCO2 variation among scenarios (is a relatively metric anyway)
        return temperatures.ModelScenario[
            np.where(np.abs(temp - temperatures.Temperature.sel(Risk=risk)) < 0.2)[0]
        ].values

    for p_i, p in enumerate(dim_prob):
        a, b = np.polyfit(
            xr_bud_co2.Temperature, xr_bud_co2.sel(Probability=np.round(p, 2)).Budget, 1
        )
        for t_i, t in enumerate(dim_temp):
            ms = ms_temp(t, round(1 - p, 2))

            # This assumes that the budget from Forster implicitly includes the
            # median nonCO2 warming among scenarios that meet that Temperature
            # target Hence, only deviation (dT) from this median is interesting
            # to assess here
            dT = xr_nonco2warming_wrt_start.sel(
                ModelScenario=ms, Risk=round(1 - p, 2)
            ) - xr_nonco2warming_wrt_start.sel(ModelScenario=ms, Risk=round(1 - p, 2)).median(
                dim="ModelScenario"
            )
            median_budget = (a * t + b) * (1 - bunker_subtraction[t_i])
            for n_i, n in enumerate(dim_nonco2):
                dT_quantile = dT.quantile(
                    n, dim="ModelScenario"
                ).PeakWarming  # Assuming relation between T and B also holds around the T-value
                dB_quantile = a * dT_quantile
                Blist[t_i, p_i, n_i] = median_budget + dB_quantile
    data2 = xr.DataArray(
        Blist,
        coords={
            "Temperature": dim_temp,
            "Risk": (1 - dim_prob).astype(float).round(2),
            "NonCO2red": dim_nonco2,
        },
        dims=["Temperature", "Risk", "NonCO2red"],
    )
    xr_co2_budgets = xr.Dataset({"Budget": data2})

    return xr_co2_budgets
