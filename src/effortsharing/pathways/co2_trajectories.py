import logging

import numpy as np
import xarray as xr

from effortsharing.cache import intermediate_file
from effortsharing.config import Config
from effortsharing.pathways.global_budgets import determine_global_budgets
from effortsharing.input.emissions import load_emissions, read_modelscenarios
from effortsharing.pathways.nonco2 import determine_global_nonco2_trajectories, nonco2variation

logger = logging.getLogger(__name__)


@intermediate_file("global_co2_trajectories.nc")
def determine_global_co2_trajectories(
    config: Config,
    emissions=None,
    scenarios=None,
    xr_temperatures=None,
    xr_co2_budgets=None,
    xr_traj_nonco2=None,
):
    logger.info("Computing global co2 trajectories")

    # Load required input data if not provided
    if emissions is None:
        emissions = load_emissions(config)
    if scenarios is None:
        scenarios = read_modelscenarios(config)
    if xr_temperatures is None:
        xr_temperatures, _ = nonco2variation(config)
    if xr_co2_budgets is None:
        xr_co2_budgets = determine_global_budgets(config)
    if xr_traj_nonco2 is None:
        xr_traj_nonco2 = determine_global_nonco2_trajectories(config)

    # Shorthand for often-used expressions
    start_year = config.params.start_year_analysis

    dim_temp = config.dimension_ranges.peak_temperature
    dim_prob = config.dimension_ranges.risk_of_exceedance
    dim_nonco2 = config.dimension_ranges.non_co2_reduction
    dim_timing = config.dimension_ranges.timing_of_mitigation_action
    dim_negemis = config.dimension_ranges.negative_emissions

    # Initialize data arrays for co2
    startpoint = emissions.sel(Time=start_year, Region="EARTH").CO2_hist
    # compensation_form = np.array(list(np.linspace(0, 1, len(np.arange(start_year, 2101)))))#**1.1#+[1]*len(np.arange(2050, 2101)))

    hy = config.params.harmonization_year
    if start_year >= 2020:
        compensation_form = np.array(
            list(np.linspace(0, 1, len(np.arange(start_year, hy)))) + [1] * len(np.arange(hy, 2101))
        )
        xr_comp = xr.DataArray(
            compensation_form,
            dims=["Time"],
            coords={"Time": np.arange(start_year, 2101)},
        )
    if start_year < 2020:
        compensation_form = (np.arange(0, 2101 - start_year)) ** 0.5
        # hy = 2100
        # compensation_form = np.array(list(np.linspace(0, 1, len(np.arange(start_year, hy))))+[1]*len(np.arange(hy, 2101)))
        xr_comp = xr.DataArray(
            compensation_form / np.sum(compensation_form),
            dims=["Time"],
            coords={"Time": np.arange(start_year, 2101)},
        )

    def budget_harm(nz):
        return xr_comp / np.sum(xr_comp.sel(Time=np.arange(start_year, nz)))

    # compensation_form2 = np.array(list(np.linspace(0, 1, len(np.arange(start_year, 2101)))))**0.5#+[1]*len(np.arange(2050, 2101)))
    xr_traj_co2 = xr.Dataset(
        coords={
            "NegEmis": dim_negemis,
            "NonCO2red": dim_nonco2,
            "Temperature": dim_temp,
            "Risk": dim_prob,
            "Timing": dim_timing,
            "Time": np.arange(start_year, 2101),
        }
    )

    xr_traj_co2_neg = xr.Dataset(
        coords={
            "NegEmis": dim_negemis,
            "Temperature": dim_temp,
            "Time": np.arange(start_year, 2101),
        }
    )

    pathways_data = {
        "CO2_globe": xr.DataArray(
            data=np.nan,
            coords=xr_traj_co2.coords,
            dims=("NegEmis", "NonCO2red", "Temperature", "Risk", "Timing", "Time"),
            attrs={"description": "Pathway data"},
        ),
        "CO2_neg_globe": xr.DataArray(
            data=np.nan,
            coords=xr_traj_co2_neg.coords,
            dims=("NegEmis", "Temperature", "Time"),
            attrs={"description": "Pathway data"},
        ),
    }

    # CO2 emissions from AR6
    xr_scen2_use = emissions.xr_ar6.sel(Variable="Emissions|CO2")
    xr_scen2_use = xr_scen2_use.reindex(Time=np.arange(2000, 2101, 10))
    xr_scen2_use = xr_scen2_use.reindex(Time=np.arange(2000, 2101))
    xr_scen2_use = xr_scen2_use.interpolate_na(dim="Time", method="linear")
    xr_scen2_use = xr_scen2_use.reindex(Time=np.arange(start_year, 2101))

    co2_start = xr_scen2_use.sel(Time=start_year) / 1e3
    offsets = startpoint / 1e3 - co2_start
    emis_all = xr_scen2_use.sel(Time=np.arange(start_year, 2101)) / 1e3 + offsets * (1 - xr_comp)
    emis2100 = emis_all.sel(Time=2100)

    # Bend IAM curves to start in the correct starting year (only shape is relevant)
    difyears = 2020 + 1 - start_year
    if difyears > 0:
        emis_all_adapt = emis_all.assign_coords({"Time": emis_all.Time - (difyears - 1)}).reindex(
            {"Time": np.arange(start_year, 2101)}
        )
        for t in np.arange(0, difyears):
            dv = emis_all.sel(Time=2101 - difyears + t).Value - emis_all.Value.sel(
                Time=2101 - difyears + t - 1
            )
            dv = dv.where(dv < 0, 0)
            emis_all_adapt.Value.loc[{"Time": 2101 - difyears + t}] = dv + emis_all_adapt.Value.sel(
                Time=2101 - difyears + t - 1
            )

        fr = (
            (emis_all.Value.sum(dim="Time") - emis_all_adapt.Value.sum(dim="Time"))
            * (xr_comp)
            / np.sum(xr_comp)
        )
        emis_all = emis_all_adapt + fr

    # Negative emissions from AR6 (CCS + DAC)
    xr_neg = emissions.xr_ar6.sel(
        Variable=["Carbon Sequestration|CCS", "Carbon Sequestration|Direct Air Capture"]
    ).sum(dim="Variable", skipna=False)
    xr_neg = xr_neg.reindex(Time=np.arange(2000, 2101, 10))
    xr_neg = xr_neg.reindex(Time=np.arange(2000, 2101))
    xr_neg = xr_neg.interpolate_na(dim="Time", method="linear")
    xr_neg = xr_neg.reindex(Time=np.arange(start_year, 2101))

    def remove_upward(ar):
        # Small function to ensure no late-century increase in emissions due to sparse scenario spaces
        ar2 = np.copy(ar)
        ar2[29:] = np.minimum.accumulate(ar[29:])
        return ar2

    # Correction on temperature calibration when using IAM shapes starting at earlier years
    difyear = 2021 - start_year
    dt = difyear / 6 * 0.1

    def ms_temp_shape(
        temp, risk
    ):  # Different temperature domain because this is purely for the shape, not for the nonCO2 variation or so
        return xr_temperatures.ModelScenario[
            np.where(
                (xr_temperatures.Temperature.sel(Risk=risk) < dt + temp + 0.0)
                & (xr_temperatures.Temperature.sel(Risk=risk) > dt + temp - 0.3)
            )[0]
        ].values

    for temp_i, temp in enumerate(dim_temp):
        ms1 = ms_temp_shape(temp, 0.5)
        # Shape impacted by timing of action
        for timing_i, timing in enumerate(dim_timing):
            if timing == "Immediate" or temp in [1.5, 1.56, 1.6] and timing == "Delayed":
                mslist = scenarios["Immediate"]
            else:
                mslist = scenarios["Delayed"]
            ms2 = np.intersect1d(ms1, mslist)

            surplus_factor = calculate_surplus_factor(emissions, emis_all, emis2100, ms2)

            for neg_i, neg in enumerate(dim_negemis):
                xset = emis_all.sel(ModelScenario=ms2) - surplus_factor * (neg - 0.5)
                pathways_neg = xr_neg.sel(ModelScenario=ms1).quantile(neg, dim="ModelScenario")
                pathways_data["CO2_neg_globe"][neg_i, temp_i, :] = np.array(pathways_neg)
                for risk_i, risk in enumerate(dim_prob):
                    for nonco2_i, nonco2 in enumerate(dim_nonco2):
                        factor = (
                            xr_co2_budgets.Budget.sel(Temperature=temp, Risk=risk, NonCO2red=nonco2)
                            - xset.where(xset > 0).sum(dim="Time")
                        ) / np.sum(compensation_form)
                        all_pathways = (1e3 * (xset + factor * xr_comp)) / 1e3
                        if len(all_pathways) > 0:
                            pathway = all_pathways.mean(dim="ModelScenario")
                            pathway_sep = np.convolve(pathway, np.ones(3) / 3, mode="valid")
                            pathway[1:-1] = pathway_sep
                            offset = float(startpoint) / 1e3 - pathway[0]
                            pathway_final = np.array((pathway.T + offset) * 1e3)

                            # Remove upward emissions (harmonize later)
                            pathway_final = remove_upward(np.array(pathway_final))

                            # Harmonize by budget (iteration 3)
                            try:
                                nz = start_year + np.where(pathway_final <= 0)[0][0]
                            except:
                                nz = 2100
                            factor = (
                                xr_co2_budgets.Budget.sel(
                                    Temperature=temp, Risk=risk, NonCO2red=nonco2
                                )
                                * 1e3
                                - pathway_final[pathway_final > 0].sum()
                            )
                            pathway_final2 = (pathway_final + factor * budget_harm(nz)).values

                            try:
                                nz = start_year + np.where(pathway_final2 <= 0)[0][0]
                            except:
                                nz = 2100
                            factor = (
                                xr_co2_budgets.Budget.sel(
                                    Temperature=temp, Risk=risk, NonCO2red=nonco2
                                )
                                * 1e3
                                - pathway_final2[pathway_final2 > 0].sum()
                            )
                            pathway_final2 = (
                                1e3 * (pathway_final2 + factor * budget_harm(nz))
                            ) / 1e3

                            try:
                                nz = start_year + np.where(pathway_final2 <= 0)[0][0]
                            except:
                                nz = 2100
                            factor = (
                                xr_co2_budgets.Budget.sel(
                                    Temperature=temp, Risk=risk, NonCO2red=nonco2
                                )
                                * 1e3
                                - pathway_final2[pathway_final2 > 0].sum()
                            )
                            pathway_final2 = (
                                1e3 * (pathway_final2 + factor * budget_harm(nz))
                            ) / 1e3

                            pathways_data["CO2_globe"][
                                neg_i, nonco2_i, temp_i, risk_i, timing_i, :
                            ] = pathway_final2

    xr_traj_co2 = xr_traj_co2.update(pathways_data)
    xr_traj_ghg = (xr_traj_co2.CO2_globe + xr_traj_nonco2.NonCO2_globe).to_dataset(name="GHG_globe")

    # projected land use emissions
    landuse_ghg = emissions.mean(dim="ModelScenario").GHG_LULUCF
    landuse_co2 = emissions.mean(dim="ModelScenario").CO2_LULUCF

    # historical land use emissions
    landuse_ghg_hist = (
        emissions.sel(Region="EARTH").GHG_hist - emissions.sel(Region="EARTH").GHG_hist_excl
    )
    landuse_co2_hist = (
        emissions.sel(Region="EARTH").CO2_hist - emissions.sel(Region="EARTH").CO2_hist_excl
    )

    # Harmonize on startyear
    diff_ghg = -landuse_ghg.sel(Time=start_year) + landuse_ghg_hist.sel(Time=start_year)
    diff_co2 = -landuse_co2.sel(Time=start_year) + landuse_co2_hist.sel(Time=start_year)

    # Corrected
    landuse_ghg_corr = landuse_ghg + diff_ghg
    landuse_co2_corr = landuse_co2 + diff_co2

    xr_traj_ghg_excl = (xr_traj_ghg.GHG_globe - landuse_ghg_corr).to_dataset(name="GHG_globe_excl")
    xr_traj_co2_excl = (xr_traj_co2.CO2_globe - landuse_co2_corr).to_dataset(name="CO2_globe_excl")

    all_projected_gases = xr.merge(
        [
            xr_traj_ghg,
            xr_traj_co2.CO2_globe,
            xr_traj_co2.CO2_neg_globe,
            xr_traj_nonco2.NonCO2_globe,
            xr_traj_ghg_excl.GHG_globe_excl,
            xr_traj_co2_excl.CO2_globe_excl,
        ]
    )

    return all_projected_gases


def calculate_surplus_factor(emissions, emis_all, emis2100, ms2):
    emis2100_i = emis2100.sel(ModelScenario=ms2)

    # The 90-percentile of 2100 emissions
    ms_90 = emissions.xr_ar6.sel(ModelScenario=ms2).ModelScenario[
        (emis2100_i >= emis2100_i.quantile(0.9 - 0.1))
        & (emis2100_i <= emis2100_i.quantile(0.9 + 0.1))
    ]

    # The 10-percentile of 2100 emissions
    ms_10 = emissions.xr_ar6.sel(ModelScenario=ms2).ModelScenario[
        (emis2100_i >= emis2100_i.quantile(0.1 - 0.1))
        & (emis2100_i <= emis2100_i.quantile(0.1 + 0.1))
    ]

    # Difference and smoothen this
    surplus_factor = emis_all.sel(ModelScenario=np.intersect1d(ms_90, ms2)).mean(
        dim="ModelScenario"
    ) - emis_all.sel(ModelScenario=np.intersect1d(ms_10, ms2)).mean(dim="ModelScenario")
    surplus_factor2 = np.convolve(surplus_factor, np.ones(3) / 3, mode="valid")
    surplus_factor[1:-1] = surplus_factor2
    return surplus_factor

    # Merge all data into a single xrarray object
