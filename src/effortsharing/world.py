import logging

import numpy as np
import pandas as pd
import xarray as xr

from effortsharing.config import Config

logger = logging.getLogger(__name__)


def nonco2variation(config: Config):
    # NOTE: I moved this file from lambollrepo to our surfdrive folder
    data_root = config.paths.input
    filename = "job-20211019-ar6-nonco2_Raw-GSAT-Non-CO2.csv"

    df = pd.read_csv(data_root / filename)
    df = df[
        [
            "model",
            "scenario",
            "Category",
            "variable",
            "permafrost",
            "median peak warming (MAGICCv7.5.3)",
            "p33 peak warming (MAGICCv7.5.3)",
            "p67 peak warming (MAGICCv7.5.3)",
            "median year of peak warming (MAGICCv7.5.3)",
            "p33 year of peak warming (MAGICCv7.5.3)",
            "p67 year of peak warming (MAGICCv7.5.3)",
        ]
        + list(df.keys()[28:])
    ]

    df.columns = df.columns.str.replace(r"(\d{4})-01-01 00:00:00", r"\1", regex=True)
    df.rename(
        columns={
            "variable": "NonCO2WarmingQuantile",
            "permafrost": "Permafrost",
            "median peak warming (MAGICCv7.5.3)": "T(0.5)",
            "p33 peak warming (MAGICCv7.5.3)": "T(0.33)",
            "p67 peak warming (MAGICCv7.5.3)": "T(0.67)",
            "median year of peak warming (MAGICCv7.5.3)": "Y(0.50)",
            "p33 year of peak warming (MAGICCv7.5.3)": "Y(0.33)",
            "p67 year of peak warming (MAGICCv7.5.3)": "Y(0.67)",
        },
        inplace=True,
    )

    # ModelScenario
    modscen = []
    df["ModelScenario"] = df["model"] + "|" + df["scenario"]
    df = df.drop(columns=["model", "scenario"])

    # Rename warming quantiles
    quantiles_map = {
        f"AR6 climate diagnostics|Raw Surface Temperature (GSAT)|Non-CO2|MAGICCv7.5.3|{i}th Percentile": float(
            i
        )
        / 100
        for i in ["10.0", "16.7", "33.0", "5.0", "50.0", "67.0", "83.3", "90.0", "95.0"]
    }
    df["NonCO2WarmingQuantile"] = (
        df["NonCO2WarmingQuantile"].replace(quantiles_map).astype(float).round(2)
    )

    # Only consider excluding permafrost
    df = df[df.Permafrost == False]
    df = df.drop(columns=["Permafrost"])

    # Xarray for time-varying data
    df_dummy = df[
        ["ModelScenario", "NonCO2WarmingQuantile"] + list(np.arange(1995, 2101).astype(str))
    ].melt(
        id_vars=["ModelScenario", "NonCO2WarmingQuantile"],
        var_name="Time",
        value_name="NonCO2warming",
    )
    df_dummy["Time"] = df_dummy["Time"].astype(int)
    df_dummy = df_dummy.set_index(["ModelScenario", "NonCO2WarmingQuantile", "Time"])
    xr_lamboll = xr.Dataset.from_dataframe(df_dummy)

    # Xarray for peak warming years
    df_peakyears = df[["ModelScenario", "NonCO2WarmingQuantile", "Y(0.50)", "Y(0.33)", "Y(0.67)"]]
    df_peakyears = df_peakyears.rename(columns={"Y(0.50)": 0.5, "Y(0.33)": 0.33, "Y(0.67)": 0.67})
    df_peakyears = df_peakyears.melt(
        id_vars=["ModelScenario", "NonCO2WarmingQuantile"],
        var_name="TCRE",
        value_name="PeakYear",
    )
    df_dummy = df_peakyears.set_index(["ModelScenario", "NonCO2WarmingQuantile", "TCRE"])
    xr_peakyears = xr.Dataset.from_dataframe(df_dummy)

    # Xarray for full peak warming
    # Also extrapolate for 17 and 83 percentiles (based on normal distribution assumption)
    df_peaktemps = df[["ModelScenario", "T(0.5)", "T(0.33)", "T(0.67)"]].drop_duplicates()
    df_peaktemps = df_peaktemps.rename(columns={"T(0.5)": 0.5, "T(0.33)": 0.33, "T(0.67)": 0.67})
    df_peaktemps = df_peaktemps.melt(
        id_vars=["ModelScenario"], var_name="TCRE", value_name="Temperature"
    )
    df_dummy = df_peaktemps.set_index(["ModelScenario", "TCRE"])
    xr_temperatures = xr.Dataset.from_dataframe(df_dummy)
    xr_temperatures17 = (
        (
            xr_temperatures.sel(TCRE=0.33)
            - 1 * (xr_temperatures.sel(TCRE=0.67) - xr_temperatures.sel(TCRE=0.33))
        )
        .drop_vars("TCRE")
        .expand_dims({"TCRE": [0.17]})
    )
    xr_temperatures83 = (
        (
            xr_temperatures.sel(TCRE=0.67)
            + 1 * (xr_temperatures.sel(TCRE=0.67) - xr_temperatures.sel(TCRE=0.33))
        )
        .drop_vars("TCRE")
        .expand_dims({"TCRE": [0.83]})
    )
    xr_temperatures = xr.merge([xr_temperatures, xr_temperatures17, xr_temperatures83])

    # Peak warming -> at the peak year.
    xr_peaknonco2warming_all = xr_lamboll.sel(Time=xr_peakyears.PeakYear).rename(
        {"NonCO2warming": "PeakWarming"}
    )

    # Now we assume that nonco2 warming quantiles are the same as the peak warming quantiles
    # That is: climate sensitivity for the full picture (TCRE) is directly related to climate sensitivity to only non-CO2
    # Also extrapolate for 17 and 83 percentiles (based on normal distribution assumption)
    # relation nonco2 peak warming to TCRE is not trivial, because the peakyears are also dependent on TCRE!
    # However, as it turns out, higher TCRE implies in practically all cases a higher nonCO2 warming at the peak year
    xr_peaknonco2warming_all = xr_peaknonco2warming_all.drop_vars("Time")
    peak50 = xr_peaknonco2warming_all.sel(NonCO2WarmingQuantile=0.5, TCRE=[0.5]).drop_vars(
        "NonCO2WarmingQuantile"
    )
    peak33 = xr_peaknonco2warming_all.sel(NonCO2WarmingQuantile=0.33, TCRE=[0.33]).drop_vars(
        "NonCO2WarmingQuantile"
    )
    peak67 = xr_peaknonco2warming_all.sel(NonCO2WarmingQuantile=0.67, TCRE=[0.67]).drop_vars(
        "NonCO2WarmingQuantile"
    )
    peak17 = (
        (
            xr_peaknonco2warming_all.sel(NonCO2WarmingQuantile=0.33, TCRE=0.33)
            - 1
            * (
                xr_peaknonco2warming_all.sel(NonCO2WarmingQuantile=0.67, TCRE=0.67)
                - xr_peaknonco2warming_all.sel(NonCO2WarmingQuantile=0.33, TCRE=0.33)
            )
        )
        .drop_vars("NonCO2WarmingQuantile")
        .drop_vars("TCRE")
        .expand_dims({"TCRE": [0.17]})
    )
    peak83 = (
        (
            xr_peaknonco2warming_all.sel(NonCO2WarmingQuantile=0.67, TCRE=0.67)
            + 1
            * (
                xr_peaknonco2warming_all.sel(NonCO2WarmingQuantile=0.67, TCRE=0.67)
                - xr_peaknonco2warming_all.sel(NonCO2WarmingQuantile=0.33, TCRE=0.33)
            )
        )
        .drop_vars("NonCO2WarmingQuantile")
        .drop_vars("TCRE")
        .expand_dims({"TCRE": [0.83]})
    )
    xr_peaknonco2warming = xr.merge([peak50, peak33, peak67, peak17, peak83])

    # Invert axis for Risk coordinate
    xr_peaknonco2warming = xr_peaknonco2warming.assign_coords(
        TCRE=[0.83, 0.67, 0.5, 0.33, 0.17]
    ).rename({"TCRE": "Risk"})
    xr_temperatures = xr_temperatures.assign_coords(TCRE=[0.83, 0.67, 0.5, 0.33, 0.17]).rename(
        {"TCRE": "Risk"}
    )

    # Save for later use
    xr_temperatures = xr_temperatures
    xr_nonco2warmings = xr_peaknonco2warming
    xr_nonco2warming_wrt_start = (
        xr_peaknonco2warming
        - xr_lamboll.rename({"NonCO2WarmingQuantile": "Risk"})
        .sel(Time=config.params.start_year_analysis)
        .NonCO2warming
    )

    # TODO: if nonco2warmings is not needed, we could merge the two others:
    # return = xr.merge([xr_temperatures, xr_nonco2warming_wrt_start])

    return (
        xr_temperatures,
        # xr_nonco2warmings,  # TODO: not used, remove?
        xr_nonco2warming_wrt_start,
    )


def determine_global_nonco2_trajectories(config: Config, emissions, scenarios, temperatures):
    logger.info("Computing global nonco2 trajectories")

    # Shorthand for often-used expressions
    start_year = config.params.start_year_analysis
    n_years = 2101 - start_year

    # TODO: this can probably do without the rounding or casting to array
    dim_temp = np.array(config.dimension_ranges.peak_temperature).astype(float).round(2)
    dim_prob = np.array(config.dimension_ranges.risk_of_exceedance).round(2)
    dim_nonco2 = np.array(config.dimension_ranges.non_co2_reduction).round(2)
    dim_timing = np.array(config.dimension_ranges.timing_of_mitigation_action)

    # Relationship between non-co2 reduction and budget is based on Rogelj et al
    # and requires the year 2020 (even though startyear may be different) - not
    # a problem
    xr_ch4_raw = emissions.xr_ar6.sel(Variable="Emissions|CH4") * config.params.gwp_ch4
    xr_n2o_raw = emissions.xr_ar6.sel(Variable="Emissions|N2O") * config.params.gwp_n2o / 1e3
    n2o_start = emissions.sel(Region="EARTH").sel(Time=start_year).N2O_hist
    ch4_start = emissions.sel(Region="EARTH").sel(Time=start_year).CH4_hist
    n2o_2020 = emissions.sel(Region="EARTH").sel(Time=2020).N2O_hist
    ch4_2020 = emissions.sel(Region="EARTH").sel(Time=2020).CH4_hist
    tot_2020 = n2o_2020 + ch4_2020
    tot_start = n2o_start + ch4_start

    # Rescale CH4 and N2O trajectories
    n_years_before = config.params.harmonization_year - start_year
    n_years_after = 2101 - config.params.harmonization_year
    compensation_form = np.array(list(np.linspace(0, 1, n_years_before)) + [1] * n_years_after)
    xr_comp = xr.DataArray(
        1 - compensation_form,
        dims=["Time"],
        coords={"Time": np.arange(start_year, 2101)},
    )
    xr_nonco2_raw = xr_ch4_raw + xr_n2o_raw
    xr_nonco2_raw_start = xr_nonco2_raw.sel(Time=start_year)
    xr_nonco2_raw = xr_nonco2_raw.sel(Time=np.arange(start_year, 2101))

    def ms_temp(temp, risk):
        return temperatures.ModelScenario[
            np.where(np.abs(temp - temperatures.Temperature.sel(Risk=risk)) < 0.2)[0]
        ].values

    def check_monotomy(traj):
        vec = [traj[0]]
        for i in range(1, len(traj)):
            if traj[i] <= vec[i - 1]:
                vec.append(traj[i])
            else:
                vec.append(vec[i - 1])
        return np.array(vec)

    def rescale(traj):
        offset = traj.sel(Time=start_year) - tot_start
        traj_scaled = -xr_comp * offset + traj
        return traj_scaled

    xr_reductions = (xr_nonco2_raw.sel(Time=2040) - xr_nonco2_raw_start) / xr_nonco2_raw_start

    temps = []
    risks = []
    times = []
    nonco2 = []
    vals = []
    timings = []

    for temp_i, temp in enumerate(dim_temp):
        for p_i, p in enumerate(dim_prob):
            ms1 = ms_temp(temp, p)
            for timing_i, timing in enumerate(dim_timing):
                if timing == "Immediate" or temp in [1.5, 1.56, 1.6] and timing == "Delayed":
                    mslist = scenarios["Immediate"]
                else:
                    mslist = scenarios["Delayed"]
                ms2 = np.intersect1d(ms1, mslist)
                if len(ms2) == 0:
                    for n_i, n in enumerate(dim_nonco2):
                        times = times + list(np.arange(start_year, 2101))
                        vals = vals + [np.nan] * n_years
                        nonco2 = nonco2 + [n] * n_years
                        temps = temps + [temp] * n_years
                        risks = risks + [p] * n_years
                        timings = timings + [timing] * n_years
                else:
                    reductions = xr_reductions.sel(
                        ModelScenario=np.intersect1d(xr_reductions.ModelScenario, ms2)
                    )
                    # TODO: note that reductions may have length 1
                    reds = reductions.quantile(dim_nonco2[::-1])
                    for n_i, n in enumerate(dim_nonco2):
                        red = reds[n_i]
                        ms2 = reductions.ModelScenario[np.where(np.abs(reductions - red) < 0.1)]
                        trajs = xr_nonco2_raw.sel(
                            ModelScenario=ms2,
                            Time=np.arange(start_year, 2101),
                        )
                        trajectory_mean = rescale(trajs.mean(dim="ModelScenario"))

                        # Harmonize reduction
                        red_traj = (trajectory_mean.sel(Time=2040) - tot_2020) / tot_2020
                        traj2 = (
                            -(1 - xr_comp) * (red_traj - red) * xr_nonco2_raw_start.mean()
                            + trajectory_mean
                        )  # 1.5*red has been removed -> check effect
                        trajectory_mean2 = check_monotomy(np.array(traj2))
                        times = times + list(np.arange(start_year, 2101))
                        vals = vals + list(trajectory_mean2)
                        nonco2 = nonco2 + [n] * n_years
                        temps = temps + [temp] * n_years
                        risks = risks + [p] * n_years
                        timings = timings + [timing] * n_years

    dict_nonco2 = {}
    dict_nonco2["Time"] = times
    dict_nonco2["NonCO2red"] = nonco2
    dict_nonco2["NonCO2_globe"] = vals
    dict_nonco2["Temperature"] = temps
    dict_nonco2["Risk"] = risks
    dict_nonco2["Timing"] = timings
    df_nonco2 = pd.DataFrame(dict_nonco2)
    dummy = df_nonco2.set_index(["NonCO2red", "Time", "Temperature", "Risk", "Timing"])
    xr_traj_nonco2 = xr.Dataset.from_dataframe(dummy)

    # Post-processing: making temperature dependence smooth
    xr_traj_nonco2 = xr_traj_nonco2.reindex({"Temperature": [1.5, 1.8, 2.1, 2.4]})
    xr_traj_nonco2 = xr_traj_nonco2.reindex({"Temperature": dim_temp})
    xr_traj_nonco2 = xr_traj_nonco2.interpolate_na(dim="Temperature")
    xr_traj_nonco2_2 = xr_traj_nonco2.copy()

    # change time coordinate in self.xr_traj_nonco2 if needed (different starting year than 2021)
    difyears = 2020 + 1 - start_year

    if difyears > 0:
        xr_traj_nonco2_adapt = xr_traj_nonco2.assign_coords(
            {"Time": xr_traj_nonco2.Time - (difyears - 1)}
        ).reindex({"Time": np.arange(start_year, 2101)})
        for t in np.arange(0, difyears):
            xr_traj_nonco2_adapt.NonCO2_globe.loc[{"Time": 2101 - difyears + t}] = (
                xr_traj_nonco2.sel(Time=2101 - difyears + t).NonCO2_globe
                - xr_traj_nonco2.NonCO2_globe.sel(Time=2101 - difyears + t - 1)
            ) + xr_traj_nonco2_adapt.NonCO2_globe.sel(Time=2101 - difyears + t - 1)
        fr = (
            (
                xr_traj_nonco2.NonCO2_globe.sum(dim="Time")
                - xr_traj_nonco2_adapt.NonCO2_globe.sum(dim="Time")
            )
            * (1 - xr_comp)
            / np.sum(1 - xr_comp)
        )
        xr_traj_nonco2 = xr_traj_nonco2_adapt + fr
    else:
        xr_traj_nonco2_adapt = None

    return (
        xr_traj_nonco2,
        # xr_traj_nonco2_2,  # TODO: not used, remove?
        # xr_traj_nonco2_adapt,  # TODO: not used, remove?
    )


def determine_global_budgets(config: Config, emissions, temperatures, xr_nonco2warming_wrt_start):
    logger.info("Get global CO2 budgets")

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

    return (
        xr_bud_co2,  # TODO: not used, remove? Fine
        xr_co2_budgets,
    )


def determine_global_co2_trajectories(
    config: Config,
    emissions,
    scenarios,
    xr_temperatures,
    xr_co2_budgets,
    xr_traj_nonco2,
):
    logger.info("Computing global co2 trajectories")

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

    return (
        # xr_traj_co2,  # TODO: not used. Remove?
        # xr_traj_ghg,  # TODO: not used. Remove?
        # landuse_ghg_corr,  # TODO: not used. Remove?
        # landuse_co2_corr,  # TODO: not used. Remove?
        # xr_traj_ghg_excl,  # TODO: not used. Remove?
        # xr_traj_co2_excl,  # TODO: not used. Remove?
        all_projected_gases,
    )


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


def merge_data(
    xr_co2_budgets,
    all_projected_gases,
    emission_data,
    ndc_data,
    socioeconomic_data,
):
    return xr.merge(
        [
            xr_co2_budgets["Budget"],
            all_projected_gases[  # TODO: could merge whole dataarray at once, no need to list all vars explicitly. Did this to get overview of what variable comes from where.
                [
                    "GHG_globe",
                    "CO2_globe",
                    "CO2_neg_globe",
                    "NonCO2_globe",
                    "GHG_globe_excl",
                    "CO2_globe_excl",
                ]
            ],
            emission_data[  # TODO: already stored elsewhere. Remove?
                [
                    "GHG_hist",
                    "GHG_hist_excl",
                    "CO2_hist",
                    "CO2_hist_excl",
                    "CH4_hist",
                    "N2O_hist",
                    "CO2_base_excl",
                    "CO2_base_incl",
                    "GHG_base_excl",
                    "GHG_base_incl",
                    "GHG_excl_C",
                    "CO2_excl_C",
                    "CO2_neg_C",
                    "CO2_bunkers_C",
                ]
            ],
            ndc_data[  # TODO: already stored elsewhere. Remove?
                [
                    "GHG_ndc",
                    "GHG_ndc_red",
                    "GHG_ndc_inv",
                    "GHG_ndc_excl_red",
                    "GHG_ndc_excl",
                    "GHG_ndc_excl_inv",
                    "GHG_ndc_excl_CR",
                ]
            ],
            socioeconomic_data[  # TODO: already stored elsewhere. Remove?
                [
                    "GDP",
                    "HDIsh",
                    "Population",
                ]
            ],
        ]
    )


def add_country_groups(config: Config, regions, xr_total):
    logger.info("Add country groups")

    data_root = config.paths.input
    filename = "UNFCCC_Parties_Groups_noeu.xlsx"
    regions_name = list(regions.keys())
    regions_iso = list(regions.values())

    df = pd.read_excel(data_root / filename, sheet_name="Country groups")
    countries_iso = np.array(df["Country ISO Code"])
    list_of_regions = list(np.array(regions_iso).copy())
    reg_iso = regions_iso.copy()
    reg_name = regions_name.copy()
    new_total = xr_total.copy()
    for group_of_choice in [
        "G20",
        "EU",
        "G7",
        "SIDS",
        "LDC",
        "Northern America",
        "Australasia",
        "African Group",
        "Umbrella",
    ]:
        if group_of_choice != "EU":
            list_of_regions = list_of_regions + [group_of_choice]
        group_indices = countries_iso[np.array(df[group_of_choice]) == 1]
        country_to_eu = {}
        for cty in np.array(new_total.Region):
            if cty in group_indices:
                country_to_eu[cty] = [group_of_choice]
            else:
                country_to_eu[cty] = [""]
        group_coord = xr.DataArray(
            [
                group
                for country in np.array(new_total["Region"])
                for group in country_to_eu[country]
            ],
            dims=["Region"],
            coords={
                "Region": [
                    country
                    for country in np.array(new_total["Region"])
                    for group in country_to_eu[country]
                ]
            },
        )
        if group_of_choice == "EU":
            xr_eu = (
                new_total[
                    [
                        "Population",
                        "GDP",
                        "GHG_hist",
                        "GHG_base_incl",
                        "CO2_hist",
                        "CO2_base_incl",
                        "GHG_hist_excl",
                        "GHG_base_excl",
                        "CO2_hist_excl",
                        "CO2_base_excl",
                    ]
                ]
                .groupby(group_coord)
                .sum()
            )  # skipna=False)
        else:
            xr_eu = (
                new_total[
                    [
                        "Population",
                        "GDP",
                        "GHG_hist",
                        "GHG_base_incl",
                        "CO2_hist",
                        "CO2_base_incl",
                        "GHG_hist_excl",
                        "GHG_base_excl",
                        "CO2_hist_excl",
                        "CO2_base_excl",
                        "GHG_ndc",
                        "GHG_ndc_inv",
                        "GHG_ndc_excl",
                        "GHG_ndc_excl_inv",
                        "GHG_ndc_excl_CR",
                    ]
                ]
                .groupby(group_coord)
                .sum(skipna=False)
            )
        xr_eu2 = xr_eu.rename({"group": "Region"})
        dummy = new_total.reindex(Region=list_of_regions)

        new_total = xr.merge([dummy, xr_eu2])
        new_total = new_total.reindex(Region=list_of_regions)
        if group_of_choice not in ["EU", "EARTH"]:
            reg_iso.append(group_of_choice)
            reg_name.append(group_of_choice)

    new_total = new_total
    new_total["GHG_base_incl"][np.where(new_total.Region == "EU")[0], np.array([3, 4])] = (
        np.nan
    )  # SSP4, 5 are empty for Europe!
    new_total["CO2_base_incl"][np.where(new_total.Region == "EU")[0], np.array([3, 4])] = (
        np.nan
    )  # SSP4, 5 are empty for Europe!
    new_total["GHG_base_excl"][np.where(new_total.Region == "EU")[0], np.array([3, 4])] = (
        np.nan
    )  # SSP4, 5 are empty for Europe!
    new_total["CO2_base_excl"][np.where(new_total.Region == "EU")[0], np.array([3, 4])] = (
        np.nan
    )  # SSP4, 5 are empty for Europe!

    new_regions = dict(zip(reg_name, reg_iso))

    return new_total, new_regions


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
    savepath = config.paths.output / f"startyear_{startyear}" / "xr_dataread.nc"

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
            rbw.to_netcdf(savepath / f"xr_rbw_{gas}_lulucf.nc")


# TODO: this doesn't need xr_version, only regions. Separate it more clearly.
def save_rci(config: Config, xr_version):
    """Save RCI data to netcdf file."""

    savepath = config.paths.output / "xr_rci.nc"
    logger.info(f"Saving RCI data to {savepath}")

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
    xr_rci = xr.Dataset.from_dataframe(dfdummy)
    xr_rci = xr_rci.reindex({"Region": xr_version.Region})
    xr_rci.to_netcdf(savepath)


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


def main(config: Config):

    import effortsharing as es

    countries, regions = es.input.socioeconomics.read_general(config)

    # Read input data
    socioeconomic_data = es.input.socioeconomics.load_socioeconomics(config)
    modelscenarios = es.input.emissions.read_modelscenarios(config)
    emission_data = es.input.emissions.load_emissions(config)
    primap_data = es.input.emissions.read_primap(config)
    ndc_data = es.input.ndcs.load_ndcs(config, emission_data)

    # Calculate global budgets and pathways
    xr_temperatures, xr_nonco2warming_wrt_start = nonco2variation(config)
    (xr_traj_nonco2,) = determine_global_nonco2_trajectories(
        config, emission_data, modelscenarios, xr_temperatures
    )
    _, xr_co2_budgets = determine_global_budgets(
        config, emission_data, xr_temperatures, xr_nonco2warming_wrt_start
    )
    (all_projected_gases,) = determine_global_co2_trajectories(
        config,
        emissions=emission_data,
        scenarios=modelscenarios,
        xr_temperatures=xr_temperatures,
        xr_co2_budgets=xr_co2_budgets,
        xr_traj_nonco2=xr_traj_nonco2,
    )

    # Merge all data into a single xrarray object
    xr_total = (
        merge_data(
            xr_co2_budgets,
            all_projected_gases,
            emission_data,  # TODO: already stored elsewhere. Skip?
            ndc_data,  # TODO: already stored elsewhere. Skip?
            socioeconomic_data,  # TODO: already stored elsewhere. Skip?
        )
        .reindex(Region=list(regions.values()))
        .reindex(Time=np.arange(1850, 2101))
        .interpolate_na(dim="Time", method="linear")
    )

    # Add country groups
    new_total, new_regions = add_country_groups(config, regions, xr_total)

    # Save the data
    save_temp = np.array(config.dimension_ranges.peak_temperature_saved).astype(float).round(2)
    xr_version = new_total.sel(Temperature=save_temp)
    save_regions(config, new_regions, countries)
    save_total(config, xr_version)
    save_rbw(config, xr_version, countries)
    save_rci(config, xr_version)

    # Country-specific data readers
    datareader_netherlands(config, new_total)
    datareader_norway(config, new_total, primap_data)


if __name__ == "__main__":
    import argparse

    from rich.logging import RichHandler

    # Set up logging
    logging.basicConfig(level="INFO", format="%(message)s", handlers=[RichHandler(show_time=False)])

    # Get the config file from command line arguments
    parser = argparse.ArgumentParser(description="Process all input data")
    parser.add_argument("config", help="Path to config file")
    args = parser.parse_args()

    config = Config.from_file(args.config)
    main(config)
