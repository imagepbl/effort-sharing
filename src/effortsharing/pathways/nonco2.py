import logging

import numpy as np
import pandas as pd
import xarray as xr

from effortsharing.config import Config
from effortsharing.input.emissions import load_emissions, read_modelscenarios

logger = logging.getLogger(__name__)


def nonco2variation(config: Config):
    df = read_nonco2_lamboll(config)

    # Xarray for time-varying data
    xr_lamboll = extract_time_varying_data(df)

    # Xarray for peak warming years
    xr_peakyears = extract_peakyears(df)

    # Xarray for full peak warming
    xr_temperatures = extract_temperatures(df)

    xr_peaknonco2warming = extract_peak_warming_quantiles(xr_lamboll, xr_peakyears)

    # Save for later use
    xr_nonco2warming_wrt_start = (
        xr_peaknonco2warming
        - xr_lamboll.rename({"NonCO2WarmingQuantile": "Risk"})
        .sel(Time=config.params.start_year_analysis)
        .NonCO2warming
    )

    return (
        xr_temperatures,
        xr_nonco2warming_wrt_start,
    )


def extract_peak_warming_quantiles(xr_lamboll, xr_peakyears):
    # Now we assume that nonco2 warming quantiles are the same as the peak warming quantiles
    # That is: climate sensitivity for the full picture (TCRE) is directly related to climate sensitivity to only non-CO2
    # Also extrapolate for 17 and 83 percentiles (based on normal distribution assumption)
    # relation nonco2 peak warming to TCRE is not trivial, because the peakyears are also dependent on TCRE!
    # However, as it turns out, higher TCRE implies in practically all cases a higher nonCO2 warming at the peak year

    # Peak warming -> at the peak year.
    xr_peaknonco2warming_all = xr_lamboll.sel(Time=xr_peakyears.PeakYear).rename(
        {"NonCO2warming": "PeakWarming"}
    )

    all = xr_peaknonco2warming_all.drop_vars("Time")
    peak50 = all.sel(NonCO2WarmingQuantile=0.5, TCRE=[0.5]).drop_vars("NonCO2WarmingQuantile")
    peak33 = all.sel(NonCO2WarmingQuantile=0.33, TCRE=[0.33]).drop_vars("NonCO2WarmingQuantile")
    peak67 = all.sel(NonCO2WarmingQuantile=0.67, TCRE=[0.67]).drop_vars("NonCO2WarmingQuantile")
    diff = all.sel(NonCO2WarmingQuantile=0.67, TCRE=0.67) - all.sel(
        NonCO2WarmingQuantile=0.33, TCRE=0.33
    )
    peak17 = (
        (all.sel(NonCO2WarmingQuantile=0.33, TCRE=0.33) - diff)
        .drop_vars("NonCO2WarmingQuantile")
        .drop_vars("TCRE")
        .expand_dims({"TCRE": [0.17]})
    )

    peak83 = (
        (all.sel(NonCO2WarmingQuantile=0.67, TCRE=0.67) + diff)
        .drop_vars("NonCO2WarmingQuantile")
        .drop_vars("TCRE")
        .expand_dims({"TCRE": [0.83]})
    )
    xr_peaknonco2warming = xr.merge([peak50, peak33, peak67, peak17, peak83])

    # Invert axis for Risk coordinate
    xr_peaknonco2warming = xr_peaknonco2warming.assign_coords(
        TCRE=[0.83, 0.67, 0.5, 0.33, 0.17]
    ).rename({"TCRE": "Risk"})

    return xr_peaknonco2warming


def extract_temperatures(df):
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

    # Invert axis for Risk coordinate
    xr_temperatures = xr_temperatures.assign_coords(TCRE=[0.83, 0.67, 0.5, 0.33, 0.17]).rename(
        {"TCRE": "Risk"}
    )
    return xr_temperatures


def extract_peakyears(df):
    df_peakyears = df[["ModelScenario", "NonCO2WarmingQuantile", "Y(0.50)", "Y(0.33)", "Y(0.67)"]]
    df_peakyears = df_peakyears.rename(columns={"Y(0.50)": 0.5, "Y(0.33)": 0.33, "Y(0.67)": 0.67})
    df_peakyears = df_peakyears.melt(
        id_vars=["ModelScenario", "NonCO2WarmingQuantile"],
        var_name="TCRE",
        value_name="PeakYear",
    )
    df_dummy = df_peakyears.set_index(["ModelScenario", "NonCO2WarmingQuantile", "TCRE"])
    xr_peakyears = xr.Dataset.from_dataframe(df_dummy)
    return xr_peakyears


def extract_time_varying_data(df):
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
    return xr_lamboll


def read_nonco2_lamboll(config):
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
        df["NonCO2WarmingQuantile"].map(quantiles_map).astype(float).round(2)
    )

    # Only consider excluding permafrost
    df = df[df.Permafrost == False]
    df = df.drop(columns=["Permafrost"])
    return df


def determine_global_nonco2_trajectories(
    config: Config, emissions=None, scenarios=None, temperatures=None
):
    logger.info("Computing global nonco2 trajectories")

    if emissions is None:
        emissions = load_emissions(config)

    if scenarios is None:
        scenarios = read_modelscenarios(config)

    if temperatures is None:
        temperatures, _ = nonco2variation(config)

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

    return xr_traj_nonco2
