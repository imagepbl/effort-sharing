from dataclasses import dataclass
import json
import logging

import numpy as np
import pandas as pd
import xarray as xr

import effortsharing.regions as _regions
from effortsharing.config import Config

logger = logging.getLogger(__name__)


def read_historicalemis_jones(config: Config, regions):
    # No harmonization with the KEV anymore, but it's also much closer now
    logger.info("Reading historical emissions (jones)")

    # Define input
    # TODO: separate functions for reading each of these files?
    data_root = config.paths.input
    guetschow_et_al = "Guetschow_et_al_2024-PRIMAP-hist_v2.5.1_final_no_rounding_27-Feb-2024.nc"
    emissions_file = "EMISSIONS_ANNUAL_1830-2022.csv"
    edgar_file = "EDGARv8.0_FT2022_GHG_booklet_2023.xlsx"
    country_groups_file = "UNFCCC_Parties_Groups_noeu.xlsx"

    # Read data
    xr_primap2 = xr.open_dataset(data_root / guetschow_et_al)
    df_nwc = pd.read_csv(data_root / emissions_file)
    xr_nwc = xr.Dataset.from_dataframe(
        df_nwc.drop(columns=["CNTR_NAME", "Unit"]).set_index(["ISO3", "Gas", "Component", "Year"])
    )

    # Rename GLOBAL to EARTH
    regs = np.array(xr_nwc.ISO3)
    regs[regs == "GLOBAL"] = "EARTH"
    xr_nwc["ISO3"] = regs

    # Calculate total(?)
    xr_nwc_tot = (
        xr_nwc.sel(Gas="CH[4]") * config.params.gwp_ch4 / 1e3
        + xr_nwc.sel(Gas="N[2]*O") * config.params.gwp_n2o / 1e3
        + xr_nwc.sel(Gas="CO[2]") * 1
    ).drop_vars(["Gas"])

    # Calculate individual contributions
    # TODO: adding these will yield same total as above and seems cleaner
    xr_nwc_co2 = xr_nwc.sel(Gas="CO[2]").drop_vars(["Gas"])
    xr_nwc_ch4 = xr_nwc.sel(Gas="CH[4]").drop_vars(["Gas"]) * config.params.gwp_ch4 / 1e3
    xr_nwc_n2o = xr_nwc.sel(Gas="N[2]*O").drop_vars(["Gas"]) * config.params.gwp_n2o / 1e3

    # Select relevant data from primap (?)
    xr_primap_agri = (
        xr_primap2["KYOTOGHG (AR6GWP100)"]
        .rename(
            {
                "area (ISO3)": "Region",
                "scenario (PRIMAP-hist)": "scen",
                "category (IPCC2006_PRIMAP)": "cat",
            }
        )
        .sel(
            scen="HISTTP",
            provenance="derived",
            cat=["M.AG"],
            source="PRIMAP-hist_v2.5.1_final_nr",
        )
        .sum(dim="cat")
        .drop_vars(["source", "provenance", "scen"])
    )
    xr_primap_agri["time"] = np.arange(1750, 2023)
    xr_primap_agri = xr_primap_agri.rename({"time": "Time"})

    # Same for CO2 from primap (?)
    xr_primap_agri_co2 = (
        xr_primap2["CO2"]
        .rename(
            {
                "area (ISO3)": "Region",
                "scenario (PRIMAP-hist)": "scen",
                "category (IPCC2006_PRIMAP)": "cat",
            }
        )
        .sel(
            scen="HISTTP",
            provenance="derived",
            cat=["M.AG"],
            source="PRIMAP-hist_v2.5.1_final_nr",
        )
        .sum(dim="cat")
        .drop_vars(["source", "provenance", "scen"])
    )
    xr_primap_agri_co2["time"] = np.arange(1750, 2023)
    xr_primap_agri_co2 = xr_primap_agri_co2.rename({"time": "Time"})

    # Select historical data from NWC
    xr_ghghist = (
        xr_nwc_tot.rename({"ISO3": "Region", "Year": "Time", "Data": "GHG_hist"})
        .sel(Component="Total")
        .drop_vars("Component")
    )
    xr_co2hist = (
        xr_nwc_co2.rename({"ISO3": "Region", "Year": "Time", "Data": "CO2_hist"})
        .sel(Component="Total")
        .drop_vars("Component")
    )
    xr_ch4hist = (
        xr_nwc_ch4.rename({"ISO3": "Region", "Year": "Time", "Data": "CH4_hist"})
        .sel(Component="Total")
        .drop_vars("Component")
    )
    xr_n2ohist = (
        xr_nwc_n2o.rename({"ISO3": "Region", "Year": "Time", "Data": "N2O_hist"})
        .sel(Component="Total")
        .drop_vars("Component")
    )

    # Calculate emissions excluding LULUCF
    xr_ghgexcl = (
        xr_nwc_tot.rename({"ISO3": "Region", "Year": "Time"})
        .sel(Component="Total")
        .drop_vars("Component")
        - xr_nwc_tot.rename({"ISO3": "Region", "Year": "Time"})
        .sel(Component="LULUCF")
        .drop_vars("Component")
        + xr_primap_agri / 1e6
    ).rename({"Data": "GHG_hist_excl"})
    xr_co2excl = (
        xr_nwc_co2.rename({"ISO3": "Region", "Year": "Time"})
        .sel(Component="Total")
        .drop_vars("Component")
        - xr_nwc_co2.rename({"ISO3": "Region", "Year": "Time"})
        .sel(Component="LULUCF")
        .drop_vars("Component")
        + xr_primap_agri_co2 / 1e6
    ).rename({"Data": "CO2_hist_excl"})

    # Combine historical data into single xarray dataset
    regions_iso = list(regions.values())
    xr_hist = (
        xr.merge([xr_ghghist, xr_ghgexcl, xr_co2hist, xr_co2excl, xr_ch4hist, xr_n2ohist]) * 1e3
    ).reindex({"Region": regions_iso})

    # Store LULUCF (?)
    xr_ghg_afolu = (
        xr_nwc_tot.rename({"ISO3": "Region", "Year": "Time"})
        .sel(Component="LULUCF")
        .drop_vars("Component")
    )

    # Change units of agri
    # TODO: move this line closer to other agri steps?
    xr_ghg_agri = xr_primap_agri / 1e6

    # Also read EDGAR for purposes of using CR data (note that this is GHG excl LULUCF)
    # TODO: move to separate function?
    df_edgar = (
        pd.read_excel(data_root / edgar_file, sheet_name="GHG_totals_by_country")
        .drop(["Country"], axis=1)
        .set_index("EDGAR Country Code")
    )
    df_edgar.columns = df_edgar.columns.astype(int)

    # drop second-to-last row
    df_edgar = df_edgar.drop(df_edgar.index[-2])

    # Rename index column
    df_edgar.index.name = "Region"

    # Melt time columns into a time index
    df_edgar = (
        df_edgar.reset_index()
        .melt(id_vars="Region", var_name="Time", value_name="Emissions")
        .set_index(["Region", "Time"])
    )

    # Convert to xarray
    xr_edgar = df_edgar.to_xarray()

    xr_primap = xr_primap2.rename(
        {
            "area (ISO3)": "Region",
            "scenario (PRIMAP-hist)": "Scenario",
            "category (IPCC2006_PRIMAP)": "Category",
        }
    ).sel(provenance="derived", source="PRIMAP-hist_v2.5.1_final_nr")
    xr_primap = xr_primap.assign_coords(time=xr_primap.time.dt.year)

    # Add EU (this is required for the NDC data reading)
    df = pd.read_excel(data_root / country_groups_file, sheet_name="Country groups")
    countries_iso = np.array(df["Country ISO Code"])
    group_eu = countries_iso[np.array(df["EU"]) == 1]
    xr_hist.GHG_hist.loc[dict(Region="EU")] = xr_hist.GHG_hist.sel(Region=group_eu).sum("Region")

    return (
        xr_hist,
        # xr_ghg_afolu,  # TODO: not used, remove?
        # xr_ghg_agri,  # TODO: not used, remove?
        # xr_edgar,  # TODO: not used, remove?
        xr_primap,
    )


def read_ar6(config: Config, xr_hist):
    logger.info("Read AR6 data")

    # Define input
    data_root = config.paths.input
    filename = "AR6_Scenarios_Database_World_v1.1.csv"
    metadata_file = "AR6_Scenarios_Database_metadata_indicators_v1.1.xlsx"
    elevate_snapshot = "elevate-internal_snapshot_1739887620.csv"

    df_ar6raw = pd.read_csv(data_root / filename)
    df_ar6 = df_ar6raw[
        df_ar6raw.Variable.isin(
            [
                "Emissions|CO2",
                "Emissions|CO2|AFOLU",
                "Emissions|Kyoto Gases",
                "Emissions|CO2|Energy and Industrial Processes",
                "Emissions|CH4",
                "Emissions|N2O",
                "Emissions|CO2|AFOLU|Land",
                "Emissions|CH4|AFOLU|Land",
                "Emissions|N2O|AFOLU|Land",
                "Carbon Sequestration|CCS",
                "Carbon Sequestration|Land Use",
                "Carbon Sequestration|Direct Air Capture",
                "Carbon Sequestration|Enhanced Weathering",
                "Carbon Sequestration|Other",
                "Carbon Sequestration|Feedstocks",
                "AR6 climate diagnostics|Exceedance Probability 1.5C|MAGICCv7.5.3",
                "AR6 climate diagnostics|Exceedance Probability 2.0C|MAGICCv7.5.3",
                "AR6 climate diagnostics|Exceedance Probability 2.5C|MAGICCv7.5.3",
                "AR6 climate diagnostics|Exceedance Probability 3.0C|MAGICCv7.5.3",
                "AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|5.0th Percentile",
                "AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|33.0th Percentile",
                "AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|50.0th Percentile",
                "AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|67.0th Percentile",
                "AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|95.0th Percentile",
            ]
        )
    ]
    df_ar6 = df_ar6.reset_index(drop=True)
    idx = (
        df_ar6[(df_ar6.Variable == "Emissions|CH4") & (df_ar6["2100"] > 1e5)]
    ).index  # Removing erroneous CH4 scenarios
    df_ar6 = df_ar6[~df_ar6.index.isin(idx)]
    df_ar6 = df_ar6.reset_index(drop=True)

    df_ar6_meta = pd.read_excel(data_root / metadata_file, sheet_name="meta_Ch3vetted_withclimate")
    mods = np.array(df_ar6_meta.Model)
    scens = np.array(df_ar6_meta.Scenario)
    modscens_meta = np.array([mods[i] + "|" + scens[i] for i in range(len(scens))])
    df_ar6_meta["ModelScenario"] = modscens_meta
    df_ar6_meta = df_ar6_meta[["ModelScenario", "Category", "Policy_category"]]
    mods = np.array(df_ar6.Model)
    scens = np.array(df_ar6.Scenario)
    modscens = np.array([mods[i] + "|" + scens[i] for i in range(len(scens))])
    df_ar6["ModelScenario"] = modscens
    df_ar6 = df_ar6.drop(["Model", "Scenario", "Region", "Unit"], axis=1)
    dummy = df_ar6.melt(id_vars=["ModelScenario", "Variable"], var_name="Time", value_name="Value")
    dummy["Time"] = np.array(dummy["Time"].astype(int))
    dummy = dummy.set_index(["ModelScenario", "Variable", "Time"])
    xr_scen2 = xr.Dataset.from_dataframe(dummy)
    xr_scen2 = xr_scen2.reindex(Time=np.arange(2000, 2101, 10))
    xr_scen2 = xr_scen2.reindex(Time=np.arange(2000, 2101))
    xr_ar6_prevet = xr_scen2.interpolate_na(dim="Time", method="linear")

    recent_increment = int(config.params.start_year_analysis // 5 * 5)
    vetting_nans = np.array(
        xr_ar6_prevet.ModelScenario[
            ~np.isnan(xr_ar6_prevet.Value.sel(Time=2100, Variable="Emissions|CO2"))
        ]
    )
    vetting_recentyear = np.array(
        xr_ar6_prevet.ModelScenario[
            np.where(
                np.abs(
                    xr_ar6_prevet.sel(Time=recent_increment, Variable="Emissions|CO2").Value
                    - xr_hist.sel(Region="EARTH", Time=recent_increment).CO2_hist
                )
                < 1e4
            )[0]
        ]
    )
    vetting_total = np.intersect1d(vetting_nans, vetting_recentyear)
    xr_ar6 = xr_ar6_prevet.sel(ModelScenario=vetting_total)
    ms_immediate = np.array(
        df_ar6_meta[df_ar6_meta.Policy_category.isin(["P2", "P2a", "P2b", "P2c"])].ModelScenario
    )
    ms_delayed = np.array(
        df_ar6_meta[df_ar6_meta.Policy_category.isin(["P3a", "P3b", "P3c"])].ModelScenario
    )

    # TODO: shouldn't ch4 also be divided by 1000? That's what was done above in read_jones...
    xr_ar6_landuse = (
        xr_ar6.sel(Variable="Emissions|CO2|AFOLU|Land") * 1
        + xr_ar6.sel(Variable="Emissions|CH4|AFOLU|Land") * config.params.gwp_ch4
        + xr_ar6.sel(Variable="Emissions|N2O|AFOLU|Land") * config.params.gwp_n2o / 1000
    )
    xr_ar6_landuse = xr_ar6_landuse.rename({"Value": "GHG_LULUCF"})
    xr_ar6_landuse = xr_ar6_landuse.assign(
        CO2_LULUCF=xr_ar6.sel(Variable="Emissions|CO2|AFOLU|Land").Value
    )

    # Take averages of GHG excluding land use for the C-categories (useful for the Robiou paper)
    xr_both = xr.merge([xr_ar6, xr_ar6_landuse])
    xr_ar6_nozeros = xr_both.where(xr_both > -1e9, np.nan).where(xr_both != 0, np.nan)
    xr_averages = []
    for i in range(6):
        C = [["C1"], ["C1", "C2"], ["C2"], ["C3"], ["C6"], ["C7"]][i]
        Cname = ["C1", "C1+C2", "C2", "C3", "C6", "C7"][i]
        C_cat = np.intersect1d(
            np.array(xr_ar6_nozeros.ModelScenario),
            np.array(df_ar6_meta[df_ar6_meta.Category.isin(C)].ModelScenario),
        )
        xr_averages.append(
            xr_ar6_nozeros.sel(ModelScenario=C_cat)
            .mean(dim="ModelScenario")
            .expand_dims(Category=[Cname])
        )
    xr_av = xr.merge(xr_averages)
    xr_ar6_C = xr.merge(
        [
            (xr_av.Value.sel(Variable="Emissions|Kyoto Gases") - xr_av.GHG_LULUCF)
            .to_dataset(name="GHG_excl_C")
            .drop_vars("Variable"),
            (xr_av.Value.sel(Variable="Emissions|CO2") - xr_av.CO2_LULUCF)
            .to_dataset(name="CO2_excl_C")
            .drop_vars("Variable"),
            (
                xr_av.Value.sel(
                    Variable=[
                        "Carbon Sequestration|CCS",
                        "Carbon Sequestration|Direct Air Capture",
                    ]
                ).sum(dim="Variable", skipna=False)
            ).to_dataset(name="CO2_neg_C"),
        ]
    )
    xr_ar6_C = xr_ar6_C.reindex(Time=np.arange(2000, 2101, 10))
    xr_ar6_C = xr_ar6_C.reindex(Time=np.arange(2000, 2101))

    # Bunker subtraction
    # TODO: move to separate function?
    df_elevate_bunkers = pd.read_csv(data_root / elevate_snapshot)[:-1]
    mods = np.array(df_elevate_bunkers.Model)
    scens = np.array(df_elevate_bunkers.Scenario)
    modscens = np.array([mods[i] + "|" + scens[i] for i in range(len(scens))])
    df_elevate_bunkers["ModelScenario"] = modscens
    df_elevate_bunkers = df_elevate_bunkers.drop(["Model", "Scenario", "Region", "Unit"], axis=1)
    dummy = df_elevate_bunkers.melt(
        id_vars=["ModelScenario", "Variable"], var_name="Time", value_name="Value"
    )
    dummy["Time"] = np.array(dummy["Time"].astype(int))
    dummy = dummy.set_index(["ModelScenario", "Variable", "Time"])
    xr_elevate_bunkers = xr.Dataset.from_dataframe(dummy)
    xr_elevate_bunkers = xr_elevate_bunkers.reindex({"Time": np.arange(2010, 2101, 10)})

    modscens = np.array(xr_elevate_bunkers.ModelScenario)
    categories = []
    for ms in modscens:
        if (
            xr_elevate_bunkers.sel(
                ModelScenario=ms,
                Variable="AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|50.0th Percentile",
            )
            .max()
            .Value
            < 1.5
        ):
            categories.append("C1")
        elif (
            xr_elevate_bunkers.sel(
                ModelScenario=ms,
                Variable="AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|50.0th Percentile",
            )
            .max()
            .Value
            < 1.7
        ):
            categories.append("C2")
        elif (
            xr_elevate_bunkers.sel(
                ModelScenario=ms,
                Variable="AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|50.0th Percentile",
            )
            .max()
            .Value
            < 2.0
        ):
            categories.append("C3")
        elif (
            xr_elevate_bunkers.sel(
                ModelScenario=ms,
                Variable="AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|50.0th Percentile",
            )
            .max()
            .Value
            < 3.0
        ):
            categories.append("C6")
        elif (
            xr_elevate_bunkers.sel(
                ModelScenario=ms,
                Variable="AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|50.0th Percentile",
            )
            .max()
            .Value
            < 4.0
        ):
            categories.append("C7")
        elif (
            xr_elevate_bunkers.sel(
                ModelScenario=ms,
                Variable="AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|50.0th Percentile",
            )
            .max()
            .Value
            > 4.0
        ):
            categories.append("C8")
        else:
            categories.append(
                "C9"
            )  # Fictuous category to show that for these mds, there is no temperature assessment
    categories = np.array(categories)

    xrs = []
    for temp in [1.7, 2.0, 3.0]:
        if temp == 1.7:
            cat = "C2"
        elif temp == 2.0:
            cat = "C3"
        elif temp == 3.0:
            cat = "C6"
        xrs.append(
            (
                xr_elevate_bunkers.sel(
                    Variable="Emissions|CO2|Energy|Demand|Bunkers",
                    ModelScenario=modscens[categories == cat],
                ).median(dim="ModelScenario")
            ).expand_dims(Temperature=[temp])
        )
    xr_all = xr.concat(xrs, dim="Temperature")
    xr_all = xr_all.reindex(Temperature=[1.5, 1.6, 1.7, 2.0, 3.0, 4.0, 4.5])

    # Extrapolation
    vals = xr_all.loc[dict(Temperature=3.0)] - xr_all.loc[dict(Temperature=1.7)]
    vals = vals.where(vals > 0, 0)
    xr_all.loc[dict(Temperature=1.5)] = xr_all.loc[dict(Temperature=1.7)] - vals * (0.2 / 1.3)
    xr_all.loc[dict(Temperature=1.6)] = xr_all.loc[dict(Temperature=1.7)] - vals * (0.1 / 1.3)
    xr_all.loc[dict(Temperature=4.0)] = xr_all.loc[dict(Temperature=3.0)] + vals * (1.0 / 1.3)
    xr_all.loc[dict(Temperature=4.5)] = xr_all.loc[dict(Temperature=3.0)] + vals * (1.5 / 1.3)

    xr_all = (
        xr_all.rename({"Temperature": "Category"})
        .drop_vars("Variable")
        .rename({"Value": "CO2_bunkers_C"})
    )

    # Rename ticks of temperature
    xr_all = xr_all.assign_coords(Category=["C1", "C1+C2", "C2", "C3", "C6", "C7", "C8"])
    xr_ar6_C_bunkers = xr_all

    ar6_data = xr.merge(
        [
            # xr_ar6_prevet.rename_vars({"Value": "xr_ar6_prevet"}),
            xr_ar6.rename_vars({"Value": "xr_ar6"}),  # TODO: better name?
            xr_ar6_landuse,
            xr_ar6_C,
            xr_ar6_C_bunkers,
        ]
    )

    scenarios = {
        "Immediate": ms_immediate.tolist(),
        "Delayed": ms_delayed.tolist(),
    }

    return ar6_data, scenarios


def read_baseline(
    config: Config,
    countries,  # TODO: pass in region instead??
    xr_hist,
):
    logger.info("Reading baseline emissions")

    data_root = config.paths.input
    start_year = config.params.start_year_analysis
    countries_iso = list(countries.values())

    xr_bases = []
    for i in range(3):
        # In the up-to-date baselines, only SSP1, 2 and 3 are included. Will be updated at some point.
        df_base = pd.read_excel(data_root / f"SSP{i + 1}.xlsx", sheet_name="Sheet1")
        df_base = df_base[df_base["Unnamed: 1"] == "Emissions|CO2|Energy"]
        df_base = df_base.drop(["Unnamed: 1"], axis=1)
        df_base = df_base.rename(columns={"COUNTRY": "Region"})
        df_base["Scenario"] = ["SSP" + str(i + 1)] * len(df_base)

        # Melt time index
        df_base = df_base.melt(
            id_vars=["Region", "Scenario"], var_name="Time", value_name="CO2_base_excl"
        )
        df_base["Time"] = np.array(df_base["Time"].astype(int))

        # Convert to xarray
        dummy = df_base.set_index(["Region", "Scenario", "Time"])
        dummy = dummy.astype(float)
        xr_bases.append(xr.Dataset.from_dataframe(dummy))

    xr_base = xr.merge(xr_bases).reindex({"Region": countries_iso})

    # Assign 2020 values in Time index
    xr_base = xr_base.reindex(Time=np.arange(start_year, 2101))
    for year in np.arange(start_year, 2021):
        xr_base.CO2_base_excl.loc[dict(Time=year, Region=countries_iso)] = xr_hist.sel(
            Time=year, Region=countries_iso
        ).CO2_hist_excl

    # TODO: might be useful to create a helper function like so:
    def mask_outside(data, lower=-1e9, upper=1e9):
        cond1 = data > lower
        cond2 = data < upper
        return data.where(cond1 & cond2)

    # Harmonize emissions from historical values to baseline emissions
    diffrac = xr_hist.CO2_hist_excl.sel(Time=start_year) / xr_base.CO2_base_excl.sel(
        Time=start_year
    )
    diffrac = diffrac.where(diffrac < 1e9)
    diffrac = diffrac.where(diffrac > -1e9)
    xr_base = xr_base.assign(CO2_base_excl=xr_base.CO2_base_excl * diffrac)

    # Using a fraction, get other emissions variables
    fraction_startyear_co2_incl = (
        xr_hist.sel(Time=start_year).CO2_hist / xr_hist.sel(Time=start_year).CO2_hist_excl
    )
    fraction_startyear_co2_incl = fraction_startyear_co2_incl.where(
        fraction_startyear_co2_incl < 1e9
    )
    fraction_startyear_co2_incl = fraction_startyear_co2_incl.where(
        fraction_startyear_co2_incl > -1e9
    )
    xr_base = xr_base.assign(CO2_base_incl=xr_base.CO2_base_excl * fraction_startyear_co2_incl)

    fraction_startyear_ghg_excl = (
        xr_hist.sel(Time=start_year).GHG_hist_excl / xr_hist.sel(Time=start_year).CO2_hist_excl
    )
    fraction_startyear_ghg_excl = fraction_startyear_ghg_excl.where(
        fraction_startyear_ghg_excl < 1e9
    )
    fraction_startyear_ghg_excl = fraction_startyear_ghg_excl.where(
        fraction_startyear_ghg_excl > -1e9
    )
    xr_base = xr_base.assign(GHG_base_excl=xr_base.CO2_base_excl * fraction_startyear_ghg_excl)

    fraction_startyear_ghg_incl = (
        xr_hist.sel(Time=start_year).GHG_hist / xr_hist.sel(Time=start_year).CO2_hist_excl
    )
    fraction_startyear_ghg_incl = fraction_startyear_ghg_incl.where(
        fraction_startyear_ghg_incl < 1e9
    )
    fraction_startyear_ghg_incl = fraction_startyear_ghg_incl.where(
        fraction_startyear_ghg_incl > -1e9
    )
    xr_base = xr_base.assign(GHG_base_incl=xr_base.CO2_base_excl * fraction_startyear_ghg_incl)

    # Assign 2020 values in Time index
    xr_base = xr_base.reindex(Time=np.arange(start_year, 2101))
    for year in np.arange(start_year, 2021):
        xr_base.GHG_base_excl.loc[dict(Time=year, Region=countries_iso)] = xr_hist.sel(
            Time=year, Region=countries_iso
        ).GHG_hist_excl
        xr_base.CO2_base_incl.loc[dict(Time=year, Region=countries_iso)] = xr_hist.sel(
            Time=year, Region=countries_iso
        ).CO2_hist
        xr_base.GHG_base_incl.loc[dict(Time=year, Region=countries_iso)] = xr_hist.sel(
            Time=year, Region=countries_iso
        ).GHG_hist

    # Harmonize global baseline emissions with sum of all countries (this is important for consistency of AP, etc.)
    base_onlyc = xr_base.reindex(Region=countries_iso)
    base_w = base_onlyc.sum(dim="Region").expand_dims({"Region": ["EARTH"]})
    xr_base = xr.merge([base_w, base_onlyc])

    return xr_base


def load_emissions(config: Config, from_intermediate=True, save=True):
    """Collect emission input data from various sources to intermediate file.

    Args:
        config: effortsharing.config.Config object
        from_intermediate: Whether to read from intermediate files if available (default: True)
        save: Whether to save intermediate data to disk (default: True)

    Returns:
        xarray.Dataset: Emission data
    """

    save_path = config.paths.intermediate / "emissions.nc"
    save_path_scenarios = config.paths.intermediate / "emissions.json"

    # Check if we can load from intermediate file
    if from_intermediate and save_path.exists():
        logger.info(f"Loading emission data from {save_path}")

        emission_data = xr.load_dataset(save_path)

        logger.info(f"Loading emission scenarios from {save_path_scenarios}")
        with open(save_path_scenarios) as f:
            scenarios = json.load(f)

        return emission_data, scenarios

    # Otherwise, process raw input files
    logger.info("Processing emission input data")

    countries, regions = _regions.read_general(config)

    xr_hist, xr_primap = read_historicalemis_jones(config, regions)
    xr_base = read_baseline(config, countries, xr_hist)
    xr_ar6, scenarios = read_ar6(config, xr_hist)

    # Merge datasets
    emission_data = xr.merge(
        [
            xr_hist,
            xr_base,
            xr_ar6,
        ]
    )
    # TODO: Reindex time and regions??

    # Save to disk
    if save:
        logger.info(f"Saving emission data to {save_path}")

        config.paths.intermediate.mkdir(parents=True, exist_ok=True)
        emission_data.to_netcdf(save_path)
        # TODO: add compression

        # Also save scenarios to JSON
        # TODO: maybe we can derive them on the fly rather than saving them?
        logger.info(f"Saving scenarios to {save_path_scenarios}")
        with open(save_path_scenarios, "w") as f:
            json.dump(scenarios, f)

    return emission_data, scenarios


if __name__ == "__main__":
    import argparse

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Get the config file from command line arguments
    parser = argparse.ArgumentParser(description="Process emission input data")
    parser.add_argument("config", help="Path to config file")
    args = parser.parse_args()

    # Read config
    config = Config.from_file(args.config)

    # Process emission data and save to intermediate file
    load_emissions(config, from_intermediate=False, save=True)
