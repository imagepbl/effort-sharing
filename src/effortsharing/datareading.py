# ======================================== #
# Class that does the data reading
# ======================================== #

# =========================================================== #
# PREAMBULE
# Put in packages that we need
# =========================================================== #

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

import effortsharing.regions as _regions
from effortsharing.config import Config


@dataclass
class General:
    countries: dict[str, str]
    regions: dict[str, str]

@dataclass
class UNPopulation:
    population: xr.Dataset
    population_long: xr.Dataset

@dataclass
class JonesData:
    xr_hist: xr.Dataset
    xr_ghg_afolu: xr.Dataset
    xr_ghg_agri: xr.DataArray
    xr_edgar: xr.Dataset
    xr_primap: xr.Dataset

@dataclass
class AR6Data:
    xr_ar6_prevet: xr.Dataset
    xr_ar6: xr.Dataset
    ms_immediate: np.ndarray
    ms_delayed: np.ndarray
    xr_ar6_landuse: xr.Dataset
    xr_ar6_C: xr.Dataset
    xr_ar6_C_bunkers: xr.Dataset


@dataclass
class NonCO2Data:
    xr_temperatures: xr.Dataset
    xr_nonco2warmings: xr.Dataset
    xr_nonco2warming_wrt_start: xr.Dataset


# TODO: perhaps combine this class with the one before?
@dataclass
class NonCO2Trajectories:
    xr_traj_nonco2: xr.Dataset
    xr_traj_nonco2_2: xr.Dataset
    xr_traj_nonco2_adapt: xr.Dataset | None


@dataclass
class GlobalBudgets:
    xr_bud_co2: xr.Dataset
    xr_co2_budgets: xr.Dataset


@dataclass
class GlobalCO2:
    xr_traj_co2: xr.Dataset
    xr_traj_ghg: xr.Dataset
    landuse_ghg_corr: xr.Dataset
    landuse_co2_corr: xr.Dataset
    xr_traj_ghg_excl: xr.Dataset
    xr_traj_co2_excl: xr.Dataset
    all_projected_gases: xr.Dataset


def read_general(config: Config) -> General:
    """Read country names and ISO from UNFCCC table."""
    print("- Reading unfccc country data")

    data_root = config.paths.input
    filename = "UNFCCC_Parties_Groups_noeu.xlsx"

    # Read and transform countries
    columns = {"Name": "name", "Country ISO Code": "iso"}
    countries = (
        pd.read_excel(
            data_root / filename,
            sheet_name="Country groups",
            usecols=columns.keys(),
        )
        .rename(columns=columns)
        .set_index("name")["iso"]
        .to_dict()
    )

    # Extend countries with non-country regions
    regions = {**countries, **_regions.ADDITIONAL_EU_AND_EARTH}

    return General(countries, regions)


def read_ssps(config, regions):
    print("- Reading GDP and population data from SSPs")

    # Define input
    data_root = config.paths.input
    filename = "SSPs_v2023.xlsx"

    for i in range(6):
        df_ssp = pd.read_excel(
            data_root / filename,
            sheet_name="data",
        )
        if i >= 1:
            df_ssp = df_ssp[
                (df_ssp.Model.isin(["OECD ENV-Growth 2023"]))
                & (df_ssp.Scenario == "Historical Reference")
            ]
        else:
            df_ssp = df_ssp[
                (df_ssp.Model.isin(["OECD ENV-Growth 2023", "IIASA-WiC POP 2023"]))
                & (df_ssp.Scenario.isin(["SSP1", "SSP2", "SSP3", "SSP4", "SSP5"]))
            ]
        region_full = np.array(df_ssp.Region)
        region_iso = []
        for r in region_full:
            iso = regions.get(r, None)
            if iso is None:
                # TODO: read from self.regions_iso instead ?!
                iso = _regions.ADDITIONAL_REGIONS_SSPS.get(r, None)
            if iso is None:
                print(r)
                iso = "oeps"
            region_iso.append(iso)
        df_ssp["Region"] = region_iso
        variable = np.array(df_ssp["Variable"])
        variable[variable == "GDP|PPP"] = "GDP"
        df_ssp["Variable"] = variable
        df_ssp = df_ssp.drop(["Model", "Unit"], axis=1)
        dummy = df_ssp.melt(
            id_vars=["Scenario", "Region", "Variable"],
            var_name="Time",
            value_name="Value",
        )
        dummy["Time"] = np.array(dummy["Time"].astype(int))
        if i >= 1:
            dummy["Scenario"] = ["SSP" + str(i)] * len(dummy)
            xr_hist_gdp_i = xr.Dataset.from_dataframe(
                dummy.pivot(
                    index=["Scenario", "Region", "Time"],
                    columns="Variable",
                    values="Value",
                )
            ).sel(Time=[1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015])
            xr_ssp = xr.merge([xr_ssp, xr_hist_gdp_i])
        else:
            xr_ssp = (
                xr.Dataset.from_dataframe(
                    dummy.pivot(
                        index=["Scenario", "Region", "Time"],
                        columns="Variable",
                        values="Value",
                    )
                )
                .reindex({"Time": np.arange(2020, 2101, 5)})
                .reindex({"Time": np.arange(1980, 2101, 5)})
            )

        return xr_ssp


def read_un_population(config, countries) -> UNPopulation:
    print("- Reading UN population data and gapminder, processed by OWID (for past population)")

    # Define input
    data_root = config.paths.input
    filename = "population_HYDE_UNP_Gapminder.csv"

    df_pop = pd.read_csv(data_root / filename)[["Code", "Year", "Population (historical)"]].rename(
        {"Code": "Region", "Population (historical)": "Population", "Year": "Time"},
        axis=1,
    )
    reg = np.array(df_pop.Region)
    reg[reg == "OWID_WRL"] = "EARTH"
    df_pop.Region = reg

    xr_unp_long = (
        xr.Dataset.from_dataframe(
            df_pop[df_pop.Region.isin(list(countries.values()) + ["EARTH"])].set_index(
                ["Region", "Time"]
            )
        )
        / 1e6
    )
    xr_unp = xr_unp_long.sel(Time=np.arange(1850, 2000))
    return UNPopulation(xr_unp, xr_unp_long)


def read_hdi(config, countries, unpopulation: UNPopulation):
    print("- Read Human Development Index data")

    # Define input
    data_root = config.paths.input
    regions_file = "AR6_regionclasses.xlsx"
    hdi_file = "HDR21-22_Statistical_Annex_HDI_Table.xlsx"

    df_regions = pd.read_excel(data_root / regions_file)
    df_regions = df_regions.sort_values(by=["name"])
    df_regions = df_regions.sort_index()

    df_hdi_raw = pd.read_excel(data_root / hdi_file, sheet_name="Rawdata")
    hdi_countries_raw = np.array(df_hdi_raw.Country)
    hdi_values_raw = np.array(df_hdi_raw.HDI).astype(str)
    hdi_values_raw[hdi_values_raw == ".."] = "nan"
    hdi_values_raw = hdi_values_raw.astype(float)
    hdi_av = np.nanmean(hdi_values_raw)

    countries_name = list(countries.keys())
    countries_iso = list(countries.values())

    # Construct new hdi object
    hdi_values = np.zeros(len(countries_iso)) + np.nan
    hdi_sh_values = np.zeros(len(countries_iso)) + np.nan
    for r_i, r in enumerate(countries_iso):
        reg = countries_name[r_i]
        wh = np.where(hdi_countries_raw == reg)[0]
        if len(wh) > 0:
            wh_i = wh[0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif r in [
            "ALA",
            "ASM",
            "AIA",
            "ABW",
            "BMU",
            "ANT",
            "SCG",
            "BES",
            "BVT",
            "IOT",
            "VGB",
            "CYM",
            "CXR",
            "CCK",
            "COK",
            "CUW",
            "FLK",
            "FRO",
            "GUF",
            "PYF",
            "ATF",
            "GMB",
            "GIB",
            "GRL",
            "GLP",
            "GUM",
            "GGY",
            "HMD",
            "VAT",
            "IMN",
            "JEY",
            "MAC",
            "MTQ",
            "MYT",
            "MSR",
            "NCL",
            "NIU",
            "NFK",
            "MNP",
            "PCN",
            "PRI",
            "REU",
            "BLM",
            "SHN",
            "SPM",
            "SXM",
            "SGS",
            "MAF",
            "SJM",
            "TKL",
            "TCA",
            "UMI",
            "VIR",
            "WLF",
            "ESH",
        ]:
            hdi_values[r_i] = np.nan
        elif r == "USA":
            wh = np.where(hdi_countries_raw == "United States")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif r == "BHS":
            wh = np.where(hdi_countries_raw == "Bahamas")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif r == "GMB":
            wh = np.where(hdi_countries_raw == "Gambia")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif r == "CPV":
            wh = np.where(hdi_countries_raw == "Cabo Verde")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif r == "BOL":
            wh = np.where(hdi_countries_raw == "Bolivia (Plurinational State of)")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif r == "COD":
            wh = np.where(hdi_countries_raw == "Congo (Democratic Republic of the)")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif r == "COG":
            wh = np.where(hdi_countries_raw == "Congo")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif r == "CZE":
            wh = np.where(hdi_countries_raw == "Czechia")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif r == "EGY":
            wh = np.where(hdi_countries_raw == "Egypt")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif r == "HKG":
            wh = np.where(hdi_countries_raw == "Hong Kong, China (SAR)")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif r == "IRN":
            wh = np.where(hdi_countries_raw == "Iran (Islamic Republic of)")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif r == "PRK":
            wh = np.where(hdi_countries_raw == "Korea (Democratic People's Rep. of)")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif r == "KOR":
            wh = np.where(hdi_countries_raw == "Korea (Republic of)")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif r == "KGZ":
            wh = np.where(hdi_countries_raw == "Kyrgyzstan")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif r == "LAO":
            wh = np.where(hdi_countries_raw == "Lao People's Democratic Republic")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif r == "FSM":
            wh = np.where(hdi_countries_raw == "Micronesia (Federated States of)")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif r == "MDA":
            wh = np.where(hdi_countries_raw == "Moldova (Republic of)")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif r == "STP":
            wh = np.where(hdi_countries_raw == "Sao Tome and Principe")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif r == "SVK":
            wh = np.where(hdi_countries_raw == "Slovakia")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif r == "KNA":
            wh = np.where(hdi_countries_raw == "Saint Kitts and Nevis")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif r == "LCA":
            wh = np.where(hdi_countries_raw == "Saint Lucia")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif r == "VCT":
            wh = np.where(hdi_countries_raw == "Saint Vincent and the Grenadines")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif r == "SWZ":
            wh = np.where(hdi_countries_raw == "Eswatini (Kingdom of)")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif r == "TWN":
            wh = np.where(hdi_countries_raw == "China")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif r == "TZA":
            wh = np.where(hdi_countries_raw == "Tanzania (United Republic of)")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif r == "TUR":
            wh = np.where(hdi_countries_raw == "Türkiye")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif r == "VEN":
            wh = np.where(hdi_countries_raw == "Venezuela (Bolivarian Republic of)")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif r == "VNM":
            wh = np.where(hdi_countries_raw == "Viet Nam")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif r == "PSE":
            wh = np.where(hdi_countries_raw == "Palestine, State of")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif r == "YEM":
            wh = np.where(hdi_countries_raw == "Yemen")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        else:
            hdi_values[r_i] = np.nan
        try:
            pop = float(unpopulation.population_long.sel(Region=r, Time=2019).Population)
        except:
            pop = np.nan
        hdi_sh_values[r_i] = hdi_values[r_i] * pop
    hdi_sh_values = hdi_sh_values / np.nansum(hdi_sh_values)
    df_hdi = {}
    df_hdi["Region"] = countries_iso
    df_hdi["Name"] = countries_name
    df_hdi["HDI"] = hdi_values
    df_hdi = pd.DataFrame(df_hdi)
    df_hdi = df_hdi[["Region", "HDI"]]
    dfdummy = df_hdi.set_index(["Region"])
    xr_hdi = xr.Dataset.from_dataframe(dfdummy)

    df_hdi = {}
    df_hdi["Region"] = countries_iso
    df_hdi["Name"] = countries_name
    df_hdi["HDIsh"] = hdi_sh_values
    df_hdi = pd.DataFrame(df_hdi)
    df_hdi = df_hdi[["Region", "HDIsh"]]
    dfdummy = df_hdi.set_index(["Region"])
    xr_hdish = xr.Dataset.from_dataframe(dfdummy)

    return xr_hdi, xr_hdish


def read_historicalemis_jones(config: Config, regions) -> JonesData:
    # No harmonization with the KEV anymore, but it's also much closer now
    print("- Reading historical emissions (jones)")

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

    return JonesData(xr_hist, xr_ghg_afolu, xr_ghg_agri, xr_edgar, xr_primap)


def read_ar6(config: Config, xr_hist) -> AR6Data:
    print("- Read AR6 data")

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

    return AR6Data(
        xr_ar6_prevet,
        xr_ar6,
        ms_immediate,
        ms_delayed,
        xr_ar6_landuse,
        xr_ar6_C,
        xr_ar6_C_bunkers,
    )


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

    return NonCO2Data(xr_temperatures, xr_nonco2warmings, xr_nonco2warming_wrt_start)


def determine_global_nonco2_trajectories(
    config: Config, ar6data: AR6Data, xr_hist, xr_temperatures
) -> NonCO2Trajectories:
    print("- Computing global nonco2 trajectories")

    # Shorthand for often-used expressions
    start_year = config.params.start_year_analysis
    n_years = 2101 - start_year

    # TODO: this can probably do without the rounding or casting to array
    dim_temp = np.array(config.dimension_ranges.peak_temperature).astype(float).round(2)
    dim_prob = np.array(config.dimension_ranges.risk_of_exceedance).round(2)
    dim_nonco2 = np.array(config.dimension_ranges.non_co2_reduction).round(2)
    dim_timing = np.array(config.dimension_ranges.timing_of_mitigation_action)

    # Relationship between non-co2 reduction and budget is based on Rogelj et al and requires the year 2020 (even though startyear may be different) - not a problem

    xr_ch4_raw = ar6data.xr_ar6.sel(Variable="Emissions|CH4") * config.params.gwp_ch4
    xr_n2o_raw = ar6data.xr_ar6.sel(Variable="Emissions|N2O") * config.params.gwp_n2o / 1e3
    n2o_start = xr_hist.sel(Region="EARTH").sel(Time=start_year).N2O_hist
    ch4_start = xr_hist.sel(Region="EARTH").sel(Time=start_year).CH4_hist
    n2o_2020 = xr_hist.sel(Region="EARTH").sel(Time=2020).N2O_hist
    ch4_2020 = xr_hist.sel(Region="EARTH").sel(Time=2020).CH4_hist
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
        return xr_temperatures.ModelScenario[
            np.where(np.abs(temp - xr_temperatures.Temperature.sel(Risk=risk)) < 0.2)[0]
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
                    mslist = ar6data.ms_immediate
                else:
                    mslist = ar6data.ms_delayed
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
                    reds = reductions.Value.quantile(dim_nonco2[::-1])
                    for n_i, n in enumerate(dim_nonco2):
                        red = reds[n_i]
                        ms2 = reductions.ModelScenario[
                            np.where(np.abs(reductions.Value - red) < 0.1)
                        ]
                        trajs = xr_nonco2_raw.sel(
                            ModelScenario=ms2,
                            Time=np.arange(start_year, 2101),
                        )
                        trajectory_mean = rescale(trajs.Value.mean(dim="ModelScenario"))

                        # Harmonize reduction
                        red_traj = (trajectory_mean.sel(Time=2040) - tot_2020) / tot_2020
                        traj2 = (
                            -(1 - xr_comp) * (red_traj - red) * xr_nonco2_raw_start.mean().Value
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

    return NonCO2Trajectories(xr_traj_nonco2, xr_traj_nonco2_2, xr_traj_nonco2_adapt)


def determine_global_budgets(config: Config, xr_hist, nonco2data: NonCO2Data) -> GlobalBudgets:
    print("- Get global CO2 budgets")

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
            budgets -= float(xr_hist.sel(Region="EARTH", Time=year).CO2_hist) / 1e3
    elif config.params.start_year_analysis < 2020:
        budgets = dummy["Budget"]
        for year in np.arange(config.params.start_year_analysis, 2020):
            budgets += float(xr_hist.sel(Region="EARTH", Time=year).CO2_hist) / 1e3
    dummy["Budget"] = budgets

    xr_bud_co2 = xr.Dataset.from_dataframe(dummy)
    xr_bud_co2 = xr_bud_co2.rename(
        {"dT_targets": "Temperature"}
    )  # .sel(Temperature = [1.5, 1.7, 2.0])
    xr_bud_co2 = xr_bud_co2

    # Determine bunker emissions to subtract from global budget
    bunker_subtraction = []
    for t_i, t in enumerate(dim_temp):
        bunker_subtraction += [
            3.3 / 100
        ]  # Assuming bunker emissions have a constant fraction of global emissions (3.3%) - https://www.pbl.nl/sites/default/files/downloads/pbl-2020-analysing-international-shipping-and-aviation-emissions-projections_4076.pdf

    Blist = np.zeros(shape=(len(dim_temp), len(dim_prob), len(dim_nonco2))) + np.nan

    def ms_temp(
        temp, risk
    ):  # 0.2 is quite wide, but useful for studying nonCO2 variation among scenarios (is a relatively metric anyway)
        return nonco2data.xr_temperatures.ModelScenario[
            np.where(np.abs(temp - nonco2data.xr_temperatures.Temperature.sel(Risk=risk)) < 0.2)[0]
        ].values

    for p_i, p in enumerate(dim_prob):
        a, b = np.polyfit(
            xr_bud_co2.Temperature, xr_bud_co2.sel(Probability=np.round(p, 2)).Budget, 1
        )
        for t_i, t in enumerate(dim_temp):
            ms = ms_temp(t, round(1 - p, 2))

            # This assumes that the budget from Forster implicitly includes the median nonCO2 warming among scenarios that meet that Temperature target
            # Hence, only deviation (dT) from this median is interesting to assess here
            dT = nonco2data.xr_nonco2warming_wrt_start.sel(
                ModelScenario=ms, Risk=round(1 - p, 2)
            ) - nonco2data.xr_nonco2warming_wrt_start.sel(
                ModelScenario=ms, Risk=round(1 - p, 2)
            ).median(dim="ModelScenario")
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

    return GlobalBudgets(xr_bud_co2, xr_co2_budgets)


def determine_global_co2_trajectories(
    config: Config,
    xr_hist,
    ar6data: AR6Data,
    xr_temperatures,
    xr_co2_budgets,
    xr_traj_nonco2,
) -> GlobalCO2:
    print("- Computing global co2 trajectories")

    # Shorthand for often-used expressions
    start_year = config.params.start_year_analysis

    # TODO: this can probably do without the rounding or casting to array
    dim_temp = np.array(config.dimension_ranges.peak_temperature).astype(float).round(2)
    dim_prob = np.array(config.dimension_ranges.risk_of_exceedance).round(2)
    dim_nonco2 = np.array(config.dimension_ranges.non_co2_reduction).round(2)
    dim_timing = np.array(config.dimension_ranges.timing_of_mitigation_action)
    dim_negemis = np.array(config.dimension_ranges.negative_emissions).round(2)

    # Initialize data arrays for co2
    startpoint = xr_hist.sel(Time=start_year, Region="EARTH").CO2_hist
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
    xr_scen2_use = ar6data.xr_ar6.sel(Variable="Emissions|CO2")
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
    xr_neg = ar6data.xr_ar6.sel(
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
                mslist = ar6data.ms_immediate
            else:
                mslist = ar6data.ms_delayed
            ms2 = np.intersect1d(ms1, mslist)
            emis2100_i = emis2100.sel(ModelScenario=ms2)

            # The 90-percentile of 2100 emissions
            ms_90 = ar6data.xr_ar6.sel(ModelScenario=ms2).ModelScenario[
                (emis2100_i >= emis2100_i.quantile(0.9 - 0.1)).Value
                & (emis2100_i <= emis2100_i.quantile(0.9 + 0.1)).Value
            ]

            # The 50-percentile of 2100 emissions
            ms_10 = ar6data.xr_ar6.sel(ModelScenario=ms2).ModelScenario[
                (emis2100_i >= emis2100_i.quantile(0.1 - 0.1)).Value
                & (emis2100_i <= emis2100_i.quantile(0.1 + 0.1)).Value
            ]

            # Difference and smoothen this
            surplus_factor = (
                emis_all.sel(ModelScenario=np.intersect1d(ms_90, ms2))
                .mean(dim="ModelScenario")
                .Value
                - emis_all.sel(ModelScenario=np.intersect1d(ms_10, ms2))
                .mean(dim="ModelScenario")
                .Value
            )
            surplus_factor2 = np.convolve(surplus_factor, np.ones(3) / 3, mode="valid")
            surplus_factor[1:-1] = surplus_factor2

            for neg_i, neg in enumerate(dim_negemis):
                xset = emis_all.sel(ModelScenario=ms2) - surplus_factor * (neg - 0.5)
                pathways_neg = xr_neg.sel(ModelScenario=ms1).quantile(neg, dim="ModelScenario")
                pathways_data["CO2_neg_globe"][neg_i, temp_i, :] = np.array(pathways_neg.Value)
                for risk_i, risk in enumerate(dim_prob):
                    for nonco2_i, nonco2 in enumerate(dim_nonco2):
                        factor = (
                            xr_co2_budgets.Budget.sel(Temperature=temp, Risk=risk, NonCO2red=nonco2)
                            - xset.where(xset.Value > 0).sum(dim="Time")
                        ) / np.sum(compensation_form)
                        all_pathways = (1e3 * (xset + factor * xr_comp)).Value / 1e3
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
                            pathway_final2 = np.array(
                                (1e3 * (pathway_final + factor * budget_harm(nz))) / 1e3
                            )

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
    # self.xr_traj_ghg = xr.merge([self.xr_traj_ghg_ds.to_dataset(name="GHG_globe"), self.xr_traj_co2.CO2_globe, self.xr_traj_co2.CO2_neg_globe, self.xr_traj_nonco2.NonCO2_globe])
    # x = (self.xr_ar6_landuse / self.xr_ar6.sel(Variable='Emissions|Kyoto Gases')).mean(dim='ModelScenario').Value
    # zero = np.arange(self.settings['params']['start_year_analysis'],2101)[np.where(x.sel(Time=np.arange(self.settings['params']['start_year_analysis'],2101))<0)[0][0]]
    # x0 = x*np.array(list(np.ones(zero-2000))+list(np.zeros(2101-zero)))
    # self.xr_traj_ghg_excl = (self.xr_traj_ghg.GHG_globe*(1-x0)).to_dataset(name='GHG_globe_excl')

    # projected land use emissions
    landuse_ghg = ar6data.xr_ar6_landuse.mean(dim="ModelScenario").GHG_LULUCF
    landuse_co2 = ar6data.xr_ar6_landuse.mean(dim="ModelScenario").CO2_LULUCF

    # historical land use emissions
    landuse_ghg_hist = (
        xr_hist.sel(Region="EARTH").GHG_hist - xr_hist.sel(Region="EARTH").GHG_hist_excl
    )
    landuse_co2_hist = (
        xr_hist.sel(Region="EARTH").CO2_hist - xr_hist.sel(Region="EARTH").CO2_hist_excl
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

    return GlobalCO2(
        xr_traj_co2,
        xr_traj_ghg,
        landuse_ghg_corr,
        landuse_co2_corr,
        xr_traj_ghg_excl,
        xr_traj_co2_excl,
        all_projected_gases,
    )


def read_baseline(
    config: Config,
    countries,  # TODO: pass in region instead??
    xr_hist,
):
    print("- Reading baseline emissions")

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


def read_ndc_climateresource(config: Config, countries):
    print("- Reading NDC data from Climate resource")

    countries_iso = list(countries.values())
    version_ndcs = config.params.version_ndcs
    data_root = config.paths.input / f"ClimateResource_{version_ndcs}"

    ghg_data = np.zeros(shape=(len(countries_iso) + 1, 3, 2, 2, len(np.arange(2010, 2051))))
    for cty_i, cty in enumerate(countries_iso):
        for cond_i, cond in enumerate(["conditional", "range", "unconditional"]):
            for hot_i, hot in enumerate(["include", "exclude"]):
                for amb_i, amb in enumerate(["low", "high"]):
                    filename = f"{cty.lower()}_ndc_{version_ndcs}_CR_{cond}_{hot}.json"
                    path = data_root / cond / hot / filename
                    try:
                        with open(path) as file:
                            json_data = json.load(file)
                        country_name = json_data["results"]["country"]["name"]
                        series_items = json_data["results"]["series"]
                        for item in series_items:
                            columns = item["columns"]
                            if (
                                columns["variable"] == "Emissions|Total GHG excl. LULUCF"
                                and columns["category"] == "Updated NDC"
                                and columns["ambition"] == amb
                            ):
                                data = item["data"]
                                time_values = [int(year) for year in data.keys()]
                                ghg_values = np.array(list(item["data"].values()))
                                ghg_values[ghg_values == "None"] = np.nan
                                ghg_values = ghg_values.astype(float)
                                ghg_values = ghg_values[np.array(time_values) >= 2010]
                                ghg_data[cty_i, cond_i, hot_i, amb_i] = ghg_values
                                # series.append([country_iso.upper(), country_name, "Emissions|Total GHG excl. LULUCF", conditionality, hot_air, ambition] + list(ghg_values))
                    except:
                        continue

    # Now also for EU
    for cond_i, cond in enumerate(["conditional", "range", "unconditional"]):
        for hot_i, hot in enumerate(["include", "exclude"]):
            for amb_i, amb in enumerate(["low", "high"]):
                filename = f"groupeu27_ndc_{version_ndcs}_CR_{cond}_{hot}.json"
                path = data_root / cond / hot / "regions" / filename
                try:
                    with open(path) as file:
                        json_data = json.load(file)
                    country_name = json_data["results"]["country"]["name"]
                    series_items = json_data["results"]["series"]
                    for item in series_items:
                        columns = item["columns"]
                        if (
                            columns["variable"] == "Emissions|Total GHG excl. LULUCF"
                            and columns["category"] == "Updated NDC"
                            and columns["ambition"] == amb
                        ):
                            data = item["data"]
                            time_values = [int(year) for year in data.keys()]
                            ghg_values = np.array(list(item["data"].values()))
                            ghg_values[ghg_values == "None"] = np.nan
                            ghg_values = ghg_values.astype(float)
                            ghg_values = ghg_values[np.array(time_values) >= 2010]
                            ghg_data[cty_i + 1, cond_i, hot_i, amb_i] = ghg_values
                            # series.append([country_iso.upper(), country_name, "Emissions|Total GHG excl. LULUCF", conditionality, hot_air, ambition] + list(ghg_values))
                except:
                    continue

    coords = {
        "Region": list(countries_iso) + ["EU"],
        "Conditionality": ["conditional", "range", "unconditional"],
        "Hot_air": ["include", "exclude"],
        "Ambition": ["min", "max"],
        "Time": np.array(time_values)[np.array(time_values) >= 2010],
    }
    data_vars = {
        "GHG_ndc_excl_CR": (
            ["Region", "Conditionality", "Hot_air", "Ambition", "Time"],
            ghg_data,
        ),
    }
    xr_ndc = xr.Dataset(data_vars, coords=coords)
    xr_ndc_CR = xr_ndc.sel(Time=2030)

    return xr_ndc_CR


def read_ndc(config: Config, countries, xr_hist):
    print("- Reading NDC data")

    data_root = config.paths.input
    filename = "Infographics PBL NDC Tool 4Oct2024_for CarbonBudgetExplorer.xlsx"

    # TODO: use package or better dict mapping method instead
    countries_name = np.array(list(countries.keys()))
    countries_iso = np.array(list(countries.values()))

    df_ndc_raw = pd.read_excel(
        data_root / filename, sheet_name="Reduction All_GHG_incl", header=[0, 1]
    )
    regs = df_ndc_raw["(Mt CO2 equivalent)"]["Country name"]
    regs_iso = []

    for r in regs:
        wh = np.where(countries_name == r)[0]
        if len(wh) == 0:
            if r == "United States":
                regs_iso.append("USA")
            elif r == "EU27":
                regs_iso.append("EU")
            elif r == "Turkey":
                regs_iso.append("TUR")
            else:
                regs_iso.append(np.nan)
        else:
            regs_iso.append(countries_iso[wh[0]])
    regs_iso = np.array(regs_iso)
    df_ndc_raw["ISO"] = regs_iso

    df_regs = []
    df_amb = []
    df_con = []
    df_emis = []
    df_lulucf = []
    df_red = []
    df_abs = []
    df_inv = []
    histemis = xr_hist.GHG_hist.sel(Time=2015)
    for r in list(countries_iso) + ["EU"]:
        histemis_r = float(histemis.sel(Region=r))
        df_ndc_raw_sub = df_ndc_raw[df_ndc_raw["ISO"] == r]
        if len(df_ndc_raw_sub) > 0:
            val_2015 = float(df_ndc_raw_sub["(Mt CO2 equivalent)"][2015])
            for lulucf in ["incl"]:  # Maybe add excl later?
                for emis_i, emis in enumerate(["NDC"]):  # , 'CP']):
                    key = ["2030 NDCs", "Domestic actions 2030"][emis_i]
                    for cond_i, cond in enumerate(["unconditional", "conditional"]):
                        condkey = ["Unconditional NDCs", "Conditional NDCs"][cond_i]
                        for ambition_i, ambition in enumerate(["min", "max"]):
                            add = ["", ".1"][ambition_i]
                            val = float(df_ndc_raw_sub[key][condkey + add])
                            red = 1 - val / val_2015
                            abs_jones = histemis_r * (1 - red)
                            df_regs.append(r)
                            df_amb.append(ambition)
                            df_con.append(cond)
                            df_emis.append(emis)
                            df_lulucf.append(lulucf)
                            df_red.append(red)
                            df_abs.append(abs_jones)
                            df_inv.append(val)

    dict_ndc = {
        "Region": df_regs,
        "Ambition": df_amb,
        "Conditionality": df_con,
        "GHG_ndc_red": df_red,
        "GHG_ndc": df_abs,
        "GHG_ndc_inv": df_inv,
    }
    df_ndc = pd.DataFrame(dict_ndc)
    xr_ndc = xr.Dataset.from_dataframe(df_ndc.set_index(["Region", "Ambition", "Conditionality"]))

    # Now for GHG excluding LULUCF
    df_ndc_raw = pd.read_excel(
        data_root / filename, sheet_name="Reduction All_GHG_excl", header=[0, 1]
    )
    regs = df_ndc_raw["(Mt CO2 equivalent)"]["Country name"]
    regs_iso = []
    for r in regs:
        wh = np.where(countries_name == r)[0]
        if len(wh) == 0:
            if r == "United States":
                regs_iso.append("USA")
            elif r == "EU27":
                regs_iso.append("EU")
            elif r == "Turkey":
                regs_iso.append("TUR")
            else:
                regs_iso.append(np.nan)
        else:
            regs_iso.append(countries_iso[wh[0]])
    regs_iso = np.array(regs_iso)
    df_ndc_raw["ISO"] = regs_iso

    df_regs = []
    df_amb = []
    df_con = []
    df_emis = []
    df_lulucf = []
    df_red = []
    df_abs = []
    df_inv = []
    histemis = xr_hist.GHG_hist.sel(Time=2015)
    for r in list(countries_iso) + ["EU"]:
        histemis_r = float(histemis.sel(Region=r))
        df_ndc_raw_sub = df_ndc_raw[df_ndc_raw["ISO"] == r]
        if len(df_ndc_raw_sub) > 0:
            val_2015 = float(df_ndc_raw_sub["(Mt CO2 equivalent)"][2015])
            for lulucf in ["incl"]:  # Maybe add excl later?
                for emis_i, emis in enumerate(["NDC"]):  # , 'CP']):
                    key = ["2030 NDCs", "Domestic actions 2030"][emis_i]
                    for cond_i, cond in enumerate(["unconditional", "conditional"]):
                        condkey = ["Unconditional NDCs", "Conditional NDCs"][cond_i]
                        for ambition_i, ambition in enumerate(["min", "max"]):
                            add = ["", ".1"][ambition_i]
                            val = float(df_ndc_raw_sub[key][condkey + add])
                            red = 1 - val / val_2015
                            abs_jones = histemis_r * (1 - red)
                            df_regs.append(r)
                            df_amb.append(ambition)
                            df_con.append(cond)
                            df_emis.append(emis)
                            df_lulucf.append(lulucf)
                            df_red.append(red)
                            df_abs.append(abs_jones)
                            df_inv.append(val)

    dict_ndc = {
        "Region": df_regs,
        "Ambition": df_amb,
        "Conditionality": df_con,
        "GHG_ndc_excl_red": df_red,
        "GHG_ndc_excl": df_abs,
        "GHG_ndc_excl_inv": df_inv,
    }
    df_ndc = pd.DataFrame(dict_ndc)
    xr_ndc_excl = xr.Dataset.from_dataframe(
        df_ndc.set_index(["Region", "Ambition", "Conditionality"])
    )

    return xr_ndc, xr_ndc_excl

def merge_xr(
    xr_ssp,
    xr_hist,
    xr_unp,
    xr_hdish,
    xr_co2_budgets,
    all_projected_gases,
    xr_base,
    xr_ndc,
    xr_ndc_excl,
    xr_ndc_CR,
    xr_ar6_C,
    xr_ar6_C_bunkers,
    regions,
):
    regions_iso = list(regions.values())
    print("- Merging xrarray object")
    xr_total = xr.merge(
        [
            xr_ssp,
            xr_hist,
            xr_unp,
            xr_hdish,
            xr_co2_budgets,
            all_projected_gases,
            xr_base,
            xr_ndc,
            xr_ndc_excl,
            xr_ndc_CR,
            xr_ar6_C,
            xr_ar6_C_bunkers,
        ]
    )
    xr_total = xr_total.reindex(Region=regions_iso)
    xr_total = xr_total.reindex(Time=np.arange(1850, 2101))
    xr_total["GHG_globe"] = xr_total["GHG_globe"].astype(float)
    xr_total = xr_total.interpolate_na(dim="Time", method="linear").drop_vars(["Variable"])
    return xr_total

def add_country_groups(config: Config, regions, xr_total):
    print("- Add country groups")

    data_root = config.paths.input
    filename = "UNFCCC_Parties_Groups_noeu.xlsx"
    regions_name = list(regions.keys())
    regions_iso = list(regions.values())

    df = pd.read_excel(data_root / filename, sheet_name="Country groups")
    countries_iso = np.array(df["Country ISO Code"])
    list_of_regions = list(np.array(regions_iso).copy())
    reg_iso = regions_iso.copy()
    reg_name = regions_name.copy()
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
        for cty in np.array(xr_total.Region):
            if cty in group_indices:
                country_to_eu[cty] = [group_of_choice]
            else:
                country_to_eu[cty] = [""]
        group_coord = xr.DataArray(
            [group for country in np.array(xr_total["Region"]) for group in country_to_eu[country]],
            dims=["Region"],
            coords={
                "Region": [
                    country
                    for country in np.array(xr_total["Region"])
                    for group in country_to_eu[country]
                ]
            },
        )
        if group_of_choice == "EU":
            xr_eu = (
                xr_total[
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
                xr_total[
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
        dummy = xr_total.reindex(Region=list_of_regions)

        new_total = xr.merge([dummy, xr_eu2])
        new_total = new_total.reindex(Region=list_of_regions)
        if group_of_choice not in ["EU", "EARTH"]:
            reg_iso.append(group_of_choice)
            reg_name.append(group_of_choice)

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

def save(config: Config, xr_total, regions, countries):
    print("- Save important files")

    savepath = config.paths.output / f"startyear_{config.params.start_year_analysis}"

    # TODO move to separate function?
    regions_name = np.array(list(regions.keys()))
    regions_iso = np.array(list(regions.values()))
    countries_name = np.array(list(countries.keys()))
    countries_iso = np.array(list(countries.values()))
    np.save(config.paths.output / "all_regions.npy", regions_iso)
    np.save(config.paths.output / "all_regions_names.npy", regions_name)
    np.save(config.paths.output / "all_countries.npy", countries_iso)
    np.save(config.paths.output / "all_countries_names.npy", countries_name)

    xr_normal = xr_total.sel(
        Temperature=np.array(config.dimension_ranges.peak_temperature_saved).astype(float).round(2)
    )
    xr_version = xr_normal


    xr_version.to_netcdf(
        savepath / "xr_dataread.nc",
        encoding={
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
        },
        format="NETCDF4",
        engine="netcdf4",
    )

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

    # TODO: move to separate function? This is reading new data
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
    xr_rci.to_netcdf(config.paths.output / "xr_rci.nc")


def country_specific_datareaders(config: Config, xr_total, xr_primap):
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
    xr_dataread_nld.sel(
        Temperature=np.array(config.dimension_ranges.peak_temperature_saved).astype(float).round(2)
    ).to_netcdf(
        savepath / "xr_dataread_NLD.nc",
        # TODO: make separate variable for encoding so we can reuse it?
        encoding={
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
            "GHG_ndc": {"zlib": True, "complevel": 9},
            "GHG_ndc_inv": {"zlib": True, "complevel": 9},
            "GHG_ndc_red": {"zlib": True, "complevel": 9},
            "GHG_ndc_excl": {"zlib": True, "complevel": 9},
            "GHG_ndc_excl_inv": {"zlib": True, "complevel": 9},
            "GHG_ndc_excl_red": {"zlib": True, "complevel": 9},
            "GHG_ndc_excl_CR": {"zlib": True, "complevel": 9},
        },
        format="NETCDF4",
        engine="netcdf4",
    )

    # TODO: move to seperate function?
    # Norwegian emissions - harmonized with EDGAR
    xr_dataread_nor = xr.open_dataset(savepath / "xr_dataread.nc").load().copy()
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
    xr_dataread_nor.sel(
        Temperature=np.array(config.dimension_ranges.peak_temperature_saved).astype(float).round(2)
    ).to_netcdf(
        savepath / "xr_dataread_NOR.nc",
        encoding={
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
            "GHG_ndc": {"zlib": True, "complevel": 9},
            "GHG_ndc_excl": {"zlib": True, "complevel": 9},
            "GHG_ndc_excl_CR": {"zlib": True, "complevel": 9},
        },
        format="NETCDF4",
        engine="netcdf4",
    )


def main(config_file):
    config = Config.from_file(config_file)

    general = read_general(config)  # TODO combine with un_population?
    xr_ssp = read_ssps(config, regions=general.regions)
    un_pop = read_un_population(config, countries=general.countries)
    xr_hdi, xr_hdish = read_hdi(config, general.countries, un_pop)
    jonesdata = read_historicalemis_jones(config, regions=general.regions)
    ar6data = read_ar6(config, xr_hist=jonesdata.xr_hist)
    nonco2data = nonco2variation(config)
    nonco2trajectories = determine_global_nonco2_trajectories(
        config, ar6data, jonesdata.xr_hist, nonco2data.xr_temperatures
    )
    globalbudgets = determine_global_budgets(config, jonesdata.xr_hist, nonco2data)
    global_co2_trajectories = determine_global_co2_trajectories(
        config,
        jonesdata.xr_hist,
        ar6data,
        nonco2data.xr_temperatures,
        globalbudgets.xr_co2_budgets,
        nonco2trajectories.xr_traj_nonco2,
    )
    xr_base = read_baseline(
        config,
        general.countries,
        jonesdata.xr_hist,
    )

    xr_ndc_CR = read_ndc_climateresource(config, general.countries)
    xr_ndc, xr_ndc_excl = read_ndc(config, general.countries, jonesdata.xr_hist)
    xr_total = merge_xr(
        xr_ssp,
        jonesdata.xr_hist,
        un_pop.population,
        xr_hdish,
        globalbudgets.xr_co2_budgets,
        global_co2_trajectories.all_projected_gases,
        xr_base,
        xr_ndc,
        xr_ndc_excl,
        xr_ndc_CR,
        ar6data.xr_ar6_C,
        ar6data.xr_ar6_C_bunkers,
        general.regions,
    )
    new_total, new_regions = add_country_groups(config, general.regions, xr_total)
    save(config, new_total, new_regions, general.countries)
    country_specific_datareaders(config, xr_total, jonesdata.xr_primap)


if __name__ == "__main__":
    import sys

    config_file = sys.argv[1]
    main(config_file)
