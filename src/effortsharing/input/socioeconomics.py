"""
Functions to read and process socio-economic input data from various sources.

Import as library:

    from effortsharing.input import socioeconomics


Or use as standalone script:

    python src/effortsharing/input/socioeconomics.py config.yml


"""

import logging

import numpy as np
import pandas as pd
import xarray as xr

from effortsharing.cache import intermediate_file
from effortsharing.config import Config

# Set up logging
logger = logging.getLogger(__name__)


def read_general(config: Config):
    """Read country names and ISO from UNFCCC table."""
    logger.info("Reading unfccc country data")

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
    regions = {**countries, "European Union": "EU", "Earth": "EARTH"}

    return countries, regions


@intermediate_file("ssps.nc")
def read_ssps(config, regions, countries):
    logger.info("Reading GDP and population data from SSPs")

    # Define input
    data_root = config.paths.input
    filename = "SSPs_v2023.xlsx"

    countries_name = np.array(list(countries.keys()))
    countries_iso = np.array(list(countries.values()))
    regions_name = np.array(list(regions.keys()))
    regions_iso = np.array(list(regions.values()))

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

        for r_i, r in enumerate(region_full):
            wh = np.where(regions_name == r)[0]
            if len(wh) > 0:
                iso = countries_iso[wh[0]]
            elif r == "Aruba":
                iso = "ABW"
            elif r == "Bahamas":
                iso = "BHS"
            elif r == "Democratic Republic of the Congo":
                iso = "COD"
            elif r == "Cabo Verde":
                iso = "CPV"
            elif r == "C?te d'Ivoire":
                iso = "CIV"
            elif r == "Western Sahara":
                iso = "ESH"
            elif r == "Gambia":
                iso = "GMB"
            elif r == "Czechia":
                iso = "CZE"
            elif r == "French Guiana":
                iso = "GUF"
            elif r == "Guam":
                iso = "GUM"
            elif r == "Hong Kong":
                iso = "HKG"
            elif r == "Iran":
                iso = "IRN"
            elif r == "Macao":
                iso = "MAC"
            elif r == "Moldova":
                iso = "MDA"
            elif r == "Mayotte":
                iso = "MYT"
            elif r == "New Caledonia":
                iso = "NCL"
            elif r == "Puerto Rico":
                iso = "PRI"
            elif r == "French Polynesia":
                iso = "PYF"
            elif r == "Turkey":
                iso = "TUR"
            elif r == "Taiwan":
                iso = "TWN"
            elif r == "Tanzania":
                iso = "TZA"
            elif r == "United States":
                iso = "USA"
            elif r == "United States Virgin Islands":
                iso = "VIR"
            elif r == "Viet Nam":
                iso = "VNM"
            elif r == "Cura?ao":
                iso = "CUW"
            elif r == "Guadeloupe":
                iso = "GLP"
            elif r == "Martinique":
                iso = "MTQ"
            elif r == "Palestine":
                iso = "PSE"
            elif r == "R?union":
                iso = "REU"
            elif r == "Syria":
                iso = "SYR"
            elif r == "Venezuela":
                iso = "VEN"
            elif r == "World":
                iso = "EARTH"
            else:
                logger.warning(r)
                iso = "oeps"
            region_iso.append(iso)
        df_ssp["Region"] = region_iso
        Variable = np.array(df_ssp["Variable"])
        Variable[Variable == "GDP|PPP"] = "GDP"
        df_ssp["Variable"] = Variable
        df_ssp = df_ssp.drop(["Model", "Unit"], axis=1)
        dummy = df_ssp.melt(
            id_vars=["Scenario", "Region", "Variable"], var_name="Time", value_name="Value"
        )
        dummy["Time"] = np.array(dummy["Time"].astype(int))
        if i >= 1:
            dummy["Scenario"] = ["SSP" + str(i)] * len(dummy)
            xr_hist_gdp_i = xr.Dataset.from_dataframe(
                dummy.pivot(
                    index=["Scenario", "Region", "Time"], columns="Variable", values="Value"
                )
            ).sel(Time=[1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015])
            xr_ssp = xr.merge([xr_ssp, xr_hist_gdp_i])
        else:
            xr_ssp = (
                xr.Dataset.from_dataframe(
                    dummy.pivot(
                        index=["Scenario", "Region", "Time"], columns="Variable", values="Value"
                    )
                )
                .reindex({"Time": np.arange(2020, 2101, 5)})
                .reindex({"Time": np.arange(1980, 2101, 5)})
            )

    xr_ssp['Region'] = xr_ssp.Region.astype(str)
    return xr_ssp


def read_un_population(config, countries):
    logger.info("Reading UN population data and gapminder, processed by OWID (for past population)")

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
    return xr_unp, xr_unp_long


def read_hdi(config, countries, population_long):
    logger.info("Read Human Development Index data")

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
            pop = float(population_long.sel(Region=r, Time=2019).Population)
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


@intermediate_file("socioeconomics.nc")
def load_socioeconomics(config: Config):
    """Collect socio-economic input data from various sources to intermediate file.

    Args:
        config: effortsharing.config.Config object

    Returns:
        xarray.Dataset: Socio-economic data
    """
    countries, regions = read_general(config)

    ssps = read_ssps(config, regions, countries)
    population, population_long = read_un_population(config, countries)
    hdi, hdi_sh = read_hdi(config, countries, population_long)

    # Merge datasets
    socioeconomic_data = xr.merge([ssps, population, hdi_sh])
    # TODO: Reindex time and regions??


    return socioeconomic_data


if __name__ == "__main__":
    import argparse

    from rich.logging import RichHandler

    # Set up logging
    logging.basicConfig(level="INFO", format="%(message)s", handlers=[RichHandler(show_time=False)])

    # Get the config file from command line arguments
    parser = argparse.ArgumentParser(description="Process socio-economic input data")
    parser.add_argument("config", help="Path to config file")
    args = parser.parse_args()

    # Read config
    config = Config.from_file(args.config)

    # Process socio-economic data and save to intermediate file
    # Note: `load_intermediate` argument is added by the decorator
    load_socioeconomics(config)
