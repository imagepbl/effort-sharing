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

from effortsharing.config import Config
import effortsharing.regions as _regions

# Set up logging
logger = logging.getLogger(__name__)


def read_ssps_refactor(config: Config, regions):
    """Read GDP and population data from SSPs."""
    logger.info("Reading GDP and population data from SSPs")

    # Define input
    data_root = config.paths.input
    filename = "SSPs_v2023.xlsx"

    # Read data
    df = pd.read_excel(data_root / filename, sheet_name="data")

    # Filter for relevant models
    df = df[(df.Model.isin(["OECD ENV-Growth 2023", "IIASA-WiC POP 2023"]))].drop(
        columns=["Model", "Unit"]
    )

    # Convert year columns into row indexes
    melted = df.melt(id_vars=["Scenario", "Region", "Variable"], var_name="Time")
    melted["Time"] = melted["Time"].astype(int)

    # Transform to xarray dataset
    ds = melted.pivot(
        index=["Scenario", "Region", "Time"], columns="Variable", values="value"
    ).to_xarray()

    # Split historical from future scenarios
    hist = ds.sel(Scenario="Historical Reference", Time=slice(1980, 2020))
    ds = ds.drop_sel(Scenario="Historical Reference")

    # Substitute historical data into the corresponding years of each scenario
    historical_expanded = hist.expand_dims(Scenario=ds.Scenario)
    ds.loc[dict(Time=slice(1980, 2020))] = historical_expanded

    # Rename variable
    ds = ds.rename_vars({"GDP|PPP": "GDP"})

    # Replace region names with ISO codes
    region_lookup_table = {**regions, **_regions.ADDITIONAL_REGIONS_SSPS}
    ds["Region"] = list(map(lambda name: region_lookup_table.get(name, "oeps"), ds.Region.values))
    ds = ds.sortby(ds.Region)

    return ds


# TODO: check why different from refactored version; fix; remove.
def read_ssps(config, regions):
    logger.info("Reading GDP and population data from SSPs")

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
                logger.warning(f"region not found: {r}")
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


def read_hdi_refactor(config, countries, population_long):
    logger.info("Read Human Development Index data")

    # Define input
    data_root = config.paths.input
    hdi_file = "HDR21-22_Statistical_Annex_HDI_Table.xlsx"

    df = pd.read_excel(data_root / hdi_file, sheet_name="Rawdata")

    # Convert missing data to NaN
    df.loc[df.HDI == "..", "HDI"] = np.nan

    # Convert country to region ISO codes
    def get_iso(x):
        extended_countries = {**countries, **_regions.ADDITIONAL_REGIONS_HDI}
        return extended_countries.get(x, "unknown")

    df["Region"] = df.Country.map(get_iso)
    df = df[~(df.Region == "unknown")]

    # Prepare for conversion to xarray
    df = df.drop(columns="Country").set_index("Region")

    # Insert NaN countries (TODO: I think we could just skip this??)
    # fmt: off
    nan_countries = [
        "ALA", "ASM", "AIA", "ABW", "BMU", "ANT", "SCG", "BES", "BVT", "IOT", "VGB",
        "CYM", "CXR", "CCK", "COK", "CUW", "FLK", "FRO", "GUF", "PYF", "ATF", #"GMB",
        "GIB", "GRL", "GLP", "GUM", "GGY", "HMD", "VAT", "IMN", "JEY", "MAC", "MTQ",
        "MYT", "MSR", "NCL", "NIU", "NFK", "MNP", "PCN", "PRI", "REU", "BLM", "SHN",
        "SPM", "SXM", "SGS", "MAF", "SJM", "TKL", "TCA", "UMI", "VIR", "WLF", "ESH",
    ]
    # fmt: on
    df = pd.concat(
        [
            df,
            pd.Series(index=nan_countries, name="HDI").rename_axis("Region"),
        ]
    )

    # Convert to xarray
    hdi = df.to_xarray().dropna("Region")

    # Add hdi_sh
    pop_2019 = population_long.sel(Time=2019).Population.drop("Time")
    with xr.set_options(arithmetic_join="outer"):
        # "outer join" ensures that all regions are kept, even if they are
        # missing in one of the terms (data will be NaN)
        hdi_sh = (hdi.HDI / hdi.HDI.sum() * pop_2019).to_dataset(name="HDIsh")

    # TODO: add NaN entries for ISO codes from countries that are not available in HDI data
    return hdi, hdi_sh


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

    # fmt: off
    nan_countries = [
        "ALA", "ASM", "AIA", "ABW", "BMU", "ANT", "SCG", "BES", "BVT", "IOT", "VGB",
        "CYM", "CXR", "CCK", "COK", "CUW", "FLK", "FRO", "GUF", "PYF", "ATF", "GMB",
        "GIB", "GRL", "GLP", "GUM", "GGY", "HMD", "VAT", "IMN", "JEY", "MAC", "MTQ",
        "MYT", "MSR", "NCL", "NIU", "NFK", "MNP", "PCN", "PRI", "REU", "BLM", "SHN",
        "SPM", "SXM", "SGS", "MAF", "SJM", "TKL", "TCA", "UMI", "VIR", "WLF", "ESH",
    ]
    # fmt: on

    # Construct new hdi object
    countries_name = list(countries.keys())
    countries_iso = list(countries.values())

    hdi_values = np.zeros(len(countries_iso)) + np.nan
    hdi_sh_values = np.zeros(len(countries_iso)) + np.nan

    for r_i, iso in enumerate(countries_iso):
        name = countries_name[r_i]
        value = np.where(hdi_countries_raw == name)[0]

        if len(value) > 0:
            wh_i = value[0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif iso in nan_countries:
            hdi_values[r_i] = np.nan
        elif iso == "USA":
            value = np.where(hdi_countries_raw == "United States")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif iso == "BHS":
            value = np.where(hdi_countries_raw == "Bahamas")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif iso == "GMB":
            value = np.where(hdi_countries_raw == "Gambia")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif iso == "CPV":
            value = np.where(hdi_countries_raw == "Cabo Verde")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif iso == "BOL":
            value = np.where(hdi_countries_raw == "Bolivia (Plurinational State of)")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif iso == "COD":
            value = np.where(hdi_countries_raw == "Congo (Democratic Republic of the)")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif iso == "COG":
            value = np.where(hdi_countries_raw == "Congo")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif iso == "CZE":
            value = np.where(hdi_countries_raw == "Czechia")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif iso == "EGY":
            value = np.where(hdi_countries_raw == "Egypt")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif iso == "HKG":
            value = np.where(hdi_countries_raw == "Hong Kong, China (SAR)")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif iso == "IRN":
            value = np.where(hdi_countries_raw == "Iran (Islamic Republic of)")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif iso == "PRK":
            value = np.where(hdi_countries_raw == "Korea (Democratic People's Rep. of)")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif iso == "KOR":
            value = np.where(hdi_countries_raw == "Korea (Republic of)")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif iso == "KGZ":
            value = np.where(hdi_countries_raw == "Kyrgyzstan")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif iso == "LAO":
            value = np.where(hdi_countries_raw == "Lao People's Democratic Republic")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif iso == "FSM":
            value = np.where(hdi_countries_raw == "Micronesia (Federated States of)")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif iso == "MDA":
            value = np.where(hdi_countries_raw == "Moldova (Republic of)")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif iso == "STP":
            value = np.where(hdi_countries_raw == "Sao Tome and Principe")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif iso == "SVK":
            value = np.where(hdi_countries_raw == "Slovakia")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif iso == "KNA":
            value = np.where(hdi_countries_raw == "Saint Kitts and Nevis")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif iso == "LCA":
            value = np.where(hdi_countries_raw == "Saint Lucia")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif iso == "VCT":
            value = np.where(hdi_countries_raw == "Saint Vincent and the Grenadines")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif iso == "SWZ":
            value = np.where(hdi_countries_raw == "Eswatini (Kingdom of)")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif iso == "TWN":
            value = np.where(hdi_countries_raw == "China")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif iso == "TZA":
            value = np.where(hdi_countries_raw == "Tanzania (United Republic of)")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif iso == "TUR":
            value = np.where(hdi_countries_raw == "Türkiye")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif iso == "VEN":
            value = np.where(hdi_countries_raw == "Venezuela (Bolivarian Republic of)")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif iso == "VNM":
            value = np.where(hdi_countries_raw == "Viet Nam")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif iso == "PSE":
            value = np.where(hdi_countries_raw == "Palestine, State of")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        elif iso == "YEM":
            value = np.where(hdi_countries_raw == "Yemen")[0][0]
            hdi_values[r_i] = hdi_values_raw[wh_i]
        else:
            hdi_values[r_i] = np.nan

        try:
            pop = float(population_long.sel(Region=iso, Time=2019).Population)
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


def process_socioeconomics(config: Config, save=True):
    """Collect socio-economic input data from various sources to intermediate file."""

    logger.info("Processing socio-economic input data")

    countries, regions = _regions.read_general(config)

    ssps = read_ssps(config, regions)
    population, population_long = read_un_population(config, countries)
    hdi, hdi_sh = read_hdi(config, countries, population_long)

    # Merge datasets
    socioeconomic_data = xr.merge([ssps, population, hdi_sh])
    # TODO: Reindex time and regions??

    # Save to disk
    if save:
        save_path = config.paths.intermediate / "socioeconomics.nc"

        logger.info(f"Saving socio-economic data to {save_path}")

        config.paths.intermediate.mkdir(parents=True, exist_ok=True)
        socioeconomic_data.to_netcdf(save_path)

    return socioeconomic_data


if __name__ == "__main__":
    import argparse

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Get the config file from command line arguments
    parser = argparse.ArgumentParser(description="Process socio-economic input data")
    parser.add_argument("config", help="Path to config file")
    args = parser.parse_args()

    # Read config
    config = Config.from_file(args.config)

    # Process socio-economic data
    process_socioeconomics(config)
