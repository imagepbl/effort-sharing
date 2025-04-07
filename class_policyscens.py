# ======================================== #
# Class that adds the policy scenarios from ENGAGE
# ======================================== #

# =========================================================== #
# PREAMBULE
# Put in packages that we need
# =========================================================== #

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

# =========================================================== #
# CLASS OBJECT
# =========================================================== #


class policyscenadding(object):
    """
    Class that adds the policy scenarios from ELEVATE to xr_total
    """

    # =========================================================== #
    # =========================================================== #

    def __init__(self):
        print("# ==================================== #")
        print("# Initializing policyscenadding class  #")
        print("# ==================================== #")

        self.current_dir = Path.cwd()
        self.df_scenarios_kyoto = None
        self.df_scenarios_co2 = None
        self.xr_eng = None
        self.xr_eng_co2 = None
        self.xr_total_co2 = None

        # Scenario names
        self.scenarios = {
            "ELV-SSP2-CP-D0": "CurPol",
            "ELV-SSP2-CP-D0-N": "CurPol",
            "Current Policies": "CurPol",
            "ELV-SSP2-NDC-D0": "NDC",
            "ELV-SSP2-LTS": "NetZero",
        }

        # Read in Input YAML file
        with open(self.current_dir / "input.yml", encoding="utf-8") as file:
            self.settings = yaml.load(file, Loader=yaml.FullLoader)
        self.xr_total = xr.open_dataset(
            self.settings["paths"]["data"]["datadrive"]
            + "/startyear_2021/xr_dataread.nc"
        )

    # =========================================================== #
    # =========================================================== #

    def read_filter_scenario_data(self):
        """
        Read in the ELEVATE data and filter for relevant scenarios and variables.
        Data can be downloaded from: https://zenodo.org/records/15114066
        """

        print("- Read ELEVATE scenarios and filter for relevant data")

        # Read the raw data
        df_scenarios_raw = pd.read_csv(
            self.settings["paths"]["data"]["external"]
            + "/ELEVATE/ELEVATE_Data_D2.3_vetted_20250211.csv",
            header=0,
        )

        # Filter for scenarios and variables
        variables = ["Emissions|Kyoto Gases", "Emissions|CO2"]

        df_scenarios_filtered = df_scenarios_raw[
            df_scenarios_raw.Scenario.isin(self.scenarios.keys())
            & df_scenarios_raw.Variable.isin(variables)
        ].copy()
        df_scenarios_filtered = df_scenarios_filtered.reset_index(drop=True)

        return df_scenarios_filtered

    def rename_and_preprocess(self, df_scenarios_filtered):
        """
        Rename columns, regions and scenarios
        """

        # Rename columns: Remove leading 'X' from year columns
        df_scenarios_filtered.columns = [
            col[1:] if col.startswith("X") and col[1:].isdigit() else col
            for col in df_scenarios_filtered.columns
        ]

        # Rename scenarios
        df_scenarios_filtered["Scenario"] = df_scenarios_filtered["Scenario"].replace(
            self.scenarios
        )

        # Rename regions
        region_mapping = {
            "World": "EARTH",
            "United States of America": "USA",
            "South-East Asia": "Southeast Asia",
            "South East Asia": "Southeast Asia",
        }
        df_scenarios_filtered["Region"] = df_scenarios_filtered["Region"].replace(
            region_mapping
        )

        df_scenarios_renamed = df_scenarios_filtered.copy()

        return df_scenarios_renamed

    # =========================================================== #
    # =========================================================== #

    def deduplicate_regions(self, df_scenarios_renamed):
        """
        Some regions are written as model|region, some only as region. Models often reported
        both versions, but these are often duplicates and need to be removed.
        More info on the AR9 and AR10 regions here:
        https://github.com/IAMconsortium/common-definitions/blob/main/definitions/region/common.yaml
        """
        # Split the region column by '|' and expand into new columns
        split_columns = df_scenarios_renamed["Region"].str.split("|", expand=True)
        split_columns.columns = ["Model_2", "Region_2"]

        # Add the new columns to the original DataFrame
        df_scenarios_renamed = pd.concat([df_scenarios_renamed, split_columns], axis=1)

        # If a region was Model|Region, we don't need the model name twice so replace with NaN
        df_scenarios_renamed["Model_2"] = np.where(
            df_scenarios_renamed["Model_2"] == df_scenarios_renamed["Model"],
            np.nan,
            df_scenarios_renamed["Model_2"],
        )

        # Merge the data on region into a new column 'Region_cleaned'
        df_scenarios_renamed["Region_cleaned"] = df_scenarios_renamed[
            "Model_2"
        ].combine_first(df_scenarios_renamed["Region_2"])

        # Sort the dataframe by 'Region_cleaned' and reset the index
        # Sorting it to give preference to e.g. "India" over "India (AR10)"
        df_scenarios_renamed.sort_values(by=["Region_cleaned"], inplace=True)
        df_scenarios_renamed.reset_index(drop=True, inplace=True)

        # Add a new column 'Is_Duplicate' to indicate subsequent duplicates
        df_scenarios_renamed["Is_Duplicate"] = df_scenarios_renamed.duplicated(
            subset=["Model", "Scenario", "Variable", "2025", "2100"], keep="first"
        )

        # Remove all rows that are Is_Duplicate = True
        df_scenarios_deduplicated = df_scenarios_renamed[
            ~df_scenarios_renamed["Is_Duplicate"]
        ]

        # TODO: Adapt in the future
        # For GEM-E3_V2023 there are several India results
        # Becomes an issue later when converting to xarray so removing for now
        df_scenarios_deduplicated = df_scenarios_deduplicated[
            ~(
                (df_scenarios_deduplicated["Model"] == "GEM-E3_V2023")
                & (df_scenarios_deduplicated["Region_cleaned"] == "India")
            )
        ]

        # Drop helper columns and reorder the DataFrame
        df_scenarios_deduplicated.drop(
            columns=["Model_2", "Region_2", "Is_Duplicate", "Region"], inplace=True
        )
        df_scenarios_deduplicated.rename(
            columns={"Region_cleaned": "Region"}, inplace=True
        )

        # Reorder the columns
        columns_to_keep = [
            "Model",
            "Scenario",
            "Region",
            "Variable",
            "Unit",
        ] + [col for col in df_scenarios_deduplicated.columns if col.isdigit()]
        df_scenarios_deduplicated = df_scenarios_deduplicated[columns_to_keep]

        return df_scenarios_deduplicated

    def format_to_xarray(self, df_co2_or_kyoto):
        """
        Convert a DataFrame to an xarray object
        """
        # Melt the DataFrame to long format
        df_co2_or_kyoto.drop_duplicates(inplace=True)

        df_melted = df_co2_or_kyoto.melt(
            id_vars=["Scenario", "Model", "Region"],
            var_name="Time",
            value_name="Value",
        )

        # Convert the 'Time' column to integers
        df_melted["Time"] = np.array(df_melted["Time"].astype(int))

        # Set the index for the xarray object
        df_melted.set_index(["Scenario", "Model", "Region", "Time"], inplace=True)

        df_melted = df_melted.drop_duplicates()
        duplicates = df_melted[df_melted.duplicated()]
        if not duplicates.empty:
            print("Duplicate rows found:")
            print(duplicates)
        # Convert to xarray Dataset
        xr_dataset = xr.Dataset.from_dataframe(df_melted)
        xr_dataset = xr_dataset.reindex(Time=np.arange(1850, 2101))

        return xr_dataset.interpolate_na(dim="Time", method="linear")

    def filter_and_convert(self, df_scenarios_deduplicated):
        """
        Split the dataframe into co2 and kyoto gas and convert to xarray objects
        """
        print("- Split dataframe and convert to xarray object")

        # Split df_scenarios_deduplicated into two DataFrames
        df_scenarios_co2 = df_scenarios_deduplicated[
            df_scenarios_deduplicated["Variable"] == "Emissions|CO2"
        ].copy()
        df_scenarios_kyoto = df_scenarios_deduplicated[
            df_scenarios_deduplicated["Variable"] == "Emissions|Kyoto Gases"
        ].copy()

        # Drop the 'Variable' column from both DataFrames
        df_scenarios_co2.drop(columns=["Variable", "Unit"], inplace=True)
        df_scenarios_kyoto.drop(columns=["Variable", "Unit"], inplace=True)
        df_scenarios_co2.reset_index(drop=True, inplace=True)
        df_scenarios_kyoto.reset_index(drop=True, inplace=True)

        # Convert to xarray objects
        xr_kyoto = self.format_to_xarray(df_scenarios_kyoto)
        xr_co2 = self.format_to_xarray(df_scenarios_co2)

        return xr_kyoto, xr_co2

    # =========================================================== #
    # =========================================================== #

    def add_to_xr(self, xr_kyoto, xr_co2):
        """'
        Add the policy scenarios to the xarray object'
        """
        print("- Add to overall xrobject")
        xr_total = self.xr_total.assign(NDC=xr_kyoto["Value"].sel(Scenario="NDC"))
        xr_total = xr_total.assign(CurPol=xr_kyoto["Value"].sel(Scenario="CurPol"))
        xr_total = xr_total.assign(NetZero=xr_kyoto["Value"].sel(Scenario="NetZero"))
        xr_total = xr_total.reindex(Time=np.arange(1850, 2101))
        self.xr_total = xr_total.interpolate_na(dim="Time", method="linear")
        xr_total_onlyalloc = self.xr_total[["NDC", "CurPol", "NetZero"]]
        xr_total_onlyalloc.to_netcdf(
            self.settings["paths"]["data"]["datadrive"] + "xr_policyscen.nc"
        )

        # CO2 version
        xr_total2 = self.xr_total.assign(NDC=xr_co2["Value"].sel(Scenario="NDC"))
        xr_total2 = xr_total2.assign(CurPol=xr_co2["Value"].sel(Scenario="CurPol"))
        xr_total2 = xr_total2.assign(NetZero=xr_co2["Value"].sel(Scenario="NetZero"))
        xr_total2 = xr_total2.reindex(Time=np.arange(1850, 2101))
        self.xr_total_co2 = xr_total2.interpolate_na(dim="Time", method="linear")
        xr_total_onlyalloc_co2 = self.xr_total_co2[["NDC", "CurPol", "NetZero"]]
        xr_total_onlyalloc_co2.to_netcdf(
            self.settings["paths"]["data"]["datadrive"] + "xr_policyscen_co2.nc"
        )

        self.xr_total.close()


if __name__ == "__main__":
    # Create an instance of the class
    policyscen = policyscenadding()

    # Call the methods in the class
    df_filtered = policyscen.read_filter_scenario_data()
    df_renamed = policyscen.rename_and_preprocess(df_filtered)
    df_deduplicated = policyscen.deduplicate_regions(df_renamed)
    xr_kyoto, xr_co2 = policyscen.filter_and_convert(df_deduplicated)
    policyscen.add_to_xr(xr_kyoto, xr_co2)
