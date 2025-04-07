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

        return df_scenarios_filtered

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

        # Remove all rows that are Is_Duplicated = True
        df_scenarios_deduplicated = df_scenarios_renamed[
            ~df_scenarios_renamed["Is_Duplicate"]
        ]

        # Drop helper columns and reorder the DataFrame
        df_scenarios_deduplicated.drop(
            columns=["Model_2", "Region_2", "Is_Duplicated", "Region"], inplace=True
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

    def filter_and_convert(self):
        """
        Filter the scenarios and convert to xarray object
        """
        print("- Filter correct scenarios and convert to xarray object")
        # curpol = "ELV-SSP2-CP-D0"
        # ndc = "ELV-SSP2-NDC-D0"
        # nz = "ELV-SSP2-LTS"

        # # Filter for relevant scenarios
        # relevant_scenarios = [curpol, ndc, nz]
        # df_filtered = self.df_scenarios_kyoto.append(
        #     self.df_scenarios_co2, ignore_index=True
        # )
        # df_filtered = df_filtered[df_filtered.Scenario.isin(relevant_scenarios)].copy()

        # # Rename scenarios and regions
        # df_filtered["Scenario"] = df_filtered["Scenario"].replace(
        #     {curpol: "CurPol", ndc: "NDC", nz: "NetZero"}
        # )
        # df_filtered["Region"] = df_filtered["Region"].replace({"World": "EARTH"})

        ###

        df_eng_ref = self.df_eng[
            ["Model", "Scenario", "Region"] + list(self.df_eng.keys()[5:])
        ]
        df_eng_ref = df_eng_ref[df_eng_ref.Scenario.isin([curpol, ndc, nz])]
        scen = np.array(df_eng_ref.Scenario)
        scen[scen == ndc] = "NDC"
        scen[scen == curpol] = "CurPol"
        scen[scen == nz] = "NetZero"
        reg = np.array(df_eng_ref.Region)
        reg[reg == "World"] = "EARTH"
        df_eng_ref["Scenario"] = scen
        df_eng_ref["Region"] = reg
        dummy = df_eng_ref.melt(
            id_vars=["Scenario", "Model", "Region"], var_name="Time", value_name="Value"
        )
        dummy["Time"] = np.array(dummy["Time"].astype(int))
        dummy = dummy.set_index(["Scenario", "Model", "Region", "Time"])
        xr_eng = xr.Dataset.from_dataframe(dummy)
        xr_eng = xr_eng.reindex(Time=np.arange(1850, 2101))
        self.xr_eng = xr_eng.interpolate_na(dim="Time", method="linear")

        # CO2 version
        df_eng_ref_co2 = self.df_eng_co2[
            ["Model", "Scenario", "Region"] + list(self.df_eng_co2.keys()[5:])
        ]
        df_eng_ref_co2 = df_eng_ref_co2[df_eng_ref_co2.Scenario.isin([curpol, ndc, nz])]
        scen = np.array(df_eng_ref_co2.Scenario)
        scen[scen == ndc] = "NDC"
        scen[scen == curpol] = "CurPol"
        scen[scen == nz] = "NetZero"
        reg = np.array(df_eng_ref_co2.Region)
        reg[reg == "World"] = "EARTH"
        df_eng_ref_co2["Scenario"] = scen
        df_eng_ref_co2["Region"] = reg
        dummy = df_eng_ref_co2.melt(
            id_vars=["Scenario", "Model", "Region"], var_name="Time", value_name="Value"
        )
        dummy["Time"] = np.array(dummy["Time"].astype(int))
        dummy = dummy.set_index(["Scenario", "Model", "Region", "Time"])
        xr_eng_co2 = xr.Dataset.from_dataframe(dummy)
        xr_eng_co2 = xr_eng_co2.reindex(Time=np.arange(1850, 2101))
        self.xr_eng_co2 = xr_eng_co2.interpolate_na(dim="Time", method="linear")

    # =========================================================== #
    # =========================================================== #

    def add_to_xr(self):
        """'
        Add the policy scenarios to the xarray object'
        """
        print("- Add to overall xrobject")
        xr_total = self.xr_total.assign(NDC=self.xr_eng["Value"].sel(Scenario="NDC"))
        xr_total = xr_total.assign(CurPol=self.xr_eng["Value"].sel(Scenario="CurPol"))
        xr_total = xr_total.assign(NetZero=self.xr_eng["Value"].sel(Scenario="NetZero"))
        xr_total = xr_total.reindex(Time=np.arange(1850, 2101))
        self.xr_total = xr_total.interpolate_na(dim="Time", method="linear")
        xr_total_onlyalloc = self.xr_total[["NDC", "CurPol", "NetZero"]]
        xr_total_onlyalloc.to_netcdf(
            self.settings["paths"]["data"]["datadrive"] + "xr_policyscen.nc"
        )

        # CO2 version
        xr_total2 = self.xr_total.assign(
            NDC=self.xr_eng_co2["Value"].sel(Scenario="NDC")
        )
        xr_total2 = xr_total2.assign(
            CurPol=self.xr_eng_co2["Value"].sel(Scenario="CurPol")
        )
        xr_total2 = xr_total2.assign(
            NetZero=self.xr_eng_co2["Value"].sel(Scenario="NetZero")
        )
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
    policyscen.read_filter_scenario_data()
    policyscen.filter_and_convert()
    policyscen.add_to_xr()
