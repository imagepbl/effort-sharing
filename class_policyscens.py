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

        # Read in Input YAML file
        with open(self.current_dir / "input.yml", encoding="utf-8") as file:
            self.settings = yaml.load(file, Loader=yaml.FullLoader)
        self.xr_total = xr.open_dataset(
            self.settings["paths"]["data"]["datadrive"]
            + "/startyear_2021/xr_dataread.nc"
        )

    # =========================================================== #
    # =========================================================== #


    def rename_regions(self, df, column_name="Region"):
        """
        Helper method to rename regions in a DataFrame.
        """
        region_mapping = {
            "Argentine Republic": "ARG",
            "Canada": "CAN",
            "Commonwealth of Australia": "AUS",
            "Federative Republic of Brazil": "BRA",
            "People's Repulic of China": "CHN",
            "European Union (28 member countries)": "EU",
            "Republic of India": "IND",
            "Republic of Indonesia": "IDN",
            "State of Japan": "JPN",
            "Russian Federation": "RUS",
            "Kingdom of Saudi Arabia": "SAU",
            "Republic of South Africa": "ZAF",
            "Republic of Korea (South Korea)": "KOR",
            "United Mexican States": "MEX",
            "Republic of Turkey": "TUR",
            "United States of America": "USA",
            "Viet Nam ": "VNM",
        }
        df[column_name] = df[column_name].replace(region_mapping)
        return df

    def read_scenario_data(self):
        """
        Read in the ELEVATE data and change region names to match names used in the model
        Data can be downloaded from: https://zenodo.org/records/15114066
        """

        print("- Read ELEVATE scenarios and change region namings")
        # Read the raw data
        df_scenarios_raw = pd.read_csv(
            self.settings["paths"]["data"]["external"]
            + "/ELEVATE/ELEVATE_scenarios_2025_emis_only.csv", sep=";", header=0
        )

        # Rename regions for the entire DataFrame
        df_scenarios_raw = self.rename_regions(df_scenarios_raw, column_name="Region")

        # Process Kyoto Gases
        df_scenarios_kyoto = df_scenarios_raw[df_scenarios_raw.Variable == "Emissions|Kyoto Gases"]
        df_scenarios_kyoto = df_scenarios_kyoto.reset_index(drop=True)
        self.df_scenarios_kyoto = df_scenarios_kyoto

        # Process CO2 Emissions
        df_scenarios_co2 = df_scenarios_raw[df_scenarios_raw.Variable == "Emissions|CO2"]
        df_scenarios_co2 = df_scenarios_co2.reset_index(drop=True)
        self.df_scenarios_co2 = df_scenarios_co2

    # =========================================================== #
    # =========================================================== #

    def filter_and_convert(self):
        """
        Filter the scenarios and convert to xarray object
        """
        print("- Filter correct scenarios and convert to xarray object")
        curpol = "ELV-SSP2-CP-D0"
        ndc = "ELV-SSP2-NDC-D0"
        nz = "ELV-SSP2-LTS"

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
    policyscen.read_scenario_data()
    policyscen.filter_and_convert()
    policyscen.add_to_xr()
